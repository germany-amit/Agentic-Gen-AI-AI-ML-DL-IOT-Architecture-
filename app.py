import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import textwrap
from urllib.parse import quote
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="🌍 City↔Planet Orchestration (Real‑Time)", page_icon="🌍", layout="wide")

# =========================
# CONFIG / DEFAULTS
# =========================
DEFAULT_CITIES = {
    # name: (lat, lon, approx fallback population in millions, wikidata_id)
    "New York": (40.7128, -74.0060, 8.5, "Q60"),
    "Delhi": (28.6139, 77.2090, 30.0, "Q1353"),
    "Tokyo": (35.6895, 139.6917, 37.0, "Q1490"),
    "Paris": (48.8566, 2.3522, 11.0, "Q90"),
    "São Paulo": (-23.5505, -46.6333, 22.0, "Q174"),
    "London": (51.5074, -0.1278, 14.8, "Q84"),
}

SCENARIOS = ["Heatwave", "Flood", "Power Grid Stress", "Water Shortage", "Wildfire Smoke"]

# Planetary toy budget (transparent, tweakable)
PLANETARY_BUDGET_TPD_CO2 = 1_000_000  # tons/day (toy)
EMISSION_FACTOR_PER_UNMET_ENERGY = 0.15  # tons CO2 per energy unit unmet (toy)

TIMEOUT = 6  # seconds for external calls on free tiers

# =========================
# OPEN APIs (free, no key)
# =========================
@st.cache_data(ttl=60)
def get_open_meteo_hourly(lat: float, lon: float) -> Dict:
    """Open‑Meteo hourly (last 24h + next 24h) for temp, wind, humidity."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        "&past_hours=24&forecast_hours=24&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("hourly", {})
    except Exception:
        return {}

@st.cache_data(ttl=60)
def get_openaq_timeseries(lat: float, lon: float) -> pd.DataFrame:
    """OpenAQ PM2.5 measurements ~ last 24h (nearest station)."""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)
    url = (
        "https://api.openaq.org/v2/measurements?"
        f"coordinates={lat},{lon}&radius=20000&parameter=pm25&limit=200&"
        f"date_from={start.replace(microsecond=0).isoformat()}Z&date_to={end.replace(microsecond=0).isoformat()}Z&"
        "order_by=datetime&sort=asc"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        js = r.json()
        rows = []
        for it in js.get("results", []):
            ts = it.get("date", {}).get("utc")
            val = it.get("value")
            if ts is not None and val is not None:
                rows.append({"datetime": pd.to_datetime(ts), "pm25": float(val)})
        if rows:
            df = pd.DataFrame(rows).sort_values("datetime")
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["datetime", "pm25"]).astype({"datetime": "datetime64[ns]", "pm25": "float"})

@st.cache_data(ttl=600)
def get_wikipedia_summary(city: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(city)}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("extract", "")
    except Exception:
        pass
    return ""

@st.cache_data(ttl=1800)
def wikidata_population_country(qid: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (population_in_millions, country_label) via SPARQL."""
    query = f"""
    SELECT ?population ?countryLabel WHERE {{
      wd:{qid} wdt:P1082 ?population.
      wd:{qid} wdt:P17 ?country.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} ORDER BY DESC(?population) LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    try:
        r = requests.get(
            url,
            params={"query": query, "format": "json"},
            timeout=TIMEOUT,
            headers={"Accept": "application/sparql-results+json"},
        )
        r.raise_for_status()
        js = r.json()
        b = js.get("results", {}).get("bindings", [])
        if b:
            pop = float(b[0]["population"]["value"]) / 1_000_000.0
            country = b[0]["countryLabel"]["value"]
            return pop, country
    except Exception:
        pass
    return None, None

# =========================
# SIMULATION UTILITIES (toy but transparent)
# =========================

def base_demands(pop_m: float, temp_c: Optional[float]) -> Dict[str, float]:
    t = temp_c if temp_c is not None else 25.0
    energy = pop_m * (1.2 + t/35.0)        # higher temp => more cooling
    water = pop_m * (1.0 + max(0,(30 - t))/55.0)
    mobility = pop_m * 0.7
    return {"energy": energy, "water": water, "mobility": mobility}


def apply_scenario(d: Dict[str, float], scenario: str, pm25_now: Optional[float]) -> Dict[str, float]:
    d = d.copy()
    if scenario == "Heatwave":
        d["energy"] *= 1.35; d["water"] *= 1.10
    elif scenario == "Flood":
        d["mobility"] *= 1.5; d["water"] *= 1.35
    elif scenario == "Power Grid Stress":
        d["energy"] *= 1.55
    elif scenario == "Water Shortage":
        d["water"] *= 1.6
    elif scenario == "Wildfire Smoke":
        d["mobility"] *= 1.2; d["energy"] *= 1.1
    # Air quality penalty
    if pm25_now is not None:
        factor = 1.0 + min(0.20, float(pm25_now)/500.0)
        d["energy"] *= factor
        d["mobility"] *= factor
    return d


def supply_capacity(demand: Dict[str, float]) -> Dict[str, float]:
    supply = {}
    # Randomized capacity range to mimic infra variability
    supply["energy"]   = demand["energy"] * np.random.uniform(0.85, 1.05)
    supply["water"]    = demand["water"]  * np.random.uniform(0.75, 1.00)
    supply["mobility"] = demand["mobility"] * np.random.uniform(0.85, 1.15)
    return supply


def apply_interventions(supply: Dict[str, float], load_shed: float, water_restrict: float, mobility_reroute: float) -> Dict[str, float]:
    s = supply.copy()
    s["energy"]   *= (1.0 + load_shed/100.0 * 0.15)
    s["water"]    *= (1.0 + water_restrict/100.0 * 0.20)
    s["mobility"] *= (1.0 + mobility_reroute/100.0 * 0.25)
    return s


def unmet_and_emissions(demand: Dict[str, float], supply: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    unmet = {k: max(0.0, demand[k] - supply[k]) for k in demand}
    co2_tons = unmet["energy"] * EMISSION_FACTOR_PER_UNMET_ENERGY
    return unmet, co2_tons

# =========================
# AGENTIC AI (local, optional)
# =========================

def generate_agentic_summary(context: str) -> str:
    """Try FLAN‑T5‑small locally; fall back to rule‑based summary if model/unavailable."""
    try:
        from transformers import pipeline
        # Small, free, open‑source model. First run downloads ~300MB.
        pipe = pipeline("text2text-generation", model="google/flan-t5-small")
        prompt = (
            "Summarize the situation and write an analytical conclusion. "
            "Be concise, actionable, and note risks, constraints, and policy levers.

" + context
        )
        out = pipe(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
        return out
    except Exception:
        # Fallback: heuristic analysis
        return heuristic_summary(context)


def heuristic_summary(context: str) -> str:
    lines = [l.strip() for l in context.splitlines() if l.strip()]
    cities = [l for l in lines if l.startswith("City:")]
    risks = []
    actions = []
    for l in lines:
        if "PM2.5" in l:
            try:
                val = float(l.split("PM2.5=")[-1].split()[0])
                if val > 75: risks.append("Air quality hazardous; limit outdoor mobility, enable HVAC filtration.")
                elif val > 35: risks.append("Air quality moderate‑poor; notify sensitive groups.")
            except Exception:
                pass
        if "Unmet energy" in l:
            try:
                v = float(l.split("Unmet energy=")[-1].split(",")[0])
                if v > 2: actions.append("Initiate demand response and intercity transfers up to cap.")
            except Exception:
                pass
        if "Water (Demand vs Supply)" in l and " vs " in l:
            try:
                dv = float(l.split("Water (Demand vs Supply): ")[-1].split(" vs ")[0])
                sv = float(l.split(" vs ")[-1])
                if dv > sv * 1.2:
                    actions.append("Escalate water restrictions; prioritize hospitals.")
            except Exception:
                pass
    risks = list(dict.fromkeys(risks))
    actions = list(dict.fromkeys(actions))
    return (
        "Analytical Conclusion (fallback):
- Key risks: " + (", ".join(risks) if risks else "No acute risks detected.") +
        "
- Recommended actions: " + (", ".join(actions) if actions else "Maintain monitoring; keep interventions at current levels.")
    )

# =========================
# KNOWLEDGE GRAPH (PyVis)
# =========================

def build_kg_html(rows: List[Dict], scenario: str, interventions: Dict[str, float]) -> Optional[str]:
    try:
        from pyvis.network import Network
        net = Network(height="560px", width="100%", bgcolor="#0b1020", font_color="#f3f4f6")
        net.barnes_hut(gravity=-25000, central_gravity=0.25, spring_length=150, spring_strength=0.02)
        theme = {
            "city": "#60a5fa", "signal": "#fbbf24", "resource": "#34d399",
            "risk": "#f87171", "govern": "#c084fc", "scenario": "#f472b6"
        }
        net.add_node("Scenario", label=f"Scenario: {scenario}", color=theme["scenario"], shape="box")
        net.add_node("Governance", label="Planetary Governance", color=theme["govern"], shape="box")
        # Interventions node
        net.add_node("Interventions", label=f"Interventions
Load {interventions['load']}%
Water {interventions['water']}%
Mob {interventions['mob']}%", color="#22d3ee", shape="box")
        net.add_edge("Scenario", "Interventions", color="#93c5fd")
        for r in rows:
            city_id = f"city_{r['City']}"
            net.add_node(city_id, label=r['City'], color=theme["city"], shape="dot", size=18)
            # Signals
            sigs = [
                (f"temp_{r['City']}", f"Temp {r['Temp (°C)'] if r['Temp (°C)'] is not None else '—'}°C"),
                (f"pm_{r['City']}", f"PM2.5 {r['PM2.5 (µg/m³)'] if r['PM2.5 (µg/m³)'] is not None else '—'}")
            ]
            for sid, slabel in sigs:
                net.add_node(sid, label=slabel, color=theme["signal"], shape="dot")
                net.add_edge(city_id, sid, color="#fcd34d")
            # Resources
            for res in ["Energy", "Water", "Mobility"]:
                nid = f"{res}_{r['City']}"
                demand = r[f"{res}_Demand"]
                supply = r[f"{res}_Supply"]
                color = theme["resource"] if supply >= demand else theme["risk"]
                net.add_node(nid, label=f"{res}: {supply:.1f}/{demand:.1f}", color=color, shape="ellipse")
                net.add_edge(city_id, nid, color="#6ee7b7")
            # Governance edge
            net.add_edge(city_id, "Governance", color="#c4b5fd")
            net.add_edge(city_id, "Scenario", color="#f9a8d4")
            net.add_edge(city_id, "Interventions", color="#67e8f9")
        return net.generate_html(notebook=False)
    except Exception:
        return None

# =========================
# UI — SIDEBAR
# =========================
st.title("🌍 City↔Planet Orchestration — Real‑Time, Free Open APIs")
st.caption("Live weather & air‑quality feeds + simulation + planetary compliance + Knowledge Graph + Agentic AI summary.")

with st.sidebar:
    st.header("Configuration")
    cities = st.multiselect("Cities (1–3)", list(DEFAULT_CITIES.keys()), default=["Delhi", "New York"])
    scenario = st.selectbox("Stress Scenario", SCENARIOS, index=0)
    st.markdown("**Interventions**")
    load_shed = st.slider("Energy Load Shedding (%)", 0, 50, 10)
    water_restrict = st.slider("Water Restrictions (%)", 0, 60, 15)
    mobility_reroute = st.slider("Mobility Rerouting (%)", 0, 60, 10)
    st.markdown("---")
    refresh_sec = st.slider("Auto‑refresh every (sec)", 5, 120, 20)
    auto = st.toggle("Auto‑refresh", value=True)
    show_kg = st.toggle("Show Knowledge Graph", value=True)
    use_local_llm = st.toggle("Enable Local LLM Agent (FLAN‑T5 small)", value=False)
    st.caption("LLM toggle downloads ~300MB the first time. If off, a fast heuristic agent is used.")

if auto:
    st.autorefresh(interval=refresh_sec * 1000, key="auto_refresh_key")

if not cities:
    st.info("Select at least one city in the sidebar.")
    st.stop()

# =========================
# MAIN DASHBOARD
# =========================

rows: List[Dict] = []

for city in cities:
    lat, lon, pop_fallback_m, qid = DEFAULT_CITIES[city]

    # Live data
    hourly = get_open_meteo_hourly(lat, lon)
    wx_temp = hourly.get("temperature_2m", [])
    wx_time = hourly.get("time", [])

    temp_now = (wx_temp[-1] if wx_temp else None)

    df_pm = get_openaq_timeseries(lat, lon)
    pm_now = float(df_pm["pm25"].iloc[-1]) if not df_pm.empty else None

    # Knowledge graph
    pop_m, country = wikidata_population_country(qid)
    pop_m = pop_m or pop_fallback_m

    # Simulation
    demand0 = base_demands(pop_m, temp_now)
    demand  = apply_scenario(demand0, scenario, pm_now)
    supply0 = supply_capacity(demand)
    supply  = apply_interventions(supply0, load_shed, water_restrict, mobility_reroute)
    unmet, co2 = unmet_and_emissions(demand, supply)

    # ------- UI panels per city
    st.subheader(f"🏙️ {city}")
    colA, colB, colC = st.columns([1.6, 1.5, 1.6], gap="large")

    with colA:
        st.markdown("#### Live Signals & Context")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temp (°C)", None if temp_now is None else f"{temp_now:.1f}")
        c2.metric("PM2.5 (µg/m³)", "—" if pm_now is None else f"{pm_now:.1f}")
        c3.metric("Population (M)", f"{pop_m:.2f}")
        if wx_time and wx_temp:
            df_temp = pd.DataFrame({"time": pd.to_datetime(wx_time), "temp": wx_temp}).set_index("time")
            st.line_chart(df_temp)
        if not df_pm.empty:
            st.area_chart(df_pm.set_index("datetime"))
        extract = get_wikipedia_summary(city)
        if extract:
            st.caption(extract[:420] + ("…" if len(extract) > 420 else ""))

    with colB:
        st.markdown("#### Demand vs Supply (now)")
        df_now = pd.DataFrame({
            "Resource": ["Energy", "Water", "Mobility"],
            "Demand": [demand["energy"], demand["water"], demand["mobility"]],
            "Supply": [supply["energy"], supply["water"], supply["mobility"]],
        }).set_index("Resource")
        st.bar_chart(df_now)
        shortfall = sum(unmet.values())
        if shortfall < 0.1:
            st.success("Balanced ✅ (minor or no unmet demand)")
        else:
            st.warning(f"Shortfall: {shortfall:.2f} (abstract units)")

    with colC:
        st.markdown("#### Governance & Compliance")
        budget_share = PLANETARY_BUDGET_TPD_CO2 / max(1, len(cities))
        if co2 <= budget_share * 0.6:
            st.success(f"Planetary compliance ✅  CO₂={co2:.1f} t/day ≤ {budget_share:.0f}")
        elif co2 <= budget_share:
            st.warning(f"Near limit ⚠️  CO₂={co2:.1f} t/day ~ {budget_share:.0f}")
        else:
            st.error(f"Overshoot ❌  CO₂={co2:.1f} t/day > {budget_share:.0f}")

    st.divider()

    rows.append({
        "City": city,
        "Country": country or "—",
        "Population (M)": round(pop_m, 2),
        "Temp (°C)": temp_now,
        "PM2.5 (µg/m³)": pm_now,
        "Energy_Demand": demand["energy"],
        "Energy_Supply": supply["energy"],
        "Water_Demand": demand["water"],
        "Water_Supply": supply["water"],
        "Mobility_Demand": demand["mobility"],
        "Mobility_Supply": supply["mobility"],
        "Unmet_Energy": unmet["energy"],
        "Unmet_Water": unmet["water"],
        "Unmet_Mobility": unmet["mobility"],
        "CO2_tpd": co2,
    })

# =========================
# MULTI‑CITY COORDINATION (toy)
# =========================
st.markdown("## 🤝 Multi‑City Coordination")
df_all = pd.DataFrame(rows)

# Simple capped transfer heuristic (up to 10% of a city's supply)
def rebalance_energy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    deficit = (df["Energy_Demand"] - df["Energy_Supply"]).clip(lower=0)
    surplus = (df["Energy_Supply"] - df["Energy_Demand"]).clip(lower=0)
    transfer_cap = (df["Energy_Supply"] * 0.10).where(surplus > 0, 0)
    pool = transfer_cap.sum(); need = deficit.sum()
    if need <= 0 or pool <= 0:
        df["Post_Rebalance_Unmet_Energy"] = deficit
        return df
    ratio = min(1.0, pool / need)
    df["Post_Rebalance_Unmet_Energy"] = deficit * (1 - ratio)
    return df

if not df_all.empty:
    df_reb = rebalance_energy(df_all)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            df_all[[
                "City","Country","Population (M)","Temp (°C)","PM2.5 (µg/m³)",
                "Energy_Demand","Energy_Supply","Unmet_Energy","CO2_tpd"
            ]].round(2), use_container_width=True
        )
    with col2:
        st.dataframe(
            df_reb[["City","Post_Rebalance_Unmet_Energy"]].round(2),
            use_container_width=True
        )

    co2_before = df_all["CO2_tpd"].sum()
    co2_after  = (df_reb["Post_Rebalance_Unmet_Energy"] * EMISSION_FACTOR_PER_UNMET_ENERGY).sum()

    st.markdown("### Planetary Compliance (aggregate)")
    budget_total = PLANETARY_BUDGET_TPD_CO2
    if co2_after <= budget_total * 0.6:
        st.success(f"Post‑rebalance ✅ Global CO₂={co2_after:.0f} t/day ≤ budget {budget_total:,}")
    elif co2_after <= budget_total:
        st.warning(f"Post‑rebalance ⚠️ Global CO₂={co2_after:.0f} t/day ~ budget {budget_total:,}")
    else:
        st.error(f"Post‑rebalance ❌ Global CO₂={co2_after:.0f} t/day > budget {budget_total:,}")

# =========================
# KNOWLEDGE GRAPH VISUAL
# =========================
if st.checkbox("Render Knowledge Graph (beautiful, interactive)", value=True) and not df_all.empty:
    html = build_kg_html(
        rows,
        scenario,
        interventions={"load": load_shed, "water": water_restrict, "mob": mobility_reroute},
    )
    if html:
        st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.info("Install pyvis in requirements.txt to enable the graph: pyvis==0.3.2")

# =========================
# AGENTIC AI SUMMARY
# =========================
st.markdown("## 🤖 Agentic AI — Analytical Conclusion")
ctx_lines = [f"Scenario: {scenario}", f"Interventions: load={load_shed} water={water_restrict} mobility={mobility_reroute}"]
for r in rows:
    ctx_lines.append(
        f"City: {r['City']} | Temp={r['Temp (°C)']} | PM2.5={r['PM2.5 (µg/m³)']} | "
        f"Energy {r['Energy_Supply']:.2f}/{r['Energy_Demand']:.2f} | Water {r['Water_Supply']:.2f}/{r['Water_Demand']:.2f} | "
        f"Mob {r['Mobility_Supply']:.2f}/{r['Mobility_Demand']:.2f} | Unmet energy={r['Unmet_Energy']:.2f}, water={r['Unmet_Water']:.2f}, mobility={r['Unmet_Mobility']:.2f}"
    )
ctx_lines.append(
    f"Aggregate: CO2_before={co2_before if 'co2_before' in locals() else 0:.1f}, CO2_after={co2_after if 'co2_after' in locals() else 0:.1f}, Budget={PLANETARY_BUDGET_TPD_CO2}"
)
context_text = "
".join(ctx_lines)

if use_local_llm:
    with st.spinner("Running local open‑source model (FLAN‑T5 small)…"):
        summary = generate_agentic_summary(context_text)
else:
    summary = heuristic_summary(context_text)

st.text_area("Conclusion", summary, height=180)

# =========================
# EXPORT EXEC SUMMARY
# =========================
st.markdown("## 📥 Export Executive Report (Markdown)")
now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
md = [
    f"# City↔Planet Orchestration Report

Generated: {now}

",
    f"**Scenario:** {scenario}

",
    f"**Interventions:** Load shedding={load_shed}%, Water restrictions={water_restrict}%, Mobility rerouting={mobility_reroute}%

",
]
for r in rows:
    md.append(textwrap.dedent(f"""
    ## {r['City']} ({r['Country']})
    - Population: {r['Population (M)']} M
    - Weather: temp={r['Temp (°C)']}°C, PM2.5={r['PM2.5 (µg/m³)']} µg/m³
    - Energy (Demand vs Supply): {r['Energy_Demand']:.2f} vs {r['Energy_Supply']:.2f}
    - Water (Demand vs Supply): {r['Water_Demand']:.2f} vs {r['Water_Supply']:.2f}
    - Mobility (Demand vs Supply): {r['Mobility_Demand']:.2f} vs {r['Mobility_Supply']:.2f}
    - Unmet: energy={r['Unmet_Energy']:.2f}, water={r['Unmet_Water']:.2f}, mobility={r['Unmet_Mobility']:.2f}
    - City CO₂ (tpd, toy): {r['CO2_tpd']:.2f}
    """))

md.append(textwrap.dedent(f"""
## Coordination Outcome
- Global CO₂ before: {co2_before if 'co2_before' in locals() else 0:.0f} t/day
- Global CO₂ after:  {co2_after if 'co2_after' in locals() else 0:.0f} t/day
- Planetary budget:  {PLANETARY_BUDGET_TPD_CO2:,} t/day

*APIs used: Open‑Meteo (hourly weather), OpenAQ (PM2.5), Wikipedia (summary), Wikidata SPARQL (population/country).*
"""))

md_text = "
".join(md)
st.download_button("Download Report (.md)", data=md_text.encode("utf-8"), file_name="city_planet_report.md")

st.markdown("---")
st.caption("© Your Name — Educational demo. All assumptions are explicit; replace toy factors with domain data when available.  |  Graph: pyvis  |  LLM: FLAN‑T5 small (optional)")
