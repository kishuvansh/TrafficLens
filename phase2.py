"""
phase2_model.py
===============
Parking-Induced Congestion Intelligence Pipeline
Builds on phase1.ipynb to produce:
  - cluster_data.csv        : DBSCAN spatial clusters with centroids & stats
  - hotspot_report.csv      : Ranked enforcement priority table
  - temporal_peaks.csv      : Peak hour/day per cluster
  - df_engineered.csv       : Full feature-engineered dataset for dashboard
  - enforcement_hotspots_v2.html : Interactive folium map (clusters + heatmap)

Run:
    python phase2_model.py
"""

import ast
import warnings
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 0 — CONFIG
# ─────────────────────────────────────────────

DATA_PATH = "trafficdata.csv"

# Violation severity weights (same as phase1, centralised here)
VIOLATION_WEIGHTS = {
    "PARKING IN A MAIN ROAD": 3,
    "WRONG PARKING": 2,
    "NO PARKING": 2,
    "PARKING ON FOOTPATH": 3,
    "DOUBLE PARKING": 3,
    "PARKING NEAR ROAD CROSSING": 2,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 2,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1,
}

# DBSCAN: ~30m radius (0.03 km / Earth radius 6371 km)
DBSCAN_EPS_KM = 0.03
DBSCAN_MIN_SAMPLES = 15

# Peak hour windows (for risk multiplier)
AM_PEAK = range(7, 11)   # 7–10 AM
PM_PEAK = range(17, 21)  # 5–8 PM


# ─────────────────────────────────────────────
# STEP 1 — LOAD & PARSE VIOLATION TYPES
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading data and parsing violation types...")

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

# Parse stringified lists → actual Python lists
def safe_parse(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["violation_list"] = df["violation_type"].apply(safe_parse)

# MultiLabelBinarizer → 27 binary violation columns
mlb = MultiLabelBinarizer()
violation_encoded = pd.DataFrame(
    mlb.fit_transform(df["violation_list"]),
    columns=mlb.classes_,
    index=df.index,
)
df = df.join(violation_encoded)
VIOLATION_COLS = mlb.classes_.tolist()
print(f"  Parsed {len(VIOLATION_COLS)} unique violation types")


# ─────────────────────────────────────────────
# STEP 2 — TEMPORAL FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\nSTEP 2: Temporal feature engineering...")

df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="mixed", utc=True)
df["hour"]        = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.dayofweek  # 0=Mon
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["month"]       = df["created_datetime"].dt.month
df["week"]        = df["created_datetime"].dt.isocalendar().week.astype(int)
df["is_peak_hour"] = df["hour"].apply(
    lambda h: 1 if (h in AM_PEAK or h in PM_PEAK) else 0
)

DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["day_name"] = df["day_of_week"].map(DAY_NAMES)

print(f"  Date range: {df['created_datetime'].min().date()} → {df['created_datetime'].max().date()}")


# ─────────────────────────────────────────────
# STEP 3 — PRIORITY SCORE
# ─────────────────────────────────────────────

print("\nSTEP 3: Computing priority scores...")

def calculate_priority(row):
    score = 0
    for violation, weight in VIOLATION_WEIGHTS.items():
        if violation in row.index and row[violation] == 1:
            score += weight
    # Peak hour multiplier
    if row.get("is_peak_hour", 0) == 1:
        score *= 1.5
    return round(score, 2)

df["priority_score"] = df.apply(calculate_priority, axis=1)

# is_resolved flag (100% null in current data, but structurally correct)
df["is_resolved"] = df["action_taken_timestamp"].notna().astype(int)
print(f"  Priority score stats: mean={df['priority_score'].mean():.2f}, max={df['priority_score'].max():.2f}")
print(f"  Resolved cases: {df['is_resolved'].sum():,} / {len(df):,}")


# ─────────────────────────────────────────────
# STEP 4 — DBSCAN SPATIAL CLUSTERING
# ─────────────────────────────────────────────

print("\nSTEP 4: DBSCAN spatial clustering...")

coords     = df[["latitude", "longitude"]].values
coords_rad = np.radians(coords)

db = DBSCAN(
    eps=DBSCAN_EPS_KM / 6371.0,
    min_samples=DBSCAN_MIN_SAMPLES,
    algorithm="ball_tree",
    metric="haversine",
)
df["cluster"] = db.fit_predict(coords_rad)

n_clusters = df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)
noise_pct   = (df["cluster"] == -1).mean() * 100
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {noise_pct:.1f}% of records (no cluster assigned)")


# ─────────────────────────────────────────────
# STEP 5 — CLUSTER AGGREGATION & RISK SCORING
# ─────────────────────────────────────────────
print("\nSTEP 5: Aggregating clusters and computing risk scores...")

df_clustered = df[df["cluster"] != -1].copy()

def mode_or_none(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else None

cluster_data = df_clustered.groupby("cluster").agg(
    center_lat        = ("latitude",       "mean"),
    center_lon        = ("longitude",      "mean"),
    violation_count   = ("id",             "count"),
    avg_priority      = ("priority_score", "mean"),
    total_priority    = ("priority_score", "sum"),
    peak_hour         = ("hour",           mode_or_none),
    peak_day          = ("day_of_week",    mode_or_none),
    peak_weekend_frac = ("is_weekend",     "mean"),
    peak_hour_frac    = ("is_peak_hour",   "mean"),
    top_junction      = ("junction_name",  mode_or_none),
    top_vehicle       = ("vehicle_type",   mode_or_none),
    resolved_rate     = ("is_resolved",    "mean"),
).reset_index()

# Map peak_day number → name
cluster_data["peak_day_name"] = cluster_data["peak_day"].map(DAY_NAMES)


def quantile_norm(series):
    """Rank-based normalisation — guarantees 0–1 spread regardless of outliers."""
    return series.rank(pct=True)

def log_norm(series):
    """Log-scale then quantile norm — for heavily skewed counts."""
    return np.log1p(series).rank(pct=True)

# Dimension 1: Frequency — log scale because max=55K vs median=37
cluster_data["_freq_norm"] = log_norm(cluster_data["violation_count"])

# Dimension 2: Severity — quantile norm because avg_priority is tightly bunched
cluster_data["_sev_norm"] = quantile_norm(cluster_data["avg_priority"])

# Dimension 3: Timing — keep raw fraction but boost signal with sqrt
# sqrt spreads out the low values (most clusters are 0–0.2)
cluster_data["_time_norm"] = np.sqrt(cluster_data["peak_hour_frac"])

# Dimension 4: Junction proximity flag (binary, no change needed)
cluster_data["_junc_flag"] = cluster_data["top_junction"].apply(
    lambda j: 0 if str(j).strip().lower() == "no junction" else 1
)

# Weighted composite
cluster_data["risk_score"] = (
    0.40 * cluster_data["_freq_norm"]  +
    0.30 * cluster_data["_sev_norm"]   +
    0.20 * cluster_data["_time_norm"]  +
    0.10 * cluster_data["_junc_flag"]
) * 10

cluster_data["risk_score"] = cluster_data["risk_score"].round(2)

# Drop temp columns
cluster_data.drop(
    columns=["_freq_norm", "_sev_norm", "_time_norm", "_junc_flag"],
    inplace=True
)

# Assign risk tiers and ranks so downstream steps can reference them
def score_to_tier(s):
    try:
        s = float(s)
    except Exception:
        return "🟢 LOW"
    if s >= 7.0:
        return "🔴 CRITICAL"
    if s >= 4.0:
        return "🟠 HIGH"
    if s >= 2.0:
        return "🟡 MEDIUM"
    return "🟢 LOW"

cluster_data["risk_tier"] = cluster_data["risk_score"].apply(score_to_tier)
# rank 1 = highest risk
cluster_data = cluster_data.sort_values("risk_score", ascending=False).reset_index(drop=True)
cluster_data["rank"] = (cluster_data.index + 1).astype(int)

# ─────────────────────────────────────────────
# STEP 6 — TEMPORAL PEAK ANALYSIS PER CLUSTER
# ─────────────────────────────────────────────

print("\nSTEP 6: Temporal peak analysis...")

temporal = df_clustered.groupby(["cluster", "hour"]).size().reset_index(name="count")
temporal = temporal.merge(
    cluster_data[["cluster", "risk_score", "risk_tier"]],
    on="cluster", how="left"
)

# Also build weekday-level peaks
daily_peaks = df_clustered.groupby(["cluster", "day_of_week"]).size().reset_index(name="count")
daily_peaks["day_name"] = daily_peaks["day_of_week"].map(DAY_NAMES)

print(f"  Temporal records: {len(temporal):,} hour-cluster combinations")


# ─────────────────────────────────────────────
# STEP 7 — ANOMALY DETECTION (IsolationForest)
# ─────────────────────────────────────────────

print("\nSTEP 7: Anomaly detection on violation spikes...")

# Aggregate to zone × week × hour grain
zone_time = df_clustered.groupby(["cluster", "week", "hour"]).agg(
    count=("id", "count"),
    avg_priority=("priority_score", "mean"),
).reset_index()

features = zone_time[["count", "avg_priority"]].fillna(0)

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
zone_time["anomaly"] = iso.fit_predict(features)
zone_time["is_anomaly"] = (zone_time["anomaly"] == -1).astype(int)

anomalies = zone_time[zone_time["is_anomaly"] == 1].copy()
anomalies = anomalies.merge(
    cluster_data[["cluster", "top_junction", "risk_score"]],
    on="cluster", how="left"
).sort_values("count", ascending=False)

print(f"  Anomalous spikes detected: {len(anomalies):,}")
print(f"  Top anomaly — Cluster {anomalies.iloc[0]['cluster']:.0f} | "
      f"Week {anomalies.iloc[0]['week']:.0f} | "
      f"Hour {anomalies.iloc[0]['hour']:.0f}:00 | "
      f"Count {anomalies.iloc[0]['count']:.0f}")


# ─────────────────────────────────────────────
# STEP 8 — ENFORCEMENT PRIORITY REPORT
# ─────────────────────────────────────────────

print("\nSTEP 8: Building enforcement priority report...")

def recommend_action(row):
    tier = row["risk_tier"]
    hour = int(row["peak_hour"]) if not pd.isna(row["peak_hour"]) else 8
    day  = row.get("peak_day_name", "Weekday")
    vh   = str(row.get("top_vehicle", "vehicles"))

    time_str = f"{hour}:00–{hour+2}:00"

    if "CRITICAL" in tier:
        return f"Deploy 2+ units {day}s {time_str}. Priority tow zone. Focus: {vh}."
    elif "HIGH" in tier:
        return f"Deploy 1 unit {day}s {time_str}. Issue penalties. Focus: {vh}."
    elif "MEDIUM" in tier:
        return f"Patrol alert {day}s {time_str}. Focus: {vh}."
    else:
        return f"Monitor via CCTV. Spot check {day}s."

hotspot_report = cluster_data[[
    "rank", "cluster", "risk_tier", "risk_score",
    "top_junction", "violation_count", "avg_priority",
    "peak_hour", "peak_day_name", "peak_weekend_frac",
    "top_vehicle", "center_lat", "center_lon",
]].copy()

hotspot_report["recommended_action"] = cluster_data.apply(recommend_action, axis=1)

print(f"\n{'='*60}")
print("  TOP 10 ENFORCEMENT PRIORITY ZONES")
print(f"{'='*60}")
cols_display = ["rank", "risk_tier", "risk_score", "top_junction", "violation_count", "peak_hour"]
print(hotspot_report[cols_display].head(10).to_string(index=False))


# ─────────────────────────────────────────────
# STEP 9 — INTERACTIVE FOLIUM MAP (v2)
# ─────────────────────────────────────────────

print("\nSTEP 9: Generating enforcement map...")

TIER_COLORS = {
    "🔴 CRITICAL": "red",
    "🟠 HIGH":     "orange",
    "🟡 MEDIUM":   "beige",
    "🟢 LOW":      "green",
}

center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

# Layer 1 — background heatmap of all violations
heat_data = df[["latitude", "longitude", "priority_score"]].copy()
heat_data["priority_score"] = heat_data["priority_score"].fillna(0)
max_p = heat_data["priority_score"].max()
if max_p > 0:
    heat_data["w"] = heat_data["priority_score"] / max_p
else:
    heat_data["w"] = 1
HeatMap(
    heat_data[["latitude", "longitude", "w"]].values.tolist(),
    radius=10, blur=15, min_opacity=0.2, max_zoom=18,
    name="Violation Heatmap",
).add_to(m)

# Layer 2 — cluster markers (top 50 by risk score)
top_clusters = hotspot_report.head(50)
for _, row in top_clusters.iterrows():
    color = TIER_COLORS.get(row["risk_tier"], "blue")
    popup_html = f"""
    <div style='font-family:sans-serif; width:220px'>
        <b>Rank #{int(row['rank'])} — {row['risk_tier']}</b><br>
        <hr style='margin:4px 0'>
        <b>Junction:</b> {row['top_junction']}<br>
        <b>Risk Score:</b> {row['risk_score']:.1f} / 10<br>
        <b>Violations:</b> {int(row['violation_count']):,}<br>
        <b>Peak Hour:</b> {int(row['peak_hour'])}:00<br>
        <b>Peak Day:</b> {row['peak_day_name']}<br>
        <b>Top Vehicle:</b> {row['top_vehicle']}<br>
        <hr style='margin:4px 0'>
        <b>Action:</b> {row['recommended_action']}
    </div>
    """
    folium.CircleMarker(
        location=[row["center_lat"], row["center_lon"]],
        radius=8 + row["risk_score"],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"#{int(row['rank'])} {row['risk_tier']} | Score: {row['risk_score']:.1f}",
    ).add_to(m)

folium.LayerControl().add_to(m)

# Legend
legend_html = """
<div style='position:fixed; bottom:40px; left:40px; z-index:1000;
     background:white; padding:12px 16px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:sans-serif; font-size:13px'>
    <b>Enforcement Priority</b><br>
    <span style='color:red'>●</span> CRITICAL (≥7.0)<br>
    <span style='color:orange'>●</span> HIGH (4–7)<br>
    <span style='color:#c8a000'>●</span> MEDIUM (2–4)<br>
    <span style='color:green'>●</span> LOW (&lt;2)<br>
    <br><i>Circle size ∝ risk score<br>Click marker for details</i>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

map_path = "enforcement_hotspots_v2.html"
m.save(map_path)
print(f"  Map saved → {map_path}")


# ─────────────────────────────────────────────
# STEP 10 — EXPORT CSVs
# ─────────────────────────────────────────────

print("\nSTEP 10: Exporting outputs...")

# 1. Full engineered dataset (for dashboard)
EXPORT_COLS = [
    "id", "latitude", "longitude", "vehicle_type", "violation_list",
    "junction_name", "police_station", "created_datetime",
    "hour", "day_of_week", "day_name", "is_weekend", "is_peak_hour",
    "month", "week", "priority_score", "is_resolved", "cluster",
] + VIOLATION_COLS

df_export = df[[c for c in EXPORT_COLS if c in df.columns]].copy()
df_export.to_csv("df_engineered.csv", index=False)
print(f"  df_engineered.csv        → {len(df_export):,} rows")

# 2. Cluster summary (for dashboard map layer)
cluster_data.to_csv("cluster_data.csv", index=False)
print(f"  cluster_data.csv         → {len(cluster_data):,} clusters")

# 3. Enforcement priority report
hotspot_report.to_csv("hotspot_report.csv", index=False)
print(f"  hotspot_report.csv       → {len(hotspot_report):,} ranked zones")

# 4. Temporal peaks (for per-cluster time charts)
temporal.to_csv("temporal_peaks.csv", index=False)
print(f"  temporal_peaks.csv       → {len(temporal):,} rows")

# 5. Anomaly log
anomalies.to_csv("anomaly_log.csv", index=False)
print(f"  anomaly_log.csv          → {len(anomalies):,} anomalous spikes")

"""
Step 11 — Reverse Geocode "No Junction" Cluster Centroids
Enriches cluster_data.csv with real road/area names.
Uses Nominatim (free, no API key needed).
Only geocodes clusters labelled "No Junction" — named junctions kept as-is.
"""

import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ── Load cluster data ─────────────────────────────────────────────
cluster_data = pd.read_csv("cluster_data.csv")

print(f"Total clusters: {len(cluster_data)}")
print(f"Named junctions : {(cluster_data['top_junction'] != 'No Junction').sum()}")
print(f"No Junction rows: {(cluster_data['top_junction'] == 'No Junction').sum()}")

# ── Setup Nominatim ───────────────────────────────────────────────
geolocator = Nominatim(
    user_agent="parking_congestion_intel_v1",
    timeout=10,
)

def reverse_geocode(lat, lon, retries=3):
    """
    Returns a clean location label from lat/lon.
    Tries up to `retries` times on timeout.
    Falls back to None on failure.
    """
    for attempt in range(retries):
        try:
            location = geolocator.reverse(
                (lat, lon),
                exactly_one=True,
                language="en",
            )
            if location is None:
                return None
            return location.raw.get("address", {})
        except GeocoderTimedOut:
            if attempt < retries - 1:
                time.sleep(2)
            continue
        except GeocoderServiceError:
            return None
    return None

def build_label(address_dict, fallback_station):
    """
    Build a clean human-readable label from Nominatim address dict.
    Priority: road > suburb > neighbourhood > city_district > police station fallback
    """
    if not address_dict:
        return f"{fallback_station} — Road Segment"

    road         = address_dict.get("road", "")
    suburb       = address_dict.get("suburb", "")
    neighbourhood= address_dict.get("neighbourhood", "")
    city_district= address_dict.get("city_district", "")
    county       = address_dict.get("county", "")

    # Build label: "Road Name, Area"
    primary   = road or neighbourhood or suburb or city_district or county
    secondary = suburb or city_district or county

    if primary and secondary and primary != secondary:
        label = f"{primary}, {secondary}"
    elif primary:
        label = primary
    else:
        label = f"{fallback_station} — Road Segment"

    return label.strip(", ")

# ── Geocode only No Junction rows ────────────────────────────────
no_junc_mask = cluster_data["top_junction"] == "No Junction"
no_junc_df   = cluster_data[no_junc_mask].copy()

print(f"\nGeocoding {len(no_junc_df)} No Junction clusters...")
print("(Nominatim rate limit: 1 request/sec — this will take a few minutes)\n")

labels      = []
failed      = 0

for i, (_, row) in enumerate(no_junc_df.iterrows()):
    lat     = row["center_lat"]
    lon     = row["center_lon"]
    station = str(row.get("top_junction", "Unknown"))  # fallback

    # Get police station from cluster data if available
    # Use it as fallback label
    ps_fallback = (
        df[df["cluster"] == row["cluster"]]["police_station"]
        .mode()
    )
    ps_fallback = ps_fallback.iloc[0] if not ps_fallback.empty else "Unknown Station"

    address = reverse_geocode(lat, lon)
    label   = build_label(address, ps_fallback)
    labels.append(label)

    # Progress log every 50 rows
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(no_junc_df)}] done — last: {label}")

    # Respect Nominatim 1 request/sec rate limit
    time.sleep(1.1)

print(f"\nGeocoding complete.")
print(f"  Successful labels : {len(labels) - failed}")
print(f"  Fallback used     : {failed}")

# ── Write labels back to cluster_data ────────────────────────────
no_junc_df["display_name"] = labels
cluster_data = cluster_data.merge(
    no_junc_df[["cluster", "display_name"]],
    on="cluster",
    how="left",
)

# For named junctions — display_name = junction name (strip BTP code prefix)
def clean_junction_name(j):
    """Convert 'BTP051 - Safina Plaza Junction' → 'Safina Plaza Junction'"""
    j = str(j)
    if j == "No Junction":
        return None  # will be filled from display_name
    # Strip BTP code if present
    if " - " in j:
        return j.split(" - ", 1)[1].strip()
    return j.strip()

cluster_data["display_name"] = cluster_data.apply(
    lambda row: (
        row["display_name"]
        if pd.notna(row.get("display_name"))
        else clean_junction_name(row["top_junction"])
    ),
    axis=1,
)

# Final fallback for any remaining nulls
cluster_data["display_name"] = cluster_data["display_name"].fillna(
    cluster_data["top_junction"]
)

# ── Preview ───────────────────────────────────────────────────────
print("\n=== LABEL PREVIEW ===")
print("\nNo Junction → resolved labels (sample 15):")
print(
    cluster_data[no_junc_mask][["cluster", "center_lat", "center_lon", "display_name"]]
    .head(15)
    .to_string(index=False)
)

print("\nNamed junctions → cleaned (sample 10):")
print(
    cluster_data[~no_junc_mask][["cluster", "top_junction", "display_name"]]
    .head(10)
    .to_string(index=False)
)

# ── Save enriched cluster_data ────────────────────────────────────
cluster_data.to_csv("cluster_data.csv", index=False)
print(f"\n✅ cluster_data.csv updated with display_name column ({len(cluster_data)} rows)")
print("   Use 'display_name' everywhere in app.py instead of 'top_junction'")
# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Total violations analysed : {len(df):,}")
print(f"  Spatial clusters found    : {n_clusters}")
print(f"  CRITICAL risk zones       : {(cluster_data['risk_tier'] == '🔴 CRITICAL').sum()}")
print(f"  HIGH risk zones           : {(cluster_data['risk_tier'] == '🟠 HIGH').sum()}")
print(f"  Anomalous spikes flagged  : {len(anomalies):,}")
print(f"\n  Outputs ready for app.py  :")
print(f"    ✅ df_engineered.csv")
print(f"    ✅ cluster_data.csv")
print(f"    ✅ hotspot_report.csv")
print(f"    ✅ temporal_peaks.csv")
print(f"    ✅ anomaly_log.csv")
print(f"    ✅ enforcement_hotspots_v2.html")
print("=" * 60)

# Step 11b — Merge display_name into hotspot_report.csv
hotspot = pd.read_csv("hotspot_report.csv")
cdata   = pd.read_csv("cluster_data.csv")[["cluster", "display_name"]]
hotspot = hotspot.merge(cdata, on="cluster", how="left")
hotspot.to_csv("hotspot_report.csv", index=False)
print(f"✅ hotspot_report.csv updated — {len(hotspot)} rows")
print(hotspot[["cluster", "display_name"]].head(10).to_string(index=False))