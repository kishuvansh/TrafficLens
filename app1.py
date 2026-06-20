"""
app.py  —  Parking Congestion Intelligence Dashboard
Reads CSVs from phase2_model.py and visualises results.
Run: streamlit run app.py
"""

import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Parking Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
)

TIER_COLORS_HEX = {
    "🔴 CRITICAL": [220, 30,  30,  200],
    "🟠 HIGH":     [255, 140, 0,   200],
    "🟡 MEDIUM":   [255, 200, 0,   200],
    "🟢 LOW":      [40,  180, 40,  200],
}

DAY_NAMES = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}

# ─────────────────────────────────────────────
# DATA LOADERS  (cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_engineered():
    df = pd.read_csv("df_engineered.csv")
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["date"] = pd.to_datetime(df["created_datetime"]).dt.normalize()
    return df

@st.cache_data
def load_clusters():
    return pd.read_csv("cluster_data.csv")

@st.cache_data
def load_hotspots():
    return pd.read_csv("hotspot_report.csv")

@st.cache_data
def load_temporal():
    return pd.read_csv("temporal_peaks.csv")

@st.cache_data
def load_anomalies():
    return pd.read_csv("anomaly_log.csv")

# ─────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────

data_ok = False
try:
    df          = load_engineered()
    cluster_df  = load_clusters()
    hotspot_df  = load_hotspots()
    temporal_df = load_temporal()
    anomaly_df  = load_anomalies()
    data_ok = True
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run phase2_model.py first.")
    # stop execution cleanly when run as a Streamlit app or during imports
    st.stop()
    raise SystemExit(e)

# ─────────────────────────────────────────────
# SIDEBAR — GLOBAL FILTERS
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🚦 Filters")
    st.markdown("---")

    # Risk tier filter
    all_tiers = ["🔴 CRITICAL", "🟠 HIGH", "🟡 MEDIUM", "🟢 LOW"]
    sel_tiers = st.multiselect(
        "Risk Tier",
        options=all_tiers,
        default=all_tiers,
    )

    # Police station filter
    stations = sorted(df["police_station"].dropna().unique().tolist())
    sel_stations = st.multiselect(
        "Police Station",
        options=stations,
        default=[],
        placeholder="All stations",
    )

    # Date range filter (use date_input for stable date handling)
    non_null_dates = pd.to_datetime(df["date"].dropna())
    if non_null_dates.empty:
        st.warning("No valid dates in dataset.")
        st.stop()
    min_dt = non_null_dates.min().normalize()
    max_dt = non_null_dates.max().normalize()
    # present as python date objects for the widget
    sel_range = st.date_input(
        "Date Range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    # normalize selected range back to timestamps for filtering
    # ensure selected dates have the same timezone-awareness as df['date']
    series_tz = df["date"].dt.tz
    def _localize(ts):
        t = pd.to_datetime(ts).normalize()
        if series_tz is not None:
            try:
                # tz may be a tzinfo or string; tz_localize accepts both
                return t.tz_localize(series_tz)
            except Exception:
                return t.tz_localize("UTC")
        return t

    if isinstance(sel_range, tuple):
        start_date = _localize(sel_range[0])
        end_date = _localize(sel_range[1])
    else:
        start_date = end_date = _localize(sel_range)
    st.markdown("---")
    st.caption("phase2_model.py outputs loaded ✅")

# Apply global filters to raw df
df_filtered = df[
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
]
if sel_stations:
    df_filtered = df_filtered[df_filtered["police_station"].isin(sel_stations)]

# Apply tier filter to cluster/hotspot df
hotspot_filtered = hotspot_df[hotspot_df["risk_tier"].isin(sel_tiers)]
cluster_filtered = cluster_df[cluster_df["risk_tier"].isin(sel_tiers)]

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🗺️ Hotspot Map",
    "⏰ Temporal Patterns",
    "🚨 Anomaly Alerts",
    "🔍 Data Explorer",
    "🔮 Predictive Dispatch",   # ← add this
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab1:
    st.header("Parking Violation Intelligence — Overview")

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Violations",   f"{len(df_filtered):,}")
    k2.metric("Spatial Clusters",   f"{cluster_filtered['cluster'].nunique():,}")
    k3.metric("Critical Zones",     f"{(hotspot_filtered['risk_tier'] == '🔴 CRITICAL').sum()}")
    k4.metric("Anomalous Spikes",   f"{len(anomaly_df):,}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Violations over time
    with col_l:
        st.subheader("Violations Over Time")
        ts = df_filtered.groupby("date").size().reset_index(name="count")
        ts["date"] = pd.to_datetime(ts["date"])
        st.line_chart(ts.set_index("date")["count"])

    # Vehicle type breakdown
    with col_r:
        st.subheader("Top Vehicle Types")
        vc = df_filtered["vehicle_type"].value_counts().head(10)
        st.bar_chart(vc)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Violation type breakdown (parse list column)
    with col_a:
        st.subheader("Top Violation Types")
        try:
            exploded = df_filtered["violation_list"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            ).explode()
            vt_counts = exploded.value_counts().head(10)
            st.bar_chart(vt_counts)
        except Exception:
            st.info("Could not parse violation_list column.")

    # Police station workload
    with col_b:
        st.subheader("Violations by Police Station")
        ps = df_filtered["police_station"].value_counts().head(10)
        st.bar_chart(ps)


# ══════════════════════════════════════════════
# TAB 2 — HOTSPOT MAP
# ══════════════════════════════════════════════


with tab2:
    st.header("Enforcement Hotspot Map")

    if cluster_filtered.empty:
        st.warning("No clusters match selected filters.")
    else:
        from streamlit_folium import st_folium
        import folium
        from folium.plugins import HeatMap

        # Merge recommended_action + display cols from hotspot_df
        cluster_map = cluster_filtered.merge(
            hotspot_df[[
                "cluster", "rank", "recommended_action",
                "peak_day_name", "top_vehicle", "peak_hour"
            ]],
            on="cluster",
            how="left",
            suffixes=("", "_hs"),
        )
        cluster_map["recommended_action"] = cluster_map["recommended_action"].fillna("No action assigned")
        cluster_map["rank"] = cluster_map["rank"].fillna(0).astype(int)

        # ── Build map ─────────────────────────────────────────
        center_lat = cluster_map["center_lat"].mean()
        center_lon = cluster_map["center_lon"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="OpenStreetMap",
        )

        # Layer 1 — Heatmap (sampled for performance)
        heat_src = df_filtered[["latitude", "longitude", "priority_score"]].dropna().copy()
        heat_src["priority_score"] = pd.to_numeric(
            heat_src["priority_score"], errors="coerce"
        ).fillna(0)
        max_p = heat_src["priority_score"].max()
        heat_src["w"] = heat_src["priority_score"] / max_p if max_p > 0 else 1
        heat_sample = heat_src.sample(min(30000, len(heat_src)), random_state=42)
        HeatMap(
            heat_sample[["latitude", "longitude", "w"]].values.tolist(),
            radius=10,
            blur=15,
            min_opacity=0.25,
            max_zoom=18,
            name="Violation Heatmap",
        ).add_to(m)

        # Layer 2 — Cluster markers
        # FIXED: use correct string keys matching risk_tier values exactly
        FOLIUM_COLORS = {
            "🔴 CRITICAL": "red",
            "🟠 HIGH":     "orange",
            "🟡 MEDIUM":   "cadetblue",
            "🟢 LOW":      "green",
        }

        for _, row in cluster_map.iterrows():
            tier   = str(row.get("risk_tier", "🟢 LOW")).strip()
            color  = FOLIUM_COLORS.get(tier, "blue")

            # FIXED: radius 5–18 pixels (was 60–160 — way too large)
            radius = int(5 + (float(row["risk_score"]) / 10.0) * 13)

            popup_html = f"""
            <div style='font-family:sans-serif; width:240px; font-size:13px;
                        color:#111111; background:#ffffff; padding:6px'>
                <b style='font-size:14px'>Rank #{int(row['rank'])} — {tier}</b>
                <hr style='margin:4px 0; border-color:#cccccc'>
                <b>Junction:</b> {row['display_name']}<br>
                <b>Risk Score:</b> {float(row['risk_score']):.1f} / 10<br>
                <b>Violations:</b> {int(row['violation_count']):,}<br>
                <b>Peak Hour:</b> {int(row['peak_hour'])}:00<br>
                <b>Peak Day:</b> {row['peak_day_name']}<br>
                <b>Top Vehicle:</b> {row['top_vehicle']}<br>
                <hr style='margin:4px 0; border-color:#cccccc'>
                <b>Action:</b><br>
                <span style='color:#333333'>{row['recommended_action']}</span>
            </div>
            """

            folium.CircleMarker(
                location=[row["center_lat"], row["center_lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=2,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=(
                    f"#{int(row['rank'])} {tier} | "
                    f"Score: {float(row['risk_score']):.1f} | "
                    f"{row['display_name']}"
                ),
            ).add_to(m)

        folium.LayerControl().add_to(m)

        # Legend — white background, dark text, always readable
        legend_html = """
        <div style='position:fixed; bottom:40px; left:40px; z-index:1000;
             background:#ffffff; padding:12px 16px; border-radius:8px;
             box-shadow:0 2px 8px rgba(0,0,0,0.3);
             font-family:sans-serif; font-size:13px; color:#111111;'>
            <b>Enforcement Priority</b><br><br>
            <span style='color:red; font-size:16px'>●</span>&nbsp; CRITICAL (≥ 7.0)<br>
            <span style='color:orange; font-size:16px'>●</span>&nbsp; HIGH (4 – 7)<br>
            <span style='color:cadetblue; font-size:16px'>●</span>&nbsp; MEDIUM (2 – 4)<br>
            <span style='color:green; font-size:16px'>●</span>&nbsp; LOW (&lt; 2)<br>
            <br><i style='font-size:11px'>Circle size ∝ risk score<br>Click marker for details</i>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, use_container_width=True, height=550, returned_objects=[])

        # ── Legend row ────────────────────────────────────────
        st.markdown("---")
        leg1, leg2, leg3, leg4 = st.columns(4)
        leg1.markdown("🔴 **CRITICAL** — Score ≥ 7")
        leg2.markdown("🟠 **HIGH** — Score 4–7")
        leg3.markdown("🟡 **MEDIUM** — Score 2–4")
        leg4.markdown("🟢 **LOW** — Score < 2")

        st.markdown("---")

        # ── Enforcement priority table — explicit dark styling ─
        st.subheader("Enforcement Priority Table")

        display_cols = [
            "rank", "risk_tier", "risk_score", "display_name",
            "violation_count", "avg_priority", "peak_hour",
            "peak_day_name", "top_vehicle", "recommended_action",
        ]
        available = [c for c in display_cols if c in hotspot_filtered.columns]

        # Styled dataframe with visible text
        styled = (
            hotspot_filtered[available]
            .sort_values("rank")
            .style
            .background_gradient(subset=["risk_score"], cmap="RdYlGn_r")
            .format({"risk_score": "{:.2f}", "avg_priority": "{:.2f}"})
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv_dl = hotspot_filtered[available].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Enforcement Report",
            data=csv_dl,
            file_name="enforcement_report.csv",
            mime="text/csv",
        )
# ══════════════════════════════════════════════
# TAB 3 — TEMPORAL PATTERNS
# ══════════════════════════════════════════════

with tab3:
    st.header("Temporal Violation Patterns")

    col_l, col_r = st.columns(2)

    # Global hour × weekday heatmap
    with col_l:
        st.subheader("Global Hour × Weekday Heatmap")
        try:
            pivot = df_filtered.pivot_table(
                index="day_of_week",
                columns="hour",
                values="id",
                aggfunc="count",
                fill_value=0,
            )
            pivot = pivot.reindex(index=range(7), fill_value=0)
            pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Weekday")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not build heatmap: {e}")

    # Peak hours bar
    with col_r:
        st.subheader("Violations by Hour of Day")
        hourly = df_filtered.groupby("hour").size().reindex(range(24), fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars = ax2.bar(range(24), hourly.values, color=[
            "#e63946" if (7 <= h <= 10 or 17 <= h <= 20) else "#457b9d"
            for h in range(24)
        ])
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Violation Count")
        ax2.set_xticks(range(24))
        ax2.set_title("Red = Peak Hours (7–10 AM, 5–8 PM)")
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # Per-cluster temporal drill-down
    st.subheader("Per-Cluster Hour Profile")

    cluster_ids = sorted(temporal_df["cluster"].dropna().unique().tolist())
    if not cluster_ids:
        st.info("No per-cluster temporal data available.")
        cluster_time = pd.DataFrame()
        sel_cluster = None
    else:
        sel_cluster = st.selectbox(
            "Select Cluster",
            options=cluster_ids,
            format_func=lambda c: (
                f"Cluster {c} — {hotspot_df[hotspot_df['cluster'] == c]['display_name'].values[0]}"
                if c in hotspot_df["cluster"].values else f"Cluster {c}"
            ),
        )

        cluster_time = temporal_df[temporal_df["cluster"] == sel_cluster]

    if not cluster_time.empty:
        col_ct, col_info = st.columns([3, 1])

        with col_ct:
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.bar(
                cluster_time["hour"],
                cluster_time["count"],
                color=[
                    "#e63946" if (7 <= h <= 10 or 17 <= h <= 20) else "#2a9d8f"
                    for h in cluster_time["hour"]
                ],
            )
            ax3.set_xlabel("Hour of Day")
            ax3.set_ylabel("Violation Count")
            ax3.set_xticks(range(24))
            st.pyplot(fig3)
            plt.close()

        with col_info:
            info = hotspot_df[hotspot_df["cluster"] == sel_cluster]
            if not info.empty:
                r = info.iloc[0]
                st.metric("Risk Score",     f"{r['risk_score']:.1f} / 10")
                st.metric("Risk Tier",      r["risk_tier"])
                st.metric("Total Violations", f"{int(r['violation_count']):,}")
                st.metric("Peak Hour",      f"{int(r['peak_hour'])}:00")
                st.caption(f"**Action:** {r['recommended_action']}")

    # Top 10 clusters peak hour summary
    st.markdown("---")
    st.subheader("Top 10 Clusters — Peak Hour Summary")
    top10 = hotspot_df.head(10)[["rank", "risk_tier", "display_name", "peak_hour", "peak_day_name", "violation_count"]]
    st.dataframe(top10, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — ANOMALY ALERTS
# ══════════════════════════════════════════════

with tab4:
    st.header("🚨 Anomalous Violation Spikes")

    st.info(
        "These zones recorded statistically unusual violation volumes "
        "in a specific week + hour window, detected via IsolationForest."
    )

    if anomaly_df.empty:
        st.success("No anomalies detected.")
    else:
        # Summary metrics
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Anomalies",    len(anomaly_df))
        a2.metric("Clusters Affected",  anomaly_df["cluster"].nunique())
        a3.metric("Peak Anomaly Count", int(anomaly_df["count"].max()))

        st.markdown("---")

        # Anomaly table
        st.subheader("Anomaly Log")
        disp_cols = [c for c in ["cluster", "display_name", "week", "hour", "count", "avg_priority", "risk_score"] if c in anomaly_df.columns]
        st.dataframe(
            anomaly_df[disp_cols].sort_values("count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        # Anomaly distribution by hour
        st.markdown("---")
        col_al, col_ar = st.columns(2)

        with col_al:
            st.subheader("Anomalies by Hour of Day")
            ah = anomaly_df.groupby("hour")["count"].sum().reindex(range(24), fill_value=0)
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            ax4.bar(range(24), ah.values, color="#e63946")
            ax4.set_xlabel("Hour")
            ax4.set_ylabel("Total Anomalous Violations")
            ax4.set_xticks(range(24))
            st.pyplot(fig4)
            plt.close()

        with col_ar:
            st.subheader("Anomalies by Week")
            if "week" in anomaly_df.columns:
                aw = anomaly_df.groupby("week")["count"].sum()
                fig5, ax5 = plt.subplots(figsize=(8, 3))
                ax5.bar(aw.index, aw.values, color="#f4a261")
                ax5.set_xlabel("Week Number")
                ax5.set_ylabel("Total Anomalous Violations")
                st.pyplot(fig5)
                plt.close()

        # Download anomaly log
        csv_a = anomaly_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Anomaly Log",
            data=csv_a,
            file_name="anomaly_log_export.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════

with tab5:
    st.header("Raw Data Explorer")

    # Sub-filters specific to this tab
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        veh_opts = sorted(df_filtered["vehicle_type"].dropna().unique().tolist())
        sel_veh  = st.multiselect("Vehicle Type", veh_opts, default=[])

    with col_f2:
        cluster_opts = sorted(df_filtered["cluster"].dropna().unique().tolist())
        sel_clust    = st.multiselect("Cluster", cluster_opts, default=[])

    with col_f3:
        tier_opts = sorted(df_filtered["cluster"].dropna().unique().tolist())
        peak_only = st.checkbox("Peak hours only (7–10, 17–20)", value=False)

    # Apply sub-filters
    df_explore = df_filtered.copy()
    if sel_veh:
        df_explore = df_explore[df_explore["vehicle_type"].isin(sel_veh)]
    if sel_clust:
        df_explore = df_explore[df_explore["cluster"].isin(sel_clust)]
    if peak_only:
        df_explore = df_explore[df_explore["is_peak_hour"] == 1]

    st.caption(f"Showing {len(df_explore):,} rows")

    # Show columns that matter
    show_cols = [c for c in [
        "id", "vehicle_type", "violation_list", "junction_name",
        "police_station", "created_datetime", "hour", "day_name",
        "is_weekend", "is_peak_hour", "priority_score", "cluster",
    ] if c in df_explore.columns]

    st.dataframe(df_explore[show_cols].head(2000), use_container_width=True, hide_index=True)

    if len(df_explore) > 2000:
        st.caption("⚠️ Showing first 2,000 rows. Download for full dataset.")

    csv_ex = df_explore[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data",
        data=csv_ex,
        file_name="filtered_data.csv",
        mime="text/csv",
    )
# ══════════════════════════════════════════════
# TAB 6 — PREDICTIVE DISPATCH
# ══════════════════════════════════════════════

with tab6:
    st.header("🔮 Predictive Dispatch Engine")
    st.caption(
        "Powered by RandomForest classifier trained on 5 months of violation patterns. "
        "Predicts which zones will spike and when to deploy enforcement units."
    )

    # ── Load phase3 outputs ───────────────────────────────────────
    try:
        spike_df    = pd.read_csv("spike_predictions.csv")
        weekly_df   = pd.read_csv("weekly_forecast.csv")
        dispatch_df = pd.read_csv("dispatch_briefing.csv")
        pattern_df  = pd.read_csv("pattern_forecast.csv")
        phase3_ok   = True
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Run phase3.ipynb first.")
        phase3_ok = False

    if phase3_ok:

        # Parse forecast dates
        spike_df["forecast_date"]    = pd.to_datetime(spike_df["forecast_date"])
        weekly_df["forecast_date"]   = pd.to_datetime(weekly_df["forecast_date"])
        dispatch_df["forecast_date"] = pd.to_datetime(dispatch_df["forecast_date"])
        pattern_df["forecast_date"]  = pd.to_datetime(pattern_df["forecast_date"])

        available_dates = sorted(spike_df["forecast_date"].dt.date.unique())

        # ── Top KPIs ─────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Forecast Horizon",
            f"{len(available_dates)} days",
        )
        k2.metric(
            "Zones Monitored",
            f"{spike_df['cluster'].nunique():,}",
        )
        k3.metric(
            "Critical Alerts Tomorrow",
            f"{(dispatch_df[dispatch_df['forecast_date'] == pd.Timestamp(available_dates[0])]['alert_level'] == '🔴 CRITICAL').sum()}",
        )
        k4.metric(
            "High Alerts Tomorrow",
            f"{(dispatch_df[dispatch_df['forecast_date'] == pd.Timestamp(available_dates[0])]['alert_level'] == '🟠 HIGH').sum()}",
        )

        st.markdown("---")

        # ── Section 1: Date selector + Dispatch Map ───────────────
        st.subheader("📅 Daily Deployment Map")

        sel_date = st.date_input(
            "Select forecast date",
            value=available_dates[0],
            min_value=available_dates[0],
            max_value=available_dates[-1],
        )

        day_spike = spike_df[
            spike_df["forecast_date"].dt.date == sel_date
        ].copy()

        if day_spike.empty:
            st.warning("No predictions available for this date.")
        else:
            # Merge display_name
            if "display_name" not in day_spike.columns:
                day_spike = day_spike.merge(
                    cluster_df[["cluster", "display_name"]],
                    on="cluster", how="left"
                )

            col_map, col_stats = st.columns([3, 1])

            with col_map:
                from streamlit_folium import st_folium
                import folium

                # Color by alert level
                ALERT_COLORS = {
                    "🔴 CRITICAL": "red",
                    "🟠 HIGH":     "orange",
                    "🟡 MEDIUM":   "cadetblue",
                    "🟢 LOW":      "green",
                }

                pred_map = folium.Map(
                    location=[
                        day_spike["center_lat"].mean(),
                        day_spike["center_lon"].mean()
                    ],
                    zoom_start=13,
                    tiles="OpenStreetMap",
                )

                for _, row in day_spike.iterrows():
                    prob   = float(row["spike_probability"])
                    alert  = str(row.get("alert_level", "🟢 LOW"))
                    color  = ALERT_COLORS.get(alert, "green")
                    radius = int(5 + prob * 15)  # 5–20px based on probability
                    dname  = str(row.get("display_name", row.get("top_junction", "Unknown")))

                    popup_html = f"""
                    <div style='font-family:sans-serif; width:220px;
                                font-size:13px; color:#111; background:#fff; padding:6px'>
                        <b>{alert}</b><br>
                        <b>Location:</b> {dname}<br>
                        <b>Spike Probability:</b> {prob*100:.1f}%<br>
                        <b>Predicted Spike:</b> {"Yes ⚠️" if row["predicted_spike"] == 1 else "No ✅"}<br>
                        <b>Historical Risk:</b> {float(row.get("risk_score", 0)):.1f} / 10
                    </div>
                    """

                    folium.CircleMarker(
                        location=[row["center_lat"], row["center_lon"]],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.75,
                        weight=2,
                        popup=folium.Popup(popup_html, max_width=240),
                        tooltip=f"{alert} | {prob*100:.0f}% | {dname}",
                    ).add_to(pred_map)

                # Legend
                legend_html = """
                <div style='position:fixed; bottom:30px; left:30px; z-index:1000;
                     background:#fff; padding:10px 14px; border-radius:8px;
                     box-shadow:0 2px 8px rgba(0,0,0,0.25);
                     font-family:sans-serif; font-size:12px; color:#111'>
                    <b>Spike Probability</b><br><br>
                    <span style='color:red'>●</span> CRITICAL (≥ 80%)<br>
                    <span style='color:orange'>●</span> HIGH (60–80%)<br>
                    <span style='color:cadetblue'>●</span> MEDIUM (40–60%)<br>
                    <span style='color:green'>●</span> LOW (&lt; 40%)<br>
                    <br><i style='font-size:11px'>Size ∝ spike probability<br>Click for details</i>
                </div>
                """
                pred_map.get_root().html.add_child(folium.Element(legend_html))

                st_folium(pred_map, use_container_width=True, height=480, returned_objects=[])

            with col_stats:
                st.markdown(f"**{sel_date.strftime('%A, %d %b')}**")
                st.markdown("##### Alert Breakdown")

                tier_counts = day_spike["alert_level"].value_counts()
                for tier in ["🔴 CRITICAL", "🟠 HIGH", "🟡 MEDIUM", "🟢 LOW"]:
                    count = tier_counts.get(tier, 0)
                    st.metric(tier, f"{count} zones")

                st.markdown("---")
                total_predicted = int(day_spike["predicted_spike"].sum())
                avg_prob = float(day_spike["spike_probability"].mean())
                st.metric("Predicted Spikes", total_predicted)
                st.metric("Avg Spike Prob", f"{avg_prob*100:.1f}%")

        st.markdown("---")

        # ── Section 2: Dispatch Briefing Table ───────────────────
        st.subheader("🚨 Deployment Briefing")

        day_dispatch = dispatch_df[
            dispatch_df["forecast_date"].dt.date == sel_date
        ].copy()

        if not day_dispatch.empty:
            # Merge display_name if missing
            if "display_name" not in day_dispatch.columns:
                day_dispatch = day_dispatch.merge(
                    cluster_df[["cluster", "display_name"]],
                    on="cluster", how="left"
                )

            disp_cols = [c for c in [
                "alert_level", "spike_probability",
                "display_name", "recommended_deploy_hour",
                "dispatch_instruction"
            ] if c in day_dispatch.columns]

            styled_dispatch = (
                day_dispatch[disp_cols]
                .sort_values("spike_probability", ascending=False)
                .style
                .format({"spike_probability": "{:.1%}"})
                .background_gradient(
                    subset=["spike_probability"],
                    cmap="RdYlGn_r"
                )
            )
            st.dataframe(styled_dispatch, use_container_width=True, hide_index=True)

            csv_dispatch = day_dispatch[disp_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Dispatch Briefing",
                data=csv_dispatch,
                file_name=f"dispatch_{sel_date}.csv",
                mime="text/csv",
            )
        else:
            st.info("No dispatch instructions for this date. Select a date within the forecast window.")

        st.markdown("---")

        # ── Section 3: Weekly Forecast Heatmap ───────────────────
        st.subheader("📆 7-Day Forecast — Top 20 Clusters")

        col_wl, col_wr = st.columns([4, 1])

        with col_wl:
            # Use saved image if available, else build on the fly
            import os
            if os.path.exists("weekly_forecast_heatmap.png"):
                st.image(
                    "weekly_forecast_heatmap.png",
                    caption="Predicted spike probability: clusters × next 7 days",
                    use_container_width=True,
                )
            else:
                # Build on the fly from weekly_df
                try:
                    top20 = (
                        cluster_df.sort_values("risk_score", ascending=False)
                        .head(20)["cluster"].tolist()
                    )
                    weekly_top = weekly_df[weekly_df["cluster"].isin(top20)].copy()

                    if "display_name" not in weekly_top.columns:
                        weekly_top = weekly_top.merge(
                            cluster_df[["cluster", "display_name"]],
                            on="cluster", how="left"
                        )

                    pivot_pred = weekly_top.pivot_table(
                        index="display_name",
                        columns="forecast_date",
                        values="spike_probability",
                        fill_value=0,
                    )
                    pivot_pred.columns = [
                        pd.Timestamp(c).strftime("%a %d %b")
                        for c in pivot_pred.columns
                    ]

                    fig, ax = plt.subplots(figsize=(14, 7))
                    sns.heatmap(
                        pivot_pred, cmap="RdYlGn_r",
                        annot=True, fmt=".2f",
                        linewidths=0.4,
                        vmin=0, vmax=1, ax=ax,
                    )
                    ax.set_title(
                        "Predicted Spike Probability — Top 20 Clusters × Next 7 Days",
                        fontsize=13, fontweight="bold"
                    )
                    ax.set_xlabel("Forecast Date")
                    ax.set_ylabel("Location")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Could not build weekly heatmap: {e}")

        with col_wr:
            st.markdown("##### Highest Risk Days")
            daily_risk = (
                weekly_df.groupby("forecast_date")["spike_probability"]
                .mean()
                .reset_index()
                .sort_values("spike_probability", ascending=False)
            )
            daily_risk["forecast_date"] = pd.to_datetime(
                daily_risk["forecast_date"]
            ).dt.strftime("%a %d %b")
            daily_risk["spike_probability"] = (
                daily_risk["spike_probability"] * 100
            ).round(1).astype(str) + "%"
            daily_risk.columns = ["Date", "Avg Risk"]
            st.dataframe(daily_risk, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Section 4: Per-Cluster 7-Day Trend ───────────────────
        st.subheader("📈 Per-Cluster Forecast Trend")

        # Merge display_name for selector
        weekly_display = weekly_df.copy()
        if "display_name" not in weekly_display.columns:
            weekly_display = weekly_display.merge(
                cluster_df[["cluster", "display_name"]],
                on="cluster", how="left"
            )

        cluster_options = (
            weekly_display.groupby("cluster")["display_name"]
            .first()
            .reset_index()
        )

        sel_cluster_pred = st.selectbox(
            "Select zone to inspect",
            options=cluster_options["cluster"].tolist(),
            format_func=lambda c: cluster_options[
                cluster_options["cluster"] == c
            ]["display_name"].values[0],
            key="pred_cluster_sel",
        )

        clust_trend = weekly_display[
            weekly_display["cluster"] == sel_cluster_pred
        ].copy()
        clust_trend["forecast_date"] = pd.to_datetime(clust_trend["forecast_date"])
        clust_trend = clust_trend.sort_values("forecast_date")

        if not clust_trend.empty:
            col_tl, col_tr = st.columns([3, 1])

            with col_tl:
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                colors = [
                    "#e63946" if p >= 0.8
                    else "#f4a261" if p >= 0.6
                    else "#a8dadc" if p >= 0.4
                    else "#52b788"
                    for p in clust_trend["spike_probability"]
                ]
                ax2.bar(
                    clust_trend["forecast_date"].dt.strftime("%a\n%d %b"),
                    clust_trend["spike_probability"] * 100,
                    color=colors,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax2.axhline(60, color="orange", linestyle="--",
                            linewidth=1, label="High threshold (60%)")
                ax2.axhline(80, color="red", linestyle="--",
                            linewidth=1, label="Critical threshold (80%)")
                ax2.set_ylabel("Spike Probability (%)")
                ax2.set_ylim(0, 100)
                ax2.legend(fontsize=9)
                ax2.set_title(
                    f"7-Day Spike Forecast — "
                    f"{clust_trend['display_name'].iloc[0]}"
                )
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            with col_tr:
                peak_day = clust_trend.loc[
                    clust_trend["spike_probability"].idxmax()
                ]
                st.markdown("##### Peak Forecast Day")
                st.metric(
                    "Date",
                    pd.Timestamp(peak_day["forecast_date"]).strftime("%a %d %b")
                )
                st.metric(
                    "Spike Probability",
                    f"{float(peak_day['spike_probability'])*100:.1f}%"
                )
                st.metric(
                    "Alert Level",
                    str(peak_day.get("alert_level", "—"))
                )
                risk_score = cluster_df[
                    cluster_df["cluster"] == sel_cluster_pred
                ]["risk_score"].values
                if len(risk_score) > 0:
                    st.metric("Historical Risk Score", f"{float(risk_score[0]):.1f} / 10")