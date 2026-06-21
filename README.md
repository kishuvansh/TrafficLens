# 🚦 trafficlens — AI-Driven Parking Congestion Intelligence

> Turning reactive traffic enforcement into predictive deployment using machine learning on real-world violation data from Bengaluru, India.

---
 ##Deployed link : https://trafficlens-ipfhyjvnrspjwazrw5gcmf.streamlit.app/
## 📌 Problem Statement

On-street illegal parking near commercial areas, metro stations, and busy junctions chokes carriageways and intersections across Bengaluru. Current enforcement is entirely **patrol-based and reactive** — officers respond after congestion has already formed, with no visibility into where or when violations are likely to peak.

**trafficlens** addresses this by:
- Detecting illegal parking hotspots from historical violation records
- Quantifying their congestion impact through a composite risk score
- Predicting future violation spikes using a trained ML classifier
- Generating actionable, time-specific deployment instructions for traffic police

---

## ✨ Key Features

- **Spatial Clustering** — DBSCAN groups 298K+ violation records into named enforcement zones
- **Composite Risk Scoring** — 4-dimension score (frequency, severity, peak-hour timing, junction proximity) using quantile normalisation to handle skewed distributions
- **Anomaly Detection** — IsolationForest flags unusual violation spikes by zone and week
- **Predictive Dispatch** — RandomForest classifier predicts next-7-day spike probability per cluster
- **Reverse Geocoding** — Nominatim resolves unnamed road-segment clusters to real street names
- **Interactive Dashboard** — 6-tab Streamlit app with live folium maps, temporal heatmaps, and downloadable briefings

---

## 🗂️ Project Structure
```
trafficlens/
├── trafficdata.csv              # Raw input — violation records (expected CSV)
├── eda.ipynb                    # Exploratory Data Analysis
├── phase1.ipynb                 # Data parsing, feature engineering, initial clustering
├── phase2.py                    # Core ML pipeline — risk scoring, DBSCAN, anomaly detection
├── phase3.ipynb                 # Predictive layer — pattern forecast + spike classifier
├── app1.py                      # Streamlit dashboard (6 tabs)
├── enforcement_hotspots.html    # Standalone folium heatmap (phase1 output)
├── df_engineered.csv            # Full feature-engineered dataset
├── cluster_data.csv             # DBSCAN clusters with risk scores and metadata
├── hotspot_report.csv           # Ranked enforcement priority table
├── temporal_peaks.csv           # Hourly violation counts per cluster
├── anomaly_log.csv              # IsolationForest detected spikes
├── pattern_forecast.csv         # 7-day pattern-based forecast (phase3)
├── spike_predictions.csv        # ML spike probabilities per cluster per day
├── weekly_forecast.csv          # Aggregated weekly forecast for top clusters
└── dispatch_briefing.csv        # Deployment instructions for dispatch teams
```

---

## 🔄 Workflow Overview

Follow this order when running the project. Each phase depends on outputs from the previous one.

1. phase1 — Exploratory parsing & feature engineering
    - Parse `violation_type`, flatten into binary features
    - Extract temporal features (`hour`, `day_of_week`, `is_peak_hour`)
    - Compute row-level `priority_score` and generate an initial heatmap

2. phase2 — Core processing & scoring
    - Spatial clustering (DBSCAN)
    - Cluster aggregation, composite `risk_score` and `risk_tier`
    - Temporal peak analysis and anomaly detection (IsolationForest)
    - Export CSVs: `df_engineered.csv`, `cluster_data.csv`, `hotspot_report.csv`, `temporal_peaks.csv`, `anomaly_log.csv`

3. phase3 — Predictive layer & dispatch briefing
    - Build 24×7 pattern matrices per cluster
    - Pattern-based 7-day forecast and RandomForest spike classifier
    - Produce `spike_predictions.csv`, `pattern_forecast.csv`, `dispatch_briefing.csv`

4. app1.py — Visualization
    - Streamlit dashboard that reads the CSV outputs and presents interactive maps, temporal analyses, anomaly logs, and downloadable enforcement reports.

Simple pipeline diagram:

```
trafficdata.csv -> phase1 -> phase2 -> phase3 -> app1.py (dashboard)
```

---

## 📁 File Descriptions

### `eda.ipynb` — Exploratory Data Analysis
Initial investigation of the raw dataset. Covers:
- Schema inspection (24 columns, 298,450 rows, 100% lat/lon coverage)
- Missing value audit (`description`, `action_taken_timestamp`, `closed_datetime` are 100% null)
- Distribution analysis of `vehicle_type`, `violation_type`, `junction_name`
- Datetime range confirmation: Nov 2023 → Apr 2024
- Identification of `violation_type` as stringified JSON lists requiring parsing

### `phase1.ipynb` — Data Parsing & Feature Engineering
Transforms raw data into ML-ready features:
- **`MultiLabelBinarizer`** — explodes 991 violation type combinations into 27 binary columns (`is_wrong_parking`, `is_no_parking`, etc.)
- **Temporal features** — extracts `hour`, `day_of_week`, `is_weekend` from `created_datetime`
- **`priority_score`** — weighted severity score per violation record (e.g. `PARKING IN A MAIN ROAD` = 3, `WRONG PARKING` = 2)
- **Initial DBSCAN** — proof-of-concept spatial clustering
- **Folium heatmap** — generates `enforcement_hotspots.html` weighted by `priority_score`

### `phase2_model.py` — Core ML Pipeline
The primary processing script. Run once to generate all downstream CSVs:

| Step | What it does |
|------|-------------|
| 1 | Load & parse `violation_type` JSON arrays via `MultiLabelBinarizer` |
| 2 | Temporal feature engineering (hour, weekday, peak hour flag, month, week) |
| 3 | Priority score with 1.5× peak-hour multiplier |
| 4 | DBSCAN clustering (`eps=30m`, `min_samples=15`) → 869 clusters |
| 5 | Cluster aggregation + composite risk score (quantile normalised) |
| 6 | Temporal peak analysis per cluster |
| 7 | IsolationForest anomaly detection on zone×week×hour grain |
| 8 | Enforcement priority report with human-readable `recommended_action` |
| 9 | Interactive folium map v2 with heatmap + colour-coded cluster markers |
| 10 | Export all CSVs |
| 11 | Reverse geocoding of "No Junction" clusters via Nominatim |

**Risk Score Formula:**
risk_score = (0.40 × freq_quantile + 0.30 × severity_quantile

+ 0.20 × sqrt(peak_hour_frac) + 0.10 × junction_flag) × 10

**Risk Tiers:**

| Tier | Score Range |
|------|------------|
| 🔴 CRITICAL | ≥ 7.0 |
| 🟠 HIGH | 4.0 – 7.0 |
| 🟡 MEDIUM | 2.0 – 4.0 |
| 🟢 LOW | < 2.0 |

### `phase3.ipynb` — Predictive Layer
Two-layer prediction engine:

**Layer 1 — Pattern-Based Forecast**
- Builds a 24×7 (hour × weekday) violation intensity matrix per cluster
- Computes weekly averages and flags slots above the 75th percentile as high-activity
- Generates a 7-day forward forecast from historical patterns alone

**Layer 2 — RandomForest Spike Classifier**
- Target: `will_spike` = 1 if daily violations ≥ cluster's 75th percentile
- Features: `day_of_week`, `is_weekend`, `week`, `avg_priority`, `peak_hour_frac`, `rolling_7_mean`, `rolling_7_std`, `rolling_7_max`, `cluster_enc`
- Uses **temporal train/test split** (80/20 chronological) — no data leakage
- `class_weight="balanced"` to handle spike/non-spike imbalance
- Outputs spike probability (0–1) per cluster per forecast day
- Generates `dispatch_briefing.csv` with deploy hour + action per zone

### `app1.py` — Streamlit Dashboard
6-tab interactive dashboard reading all phase2/phase3 CSV outputs:

| Tab | Content |
|-----|---------|
| 📊 Overview | KPI cards, violations over time, vehicle/violation/station breakdowns |
| 🗺️ Hotspot Map | Live folium map with risk-tiered cluster markers + heatmap overlay + enforcement table |
| ⏰ Temporal Patterns | Global hour×weekday heatmap, hourly bar chart, per-cluster drill-down |
| 🚨 Anomaly Alerts | IsolationForest spike log, hour/week distribution charts |
| 🔍 Data Explorer | Filterable raw data table with CSV download |
| 🔮 Predictive Dispatch | Date-picker forecast map, dispatch briefing table, 7-day heatmap, per-cluster trend |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/trafficlens.git
cd trafficlens
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `trafficdata.csv` in the project root directory.

---

## ▶️ How to Run

> **Run in this order.** Each phase depends on outputs from the previous one.

### Step 1 — EDA (optional but recommended)
```bash
jupyter notebook eda.ipynb
```

### Step 2 — Feature Engineering
```bash
jupyter notebook phase1.ipynb
# Run all cells top to bottom
```

### Step 3 — ML Pipeline (generates all CSVs)
```bash
python phase2_model.py
```
> ⏱️ Takes 3–5 minutes. The Nominatim geocoding step (~590 clusters at 1.1s each) takes ~11 minutes additionally.

### Step 4 — Predictive Layer
```bash
jupyter notebook phase3.ipynb
# Run all cells top to bottom
```

### Step 5 — Launch Dashboard
```bash
streamlit run app1.py
```
Open `http://localhost:8501` in your browser.

---

## 📦 Dependencies

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
folium>=0.14
streamlit>=1.28
streamlit-folium>=0.15
geopy>=2.3
pydeck>=0.8
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium streamlit streamlit-folium geopy pydeck
```

---

## 📥 Input / Output

### Input
| File | Description |
|------|-------------|
| `trafficdata.csv` | Raw violation records — 298,450 rows × 24 columns. Key fields: `latitude`, `longitude`, `violation_type`, `created_datetime`, `junction_name`, `vehicle_type`, `police_station` |

### Outputs

| File | Description |
|------|-------------|
| `df_engineered.csv` | Feature-engineered full dataset with 27 binary violation columns, temporal features, priority score, cluster assignment |
| `cluster_data.csv` | 869 spatial clusters with centroid coordinates, risk score, risk tier, peak hour/day, display name |
| `hotspot_report.csv` | Ranked enforcement zones with recommended actions |
| `temporal_peaks.csv` | Hourly violation counts per cluster for temporal charts |
| `anomaly_log.csv` | Detected anomalous violation spikes with context |
| `pattern_forecast.csv` | 7-day ahead pattern-based violation forecasts per cluster |
| `spike_predictions.csv` | ML-predicted spike probability per cluster × forecast day |
| `weekly_forecast.csv` | Aggregated weekly forecast for top 20 clusters |
| `dispatch_briefing.csv` | Actionable deployment instructions for the next day |
| `enforcement_hotspots.html` | Standalone interactive heatmap (no server needed) |

---

## 📊 Dataset

- **Source:** Bengaluru Traffic Police (BTP) violation records via HackerEarth
- **Period:** November 2023 – April 2024 (5 months)
- **Size:** 298,450 records × 24 columns
- **Coverage:** 54 police stations, 169 named junctions, 100% GPS coordinates

---

## ⚠️ Limitations

- **No real-time data** — predictions are pattern-based from historical records; live camera or sensor feeds would significantly improve accuracy
- **No external signals** — events, weather, and road construction are not modelled
- **Enforcement response gap** — `action_taken_timestamp` and `closed_datetime` are 100% null in the dataset, so enforcement effectiveness cannot be measured
- **Geocoding rate limit** — Nominatim enforces 1 request/second; reverse geocoding ~590 clusters takes ~11 minutes
- **Forecast horizon** — the RandomForest model predicts 7 days ahead; accuracy degrades beyond 3–4 days without retraining on fresh data
- **Bengaluru-specific** — junction names, police station codes, and spatial parameters are tuned for Bengaluru's geography

---

## 🚀 Future Improvements

- **Live camera integration** — YOLO-based real-time illegal parking detection from CCTV feeds
- **External data fusion** — incorporate event calendars, weather APIs, and road closure feeds
- **Retraining pipeline** — automated weekly model retraining as new violation data arrives
- **Mobile dispatch app** — push notifications to patrol officers based on predicted hotspots
- **Enforcement feedback loop** — track patrol response and outcomes to measure deterrence impact
- **City generalisation** — parameterise spatial and temporal thresholds for deployment in other Indian cities

---

## 👥 Team

Built for the **Flipkart GRiDLock 2.0 Hackathon** — Round 2.

---

## 📄 License

This project is for academic and hackathon purposes. Violation data is anonymised and provided by the hackathon organiser.
