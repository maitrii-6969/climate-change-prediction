import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Climate impact prediction", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(ROOT, "outputs", "maps")
DATA = os.path.join(ROOT, "data", "dashboard")

INK = "#1f2933"
PALETTE = ["#2f6b5e", "#b9852c", "#9b4a3a", "#4a6fa5", "#7a5c8e",
           "#5b8a72", "#c2693a", "#6b7280"]
LEVEL_COLOURS = {"CRITICAL": "#9b2c2c", "HIGH": "#c2693a",
                 "MODERATE": "#c9a227", "LOW": "#5b8a72"}

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
h1 {font-size: 1.9rem; font-weight: 600;}
h2, h3 {font-weight: 600;}
section[data-testid="stSidebar"] {border-right: 1px solid #e3e0d8;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def load_rain():
    path = os.path.join(DATA, "imd_city_monthly.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["month"])


@st.cache_data
def load_annual():
    path = os.path.join(DATA, "imd_india_annual.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def synthetic_ndvi():
    rng = np.random.default_rng(42)
    n = 120
    t = np.arange(n)
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    series = {
        "Western Ghats (forest)": np.clip(.72 - .05 * t / n + .18 * np.sin((t + 9) * np.pi / 6) + rng.normal(0, .03, n), 0, 1),
        "Punjab (agriculture)": np.clip(.55 + .02 * t / n + .25 * np.sin((t + 9) * np.pi / 6) + rng.normal(0, .05, n), 0, 1),
        "Rajasthan (arid)": np.clip(.18 - .02 * t / n + .07 * np.sin((t + 9) * np.pi / 6) + rng.normal(0, .02, n), 0, 1),
    }
    return dates, series


COORDS = {
    "W. Rajasthan": (26.9, 72.2), "E. Rajasthan": (26.5, 75.8),
    "Vidarbha": (20.7, 78.6), "Marathwada": (19.5, 76.5),
    "Konkan": (17.0, 73.3), "Bundelkhand": (25.0, 79.5),
    "Malwa": (23.1, 75.8), "Saurashtra": (22.3, 71.2),
    "N. Gujarat": (24.0, 72.5), "N. Karnataka": (15.3, 75.7),
    "S. Karnataka": (12.9, 77.5), "Rayalaseema": (14.6, 79.3),
    "Coastal AP": (16.5, 81.0), "N. Telangana": (18.1, 79.0),
    "W. Odisha": (20.9, 83.3), "Coastal Odisha": (20.0, 86.0),
    "Jharkhand": (23.6, 85.3), "Chhattisgarh": (21.3, 81.6),
    "W. UP": (27.0, 79.0), "E. UP": (25.5, 82.5),
    "S. Bihar": (24.8, 85.0), "W. Bengal": (22.9, 87.8),
    "S. Haryana": (29.0, 76.0), "Punjab": (30.9, 75.8),
    "NCR Delhi": (28.6, 77.2), "N. Tamil Nadu": (12.9, 79.2),
    "S. Tamil Nadu": (10.5, 77.5), "Kerala": (10.5, 76.2),
    "W. Ghats Forest": (14.0, 75.5), "Assam": (26.2, 92.9),
    "Uttarakhand": (30.3, 78.0), "HP Hills": (31.1, 77.2),
    "J&K": (34.1, 77.6), "Meghalaya": (25.5, 91.4),
    "Manipur": (24.6, 93.9), "Mizoram": (23.1, 92.9),
    "Nagaland": (26.1, 94.6), "Tripura": (23.9, 91.9),
    "Arunachal": (27.1, 93.6), "Sikkim": (27.5, 88.5),
    "Goa": (15.3, 74.1),
}


@st.cache_data
def load_scores():
    path = os.path.join(MAPS, "predicted_anomaly_scores.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["lat"] = df["region"].map(lambda r: COORDS.get(r, (20.0, 78.0))[0])
    df["lon"] = df["region"].map(lambda r: COORDS.get(r, (20.0, 78.0))[1])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def tidy(fig, height=340):
    fig.update_layout(
        template="simple_white", height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=INK, size=13),
        legend=dict(orientation="h", y=1.12, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def show_image(title, filename):
    path = os.path.join(MAPS, filename)
    if os.path.exists(path):
        st.markdown(f"**{title}**")
        st.image(Image.open(path))


# ----------------------------------------------------------------- pages
def page_overview():
    st.title("Climate impact prediction for India")
    st.write(
        "We are building a pipeline that predicts land surface temperature and "
        "vegetation indicators from satellite and reanalysis data. Models are "
        "validated on a chronological split: train 2015-2021, validation 2022, "
        "test 2023-2024."
    )

    annual = load_annual()
    c1, c2, c3 = st.columns(3)
    if annual is not None:
        wet = annual.loc[annual["india_mean_mm"].idxmax()]
        dry = annual.loc[annual["india_mean_mm"].idxmin()]
        c1.metric("Wettest year (IMD)", f"{int(wet['year'])} ({wet['india_mean_mm']:.0f} mm)")
        c2.metric("Driest year (IMD)", f"{int(dry['year'])} ({dry['india_mean_mm']:.0f} mm)")
    c3.metric("Rainfall record", "2015-2024")

    st.subheader("Where the data stands")
    st.dataframe(
        pd.DataFrame([
            ["IMD gridded rainfall", "In use", "Daily, 0.25 degree, 2015-2024"],
            ["ERA5 reanalysis", "Needs re-download", "12 months missing from our file"],
            ["MODIS NDVI / EVI / LST", "Downloading", "5 km monthly composites via Earth Engine"],
            ["ISRO Resourcesat / INSAT-3DR", "Requested", "Waiting on NRSC and MOSDAC"],
        ], columns=["Source", "Status", "Note"]),
        hide_index=True,
    )

    with st.expander("Pipeline test series (synthetic, not real observations)"):
        dates, series = synthetic_ndvi()
        fig = go.Figure()
        for k, (name, y) in enumerate(series.items()):
            fig.add_trace(go.Scatter(x=dates, y=y, name=name,
                                     line=dict(color=PALETTE[k], width=1.8)))
        fig.add_vrect(x0="2022-01-01", x1="2023-01-01", fillcolor="#b9852c", opacity=0.08, line_width=0)
        fig.add_vrect(x0="2023-01-01", x1="2025-01-01", fillcolor="#9b4a3a", opacity=0.08, line_width=0)
        st.plotly_chart(tidy(fig))
        st.caption("Shaded: validation (2022) and test (2023-2024).")


def page_rainfall():
    st.title("Rainfall (IMD gridded data)")
    rain = load_rain()
    if rain is None:
        st.info("No data yet. Run scripts/11_export_dashboard_data.py and push the data/dashboard folder.")
        return

    cities = sorted(rain["city"].unique())
    picked = st.multiselect("Cities", cities, default=["Mumbai", "Delhi", "Chennai"])
    if not picked:
        st.info("Pick at least one city.")
        return
    view = rain[rain["city"].isin(picked)].copy()
    view["year"] = view["month"].dt.year

    st.subheader("Monthly rainfall")
    fig = go.Figure()
    for k, city in enumerate(picked):
        d = view[view["city"] == city]
        fig.add_trace(go.Scatter(x=d["month"], y=d["rainfall_mm"], name=city,
                                 line=dict(color=PALETTE[k % len(PALETTE)], width=1.6)))
    for y in range(2015, 2025):
        fig.add_vrect(x0=f"{y}-06-01", x1=f"{y}-10-01", fillcolor="#2f6b5e",
                      opacity=0.07, line_width=0)
    fig.update_yaxes(title="mm per month")
    st.plotly_chart(tidy(fig))
    st.caption("Shaded bands mark June to September. Each city is the average of the 3x3 grid cells around it.")

    st.subheader("Annual total by city")
    totals = view.groupby(["city", "year"])["rainfall_mm"].sum().reset_index()
    fig = go.Figure()
    for k, city in enumerate(picked):
        d = totals[totals["city"] == city]
        fig.add_trace(go.Bar(x=d["year"], y=d["rainfall_mm"], name=city,
                             marker_color=PALETTE[k % len(PALETTE)]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="mm per year")
    st.plotly_chart(tidy(fig))

    st.subheader("How much falls in the monsoon")
    view["monsoon"] = view["month"].dt.month.between(6, 9)
    tot = view.groupby("city")["rainfall_mm"].sum()
    mon = view[view["monsoon"]].groupby("city")["rainfall_mm"].sum()
    table = pd.DataFrame({
        "Mean annual rainfall (mm)": (tot / view["year"].nunique()).round(0),
        "Share in Jun-Sep (%)": (mon / tot * 100).round(0),
    }).reset_index().rename(columns={"city": "City"})
    st.dataframe(table, hide_index=True)

    st.download_button("Download this data (CSV)",
                       view.drop(columns=["monsoon"]).to_csv(index=False),
                       file_name="imd_city_monthly.csv")


def page_models():
    st.title("Model results")
    st.write("Scores are on the held-out test period, 2023-2024.")

    st.dataframe(
        pd.DataFrame([
            ["XGBoost", "Synthetic NDVI", "0.0441", "0.0338", "0.9709", "Pipeline test only"],
            ["CNN-LSTM", "Synthetic NDVI", "0.0442", "0.0325", "0.9722", "Pipeline test only"],
            ["XGBoost", "ERA5 skin temperature", "1.216 C", "0.917 C", "0.9277", "Re-running"],
            ["XGBoost", "ERA5 + IMD rainfall", "1.242 C", "0.923 C", "0.9246", "Re-running"],
        ], columns=["Model", "Data", "RMSE", "MAE", "R2", "Status"]),
        hide_index=True,
    )
    st.markdown(
        "- The synthetic rows only prove the code runs end to end. We generated that data, so the scores say nothing about real climate.\n"
        "- The real-data rows are being re-run. Our ERA5 file is missing 12 months and the two datasets were joined by position instead of by date.\n"
        "- Still to add: a seasonal-naive baseline (same month last year) to show what the models gain over seasonality alone."
    )

    with st.expander("Plots"):
        show_image("XGBoost, actual vs predicted (synthetic)", "xgboost_predictions.png")
        show_image("CNN-LSTM training loss (synthetic)", "cnn_lstm_loss_curve.png")
        show_image("CNN-LSTM, actual vs predicted (synthetic)", "cnn_lstm_predictions.png")
        show_image("XGBoost feature importance (synthetic)", "xgboost_feature_importance.png")


def page_map():
    st.title("Drought risk by region")
    st.caption("Demo page. These scores come from a model trained on synthetic regional series, so they are not a real drought assessment.")
    scores = load_scores()
    if scores is None:
        st.info("No scores found. Run scripts/07_retrain_all_states.py first.")
        return

    level = st.selectbox("Alert level", ["All"] + list(LEVEL_COLOURS))
    view = scores if level == "All" else scores[scores["level"] == level]

    fig = go.Figure()
    for lvl, colour in LEVEL_COLOURS.items():
        sub = view[view["level"] == lvl]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], mode="markers", name=lvl.title(),
            text=sub["state"] + " / " + sub["region"] + "<br>score " + sub["score"].round(2).astype(str),
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sub["score"].clip(lower=0.4) * 10, color=colour,
                        opacity=0.8, line=dict(width=0.5, color="white")),
        ))
    fig.update_geos(
        scope="asia", center=dict(lat=22, lon=82), projection_scale=4.5,
        showland=True, landcolor="#efede6", showocean=True, oceancolor="#e3edf2",
        showcountries=True, countrycolor="#9ca3af", showsubunits=True,
        subunitcolor="#d1d5db", showcoastlines=True, coastlinecolor="#9ca3af",
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=0))
    st.plotly_chart(fig)
    st.dataframe(view[["state", "region", "score", "level"]], hide_index=True)


def page_about():
    st.title("About")
    st.markdown(
        """
**Project:** Climate Change Impact Prediction using ISRO satellite and remote sensing data
**Institute:** Manipal Institute of Technology, Manipal
**Guide:** Prof. Pooja S., Department of CSE
**Team:** Maitreya Rajesh Tarmale (230929306), Raghav Bandral

**Method**
- Features: lagged and rolling temperature and rainfall, cyclical month encoding
- Models: XGBoost baseline, CNN-LSTM
- Validation: chronological split, no shuffling

**Known limitations**
- ISRO data has been requested but not received, so nothing here is ISRO-derived yet
- The drought map and the CNN-LSTM results use synthetic data
- ERA5-based results are being re-run after a date-alignment fix

**Code:** github.com/maitrii-6969/climate-change-prediction
"""
    )


PAGES = {
    "Overview": page_overview,
    "Rainfall (real data)": page_rainfall,
    "Model results": page_models,
    "Drought risk map (demo)": page_map,
    "About": page_about,
}

with st.sidebar:
    st.header("Climate impact prediction")
    st.caption("MIT Manipal, Data Science minor")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")

PAGES[choice]()