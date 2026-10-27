"""Interactive Streamlit dashboard for the E-commerce Store Insights EDA.

Start from this folder with: streamlit run app.py
Dataset expected: ecommerce_sales_data.csv (or upload the CSV in the sidebar).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Store Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#19324F"
TEAL = "#0D9488"
BLUE = "#5276E8"
GOLD = "#E7A83E"
PALETTE = [TEAL, BLUE, GOLD, "#DA6D7C", "#8B72D4", "#4DA3BE", "#7FA45B"]
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_BY_MONTH = {
    "December": "Winter", "January": "Winter", "February": "Winter",
    "March": "Spring", "April": "Spring", "May": "Spring",
    "June": "Summer", "July": "Summer", "August": "Summer",
    "September": "Autumn", "October": "Autumn", "November": "Autumn",
}

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      .stApp { background: #F5F8FC; color: #19324F; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1,h2,h3 { font-family: 'Manrope','DM Sans',sans-serif !important; color:#19324F; }
      .hero { padding:1.45rem 1.7rem; border-radius:20px; color:#fff;
        background:linear-gradient(115deg,#19324F 0%,#176A72 58%,#0D9488 100%);
        margin-bottom:1rem; box-shadow:0 12px 28px rgba(25,50,79,.13); }
      .hero-kicker { font-size:.75rem; font-weight:700; letter-spacing:.15em;
        text-transform:uppercase; opacity:.78; }
      .hero-title { color:#fff; font-family:'Manrope',sans-serif; font-size:2rem;
        font-weight:800; letter-spacing:-.04em; margin:.3rem 0; }
      .hero-subtitle { color:#E5F5F3; margin:0; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #E4EBF3;
        border-radius:15px; padding:.9rem 1rem; box-shadow:0 3px 12px #19324F0A; }
      [data-testid="stMetricLabel"] { color:#64748B; font-weight:600; }
      [data-testid="stMetricValue"] { color:#19324F; font-family:'Manrope',sans-serif; }
      div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #E4EBF3;
        border-radius:15px; padding:.25rem; }
      section[data-testid="stSidebar"] { background:#EFF4F9; border-right:1px solid #E2EAF2; }
      .insight { background:#EFF9F7; border:1px solid #D7ECE8; border-radius:14px;
        padding:.85rem 1rem; margin:.35rem 0 .75rem; color:#244C50; }
      .muted { color:#64748B; font-size:.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Dataset access and preparation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=None, engine="python")


@st.cache_data(show_spinner=False)
def read_local_csv(path: str, mtime_ns: int, file_size: int) -> pd.DataFrame:
    # File size and mtime are part of Streamlit's cache key.
    del mtime_ns, file_size
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


def find_local_csvs() -> list[Path]:
    root = Path(__file__).resolve().parent
    candidates = []
    for folder in (root, root / "data", root / "streamlit", root / "streamlit" / "data"):
        if folder.is_dir():
            candidates.extend(p.resolve() for p in folder.glob("*.csv") if p.is_file())
    return list(dict.fromkeys(candidates))


def normalize_and_derive(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names. Rename duplicates and upload again.")

    required = [
        "Year", "Month", "Category", "Region", "Units_Sold", "Revenue",
        "Profit", "Customer_Rating", "Ad_Spend",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "This dashboard expects the EDA dataset columns: "
            + ", ".join(required)
            + ". Missing: " + ", ".join(missing)
        )

    df = df.dropna(how="all").reset_index(drop=True)
    for column in ("Year", "Units_Sold", "Revenue", "Profit", "Customer_Rating", "Ad_Spend"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Month"] = df["Month"].astype("string").str.strip().str.title()
    df["Date"] = pd.to_datetime(
        df["Year"].astype("Int64").astype("string") + "-" + df["Month"].astype("string"),
        format="%Y-%B",
        errors="coerce",
    )
    # Keep month ordering stable in tables and charts.
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    df["Season"] = df["Month"].astype("string").map(SEASON_BY_MONTH)
    df["Season"] = pd.Categorical(df["Season"], categories=SEASON_ORDER, ordered=True)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype("string")
    df["Profit_Margin"] = df["Profit"] / df["Revenue"].replace(0, np.nan)
    df["Price_Per_Unit"] = df["Revenue"] / df["Units_Sold"].replace(0, np.nan)
    df["ROAS"] = df["Revenue"] / df["Ad_Spend"].replace(0, np.nan)
    return df


# -----------------------------------------------------------------------------
# Formatting and chart helpers
# -----------------------------------------------------------------------------
def fmt_money(value: float, symbol: str) -> str:
    return "—" if pd.isna(value) else f"{symbol}{value:,.2f}"


def style_chart(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=16, t=38, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=NAVY, size=12),
        title=dict(font=dict(family="Manrope, sans-serif", size=16, color=NAVY), x=.02),
        hoverlabel=dict(bgcolor="white", font_color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E6ECF3", zerolinecolor="#E6ECF3")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F7", zerolinecolor="#E6ECF3")
    return fig


def show_plot(fig, height: int = 360):
    st.plotly_chart(
        style_chart(fig, height),
        use_container_width=True,
        config={"displayModeBar": "hover", "scrollZoom": True},
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def category_summary(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    result = frame.groupby(dimension, observed=True).agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Ad_Spend=("Ad_Spend", "sum"),
        Customer_Rating=("Customer_Rating", "mean"),
        Records=("Revenue", "size"),
    ).reset_index()
    result["Profit_Margin"] = result["Profit"] / result["Revenue"].replace(0, np.nan)
    result["ROAS"] = result["Revenue"] / result["Ad_Spend"].replace(0, np.nan)
    return result


def notebook_baseline_matches(frame: pd.DataFrame) -> bool:
    """Recognize the exact full dataset summarized in the saved EDA outputs."""
    if len(frame) != 1000:
        return False
    expected = {
        "Units_Sold": 263337.0,
        "Revenue": 27123544.52,
        "Profit": 5494732.66,
        "Ad_Spend": 2649554.61,
    }
    return all(
        np.isclose(frame[column].sum(), total, rtol=0, atol=.02)
        for column, total in expected.items()
    )


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🛍️ Store insights")
st.sidebar.caption("Interactive view of the EDA sales dataset")
uploaded = st.sidebar.file_uploader("Upload ecommerce_sales_data.csv", type=["csv"])
local_files = find_local_csvs()
local_file = None
if uploaded is None and local_files:
    selected_file = st.sidebar.selectbox("Or use a local CSV", local_files, format_func=lambda p: p.name)
    local_file = selected_file

if uploaded is not None:
    source_name = uploaded.name
    raw_data = read_csv_bytes(uploaded.getvalue())
elif local_file is not None:
    source_name = local_file.name
    stat = local_file.stat()
    raw_data = read_local_csv(str(local_file), stat.st_mtime_ns, stat.st_size)
else:
    raw_data = None
    source_name = ""

if raw_data is None:
    st.markdown(
        "<div class='hero'><div class='hero-kicker'>Commerce analytics</div>"
        "<p class='hero-title'>E-commerce Store Insights</p>"
        "<p class='hero-subtitle'>Sales, profitability, seasonality, regional performance, marketing efficiency, and customer ratings.</p></div>",
        unsafe_allow_html=True,
    )
    st.info(
        "The notebook references `/content/ecommerce_sales_data.csv`, but that CSV is not present in this workspace. "
        "Upload the same CSV from the sidebar to populate the interactive dashboard."
    )
    with st.expander("Notebook's saved full-dataset summary"):
        st.markdown(
            "The notebook output reports **1,000 rows**, no missing values, and no duplicate rows. "
            "It records **263,337 units sold**, **27,123,544.52 revenue**, **5,494,732.66 profit**, "
            "and **2,649,554.61 ad spend**. Mean customer rating is **3.04 / 5**. "
            "These figures are reference values from the notebook output; interactive charts require the CSV."
        )
    st.stop()

try:
    data = normalize_and_derive(raw_data)
except Exception as exc:
    st.error(f"Could not prepare the uploaded data: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
years = sorted(data["Year"].dropna().astype(int).unique().tolist())
if years:
    year_range = st.sidebar.slider(
        "Year range",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=(int(years[0]), int(years[-1])),
    )
else:
    year_range = None

def filter_select(column: str, label: str):
    options = sorted(data[column].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=[])

selected_categories = filter_select("Category", "Category")
selected_regions = filter_select("Region", "Region")
selected_months = filter_select("Month", "Month")
selected_seasons = filter_select("Season", "Season")

rating_min = float(data["Customer_Rating"].min()) if data["Customer_Rating"].notna().any() else 0.0
rating_max = float(data["Customer_Rating"].max()) if data["Customer_Rating"].notna().any() else 5.0
if rating_min < rating_max:
    rating_range = st.sidebar.slider(
        "Customer rating",
        min_value=rating_min,
        max_value=rating_max,
        value=(rating_min, rating_max),
        step=.1,
    )
else:
    rating_range = (rating_min, rating_max)

currency_label = st.sidebar.selectbox("Value display", ["No symbol", "USD ($)", "EUR (€)", "GBP (£)"])
currency_symbol = {"No symbol": "", "USD ($)": "$", "EUR (€)": "€", "GBP (£)": "£"}[currency_label]
trend_metric = st.sidebar.selectbox("Trend measure", ["Revenue", "Profit", "Units_Sold", "Ad_Spend"])
top_n = st.sidebar.slider("Top categories/products shown", 3, 10, 6)

filtered = data.copy()
if year_range:
    filtered = filtered[filtered["Year"].between(year_range[0], year_range[1])]
if selected_categories:
    filtered = filtered[filtered["Category"].astype(str).isin(selected_categories)]
if selected_regions:
    filtered = filtered[filtered["Region"].astype(str).isin(selected_regions)]
if selected_months:
    filtered = filtered[filtered["Month"].astype(str).isin(selected_months)]
if selected_seasons:
    filtered = filtered[filtered["Season"].astype(str).isin(selected_seasons)]
if rating_min < rating_max:
    filtered = filtered[filtered["Customer_Rating"].between(rating_range[0], rating_range[1])]


# -----------------------------------------------------------------------------
# Header and KPI cards
# -----------------------------------------------------------------------------
st.markdown(
    f"<div class='hero'><div class='hero-kicker'>Store performance · {len(filtered):,} selected records</div>"
    "<p class='hero-title'>E-commerce Store Insights</p>"
    f"<p class='hero-subtitle'>Dataset: <b>{source_name}</b> · Explore category, region, season, and marketing patterns.</p></div>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No rows match the current filters. Expand the year range or clear one of the sidebar filters.")
    st.stop()

revenue_total = filtered["Revenue"].sum()
profit_total = filtered["Profit"].sum()
units_total = filtered["Units_Sold"].sum()
ad_spend_total = filtered["Ad_Spend"].sum()
weighted_margin = safe_ratio(profit_total, revenue_total)
overall_roas = safe_ratio(revenue_total, ad_spend_total)
mean_rating = filtered["Customer_Rating"].mean()
avg_revenue_record = filtered["Revenue"].mean()

kpi_row_1 = st.columns(4)
kpi_row_1[0].metric("Revenue", fmt_money(revenue_total, currency_symbol))
kpi_row_1[1].metric("Profit", fmt_money(profit_total, currency_symbol))
kpi_row_1[2].metric("Units sold", f"{units_total:,.0f}")
kpi_row_1[3].metric("Records", f"{len(filtered):,}")
kpi_row_2 = st.columns(4)
kpi_row_2[0].metric("Profit margin", f"{weighted_margin:.1%}" if pd.notna(weighted_margin) else "—")
kpi_row_2[1].metric("ROAS", f"{overall_roas:.2f}×" if pd.notna(overall_roas) else "—")
kpi_row_2[2].metric("Avg. customer rating", f"{mean_rating:.2f} / 5" if pd.notna(mean_rating) else "—")
kpi_row_2[3].metric("Ad spend", fmt_money(ad_spend_total, currency_symbol))

if notebook_baseline_matches(data) and len(filtered) == 1000:
    st.caption("Loaded CSV matches the notebook's saved 1,000-row totals. Average revenue per record: "
               f"{fmt_money(avg_revenue_record, currency_symbol)}. The source has no order ID, so this is not labeled as average order value.")
else:
    st.caption("Revenue per record is not shown as average order value because the dataset has no order identifier.")

overview_tab, product_tab, marketing_tab, quality_tab = st.tabs(
    ["Overview", "Category & region", "Marketing & ratings", "Data quality"]
)


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
with overview_tab:
    st.subheader("Performance at a glance")
    highlights = []
    by_category = category_summary(filtered, "Category").sort_values("Revenue", ascending=False)
    by_region = category_summary(filtered, "Region").sort_values("Revenue", ascending=False)
    if not by_category.empty:
        top = by_category.iloc[0]
        share = safe_ratio(top["Revenue"], revenue_total)
        highlights.append(
            f"**{top['Category']}** is the leading category with {fmt_money(top['Revenue'], currency_symbol)} "
            f"({share:.1%} of selected revenue)."
        )
    if not by_region.empty:
        top_region = by_region.iloc[0]
        highlights.append(
            f"**{top_region['Region']}** is the highest-revenue region at {fmt_money(top_region['Revenue'], currency_symbol)}."
        )
    by_season = filtered.groupby("Season", observed=True)["Revenue"].sum().reindex(SEASON_ORDER).dropna()
    if not by_season.empty:
        highlights.append(f"**{by_season.idxmax()}** has the highest combined seasonal revenue in this selection.")
    if highlights:
        st.markdown("<div class='insight'>" + "<br>".join(f"• {text}" for text in highlights) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown(f"#### {trend_metric.replace('_', ' ')} over time")
        monthly = (
            filtered.groupby("Date", dropna=True)[trend_metric]
            .sum().sort_index().rename("Value").reset_index()
        )
        fig = px.line(
            monthly, x="Date", y="Value", markers=True,
            color_discrete_sequence=[TEAL],
            labels={"Date": "Month", "Value": trend_metric.replace("_", " ")},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=6), fill="tozeroy", fillcolor="rgba(13,148,136,.08)")
        fig.update_xaxes(tickformat="%b %Y")
        if trend_metric in ("Revenue", "Profit", "Ad_Spend"):
            fig.update_yaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 370)

    with right:
        st.markdown("#### Revenue by season")
        season_data = filtered.groupby("Season", observed=True)["Revenue"].sum().reindex(SEASON_ORDER).dropna().reset_index()
        fig = px.bar(
            season_data, x="Season", y="Revenue", color="Season",
            color_discrete_sequence=PALETTE,
            category_orders={"Season": SEASON_ORDER},
            labels={"Revenue": "Revenue"},
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 370)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Monthly revenue and profit")
        compare = filtered.groupby("Date")[ ["Revenue", "Profit"] ].sum().sort_index().reset_index()
        compare_long = compare.melt(id_vars="Date", var_name="Measure", value_name="Value")
        fig = px.line(
            compare_long, x="Date", y="Value", color="Measure", markers=True,
            color_discrete_map={"Revenue": TEAL, "Profit": BLUE},
        )
        fig.update_xaxes(tickformat="%b %Y")
        fig.update_yaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 340)
    with col_b:
        st.markdown("#### Revenue by month of year")
        month_data = filtered.groupby("Month", observed=True)["Revenue"].sum().reindex(MONTH_ORDER).dropna().reset_index()
        fig = px.bar(
            month_data, x="Month", y="Revenue", color="Revenue",
            color_continuous_scale=["#CDEBE6", TEAL],
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 340)


# -----------------------------------------------------------------------------
# Product/category and geography
# -----------------------------------------------------------------------------
with product_tab:
    st.subheader("Category & regional performance")
    left, right = st.columns(2)
    category = category_summary(filtered, "Category").sort_values("Revenue", ascending=False)
    region = category_summary(filtered, "Region").sort_values("Revenue", ascending=False)

    with left:
        st.markdown("#### Category revenue")
        fig = px.bar(
            category.sort_values("Revenue"), x="Revenue", y="Category", orientation="h",
            color="Profit_Margin", color_continuous_scale=["#F2C7BF", "#E9D38A", TEAL],
            hover_data={"Profit": ":,.2f", "Profit_Margin": ":.1%", "Units_Sold": ":,.0f", "ROAS": ":.2f"},
        )
        fig.update_layout(coloraxis_colorbar_title="Profit margin", coloraxis_colorbar_tickformat=".0%")
        fig.update_xaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 390)

    with right:
        st.markdown("#### Category share of revenue")
        fig = px.pie(
            category, names="Category", values="Revenue", hole=.6,
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False)
        show_plot(fig, 390)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Revenue by region")
        fig = px.bar(
            region.sort_values("Revenue"), x="Revenue", y="Region", orientation="h",
            color="Revenue", color_continuous_scale=["#C8D7FC", BLUE],
            hover_data={"Profit": ":,.2f", "Units_Sold": ":,.0f", "Customer_Rating": ":.2f", "ROAS": ":.2f"},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 340)

    with col_b:
        st.markdown("#### Profit margin by category")
        margin_data = category.sort_values("Profit_Margin")
        fig = px.bar(
            margin_data, x="Profit_Margin", y="Category", orientation="h",
            color="Profit_Margin", color_continuous_scale=["#F2C7BF", "#E9D38A", TEAL],
            labels={"Profit_Margin": "Profit margin"},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        show_plot(fig, 340)

    st.markdown("#### Category scorecard")
    scorecard = category.copy()
    scorecard["Profit_Margin"] = scorecard["Profit_Margin"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    scorecard["ROAS"] = scorecard["ROAS"].map(lambda x: f"{x:.2f}×" if pd.notna(x) else "—")
    scorecard["Customer_Rating"] = scorecard["Customer_Rating"].map(lambda x: f"{x:.2f} / 5" if pd.notna(x) else "—")
    scorecard["Revenue"] = scorecard["Revenue"].map(lambda x: fmt_money(x, currency_symbol))
    scorecard["Profit"] = scorecard["Profit"].map(lambda x: fmt_money(x, currency_symbol))
    st.dataframe(
        scorecard[["Category", "Revenue", "Profit", "Profit_Margin", "Units_Sold", "Ad_Spend", "ROAS", "Customer_Rating", "Records"]],
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Marketing, rating, and profit distributions
# -----------------------------------------------------------------------------
with marketing_tab:
    st.subheader("Marketing efficiency & customer ratings")
    ad_ratio = safe_ratio(revenue_total, ad_spend_total)
    st.caption(
        f"Overall ROAS is calculated as total revenue ÷ total ad spend ({ad_ratio:.2f}× when defined). "
        "The scatter plot shows association and does not imply that ad spend caused revenue changes."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Ad spend vs. revenue")
        fig = px.scatter(
            filtered, x="Ad_Spend", y="Revenue", color="Category", symbol="Region",
            size="Units_Sold", size_max=16, opacity=.72,
            hover_data={"Profit": ":,.2f", "Customer_Rating": ":.1f", "ROAS": ":.2f", "Date": True},
            color_discrete_sequence=PALETTE,
        )
        fig.update_xaxes(tickprefix=currency_symbol, separatethousands=True, title="Advertising spend")
        fig.update_yaxes(tickprefix=currency_symbol, separatethousands=True, title="Revenue")
        show_plot(fig, 390)

    with right:
        st.markdown("#### ROAS by category")
        roas_category = category_summary(filtered, "Category").sort_values("ROAS", ascending=True)
        fig = px.bar(
            roas_category, x="ROAS", y="Category", orientation="h",
            color="ROAS", color_continuous_scale=["#F0D99A", TEAL],
            labels={"ROAS": "Revenue / ad spend"},
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(ticksuffix="×")
        show_plot(fig, 390)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Customer ratings by category")
        rating_data = filtered.groupby("Category", observed=True)["Customer_Rating"].mean().sort_values().reset_index()
        fig = px.bar(
            rating_data, x="Customer_Rating", y="Category", orientation="h",
            color="Customer_Rating", color_continuous_scale=["#F0C2BA", "#E8D68F", TEAL],
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, 5], title="Average rating (out of 5)")
        show_plot(fig, 350)

    with col_b:
        st.markdown("#### Profit distribution by category")
        fig = px.box(
            filtered, x="Profit", y="Category", color="Category", points="outliers",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickprefix=currency_symbol, separatethousands=True)
        show_plot(fig, 350)

    rating_corr = filtered[["Customer_Rating", "Revenue", "Profit"]].corr(method="spearman")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean customer rating", f"{filtered['Customer_Rating'].mean():.2f} / 5")
    c2.metric("Rating ↔ revenue (Spearman)", f"{rating_corr.loc['Customer_Rating', 'Revenue']:.2f}")
    c3.metric("Rating ↔ profit (Spearman)", f"{rating_corr.loc['Customer_Rating', 'Profit']:.2f}")
    st.caption("Correlation summarizes association in the filtered records; it does not establish causation.")


# -----------------------------------------------------------------------------
# Data quality and download
# -----------------------------------------------------------------------------
with quality_tab:
    st.subheader("Data quality & filtered records")
    duplicate_count = int(filtered.duplicated().sum())
    missing_cells = int(filtered.isna().sum().sum())
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Filtered rows", f"{len(filtered):,}")
    q2.metric("Columns", f"{filtered.shape[1]:,}")
    q3.metric("Missing cells", f"{missing_cells:,}")
    q4.metric("Duplicate rows", f"{duplicate_count:,}")

    quality = pd.DataFrame({
        "Column": filtered.columns.astype(str),
        "Type": [str(filtered[column].dtype) for column in filtered.columns],
        "Missing": [int(filtered[column].isna().sum()) for column in filtered.columns],
        "Missing %": [float(filtered[column].isna().mean() * 100) for column in filtered.columns],
        "Unique values": [int(filtered[column].nunique(dropna=True)) for column in filtered.columns],
    })
    st.dataframe(quality.style.format({"Missing %": "{:.1f}%"}), use_container_width=True, hide_index=True)

    st.markdown("#### Filtered data preview")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered dataset as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="ecommerce_sales_filtered.csv",
        mime="text/csv",
    )

st.markdown(
    "<p class='muted' style='text-align:center;margin-top:1.25rem'>"
    "E-commerce Store Insights · Revenue and ROAS respond to the selected filters</p>",
    unsafe_allow_html=True,
)
