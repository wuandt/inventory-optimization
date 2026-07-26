import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.data.loader import DataLoader

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 300px !important;
    max-width: 300px !important;
    background-color: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 300px !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label p {
    white-space: nowrap !important;
}
.main { background-color: #f8f9fc; }

.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-left: 4px solid #1E90FF;
    margin-bottom: 8px;
}
.kpi-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
.kpi-value { font-size: 24px; font-weight: 600; color: #0f1117; margin-top: 4px; }
.kpi-delta-pos { font-size: 12px; color: #22c55e; margin-top: 2px; }
.kpi-delta-neg { font-size: 12px; color: #ef4444; margin-top: 2px; }
.kpi-delta-neutral { font-size: 12px; color: #6b7280; margin-top: 2px; }
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 8px 0 18px 0;
}
.section-header { font-size: 16px; font-weight: 600; color: #0f1117; margin: 20px 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; }
@media (max-width: 760px) {
    .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }
    .kpi-card { padding: 14px 15px; }
    .kpi-value { font-size: 20px; }
    .section-header { margin-top: 16px; }
}
</style>
""", unsafe_allow_html=True)

BLUE = "#1E90FF"
GRAY = "#A1A6AB"

_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "config", "config.json")
)
with open(_CONFIG_PATH, encoding="utf-8") as config_file:
    APP_CONFIG = json.load(config_file)

UNDERSTOCK_FILL_RATE_MAX = float(
    APP_CONFIG["optimization_scope"]["understock_fill_rate_max"]
)
UNDERSTOCK_FILL_RATE_LABEL = f"{UNDERSTOCK_FILL_RATE_MAX:.0%}"


def render_kpi_cards(cards):
    """Render a row of responsive business KPI cards."""
    card_blocks = []
    for card in cards:
        card_blocks.append(build_kpi_card_html(card))

    all_cards_html = "".join(card_blocks)
    st.markdown(
        f'<div class="kpi-grid">{all_cards_html}</div>',
        unsafe_allow_html=True,
    )


def build_kpi_card_html(card):
    """Build the HTML for one KPI card."""
    card_html = '<div class="kpi-card">'
    card_html += f'<div class="kpi-label">{card["label"]}</div>'
    card_html += f'<div class="kpi-value">{card["value"]}</div>'

    delta = card.get("delta", "")
    if delta:
        delta_class = card.get("delta_class", "kpi-delta-neutral")
        card_html += f'<div class="{delta_class}">{delta}</div>'

    card_html += "</div>"
    return card_html


def safe_pct_change(new_value, old_value):
    """Return percent change while avoiding misleading divide-by-zero results."""
    if not old_value:
        return np.nan
    return 100 * (new_value / old_value - 1)


def hex_to_rgb(hex_color):
    """Convert a six-character hex color to an RGB tuple."""
    clean_color = hex_color.lstrip("#")
    red = int(clean_color[0:2], 16) / 255
    green = int(clean_color[2:4], 16) / 255
    blue = int(clean_color[4:6], 16) / 255
    return red, green, blue


def rgb_to_hex(red, green, blue):
    """Convert RGB values between zero and one back to hex."""
    return "#{:02x}{:02x}{:02x}".format(
        int(red * 255),
        int(green * 255),
        int(blue * 255),
    )


def interpolate_hex_color(start_color, end_color, position):
    """Return one color between the chosen start and end colors."""
    start_red, start_green, start_blue = hex_to_rgb(start_color)
    end_red, end_green, end_blue = hex_to_rgb(end_color)

    red = start_red + (end_red - start_red) * position
    green = start_green + (end_green - start_green) * position
    blue = start_blue + (end_blue - start_blue) * position
    return rgb_to_hex(red, green, blue)


def build_sorted_bar_chart(
    data,
    value_column,
    highlighted_classes,
    title,
    text_formatter,
    yaxis_format=None,
    highlight_color=None,
):
    """Build one class chart with selected classes highlighted."""
    chart_data = data.sort_values(value_column, ascending=False).copy()
    selected_color = highlight_color or BLUE

    bar_colors = []
    for class_name in chart_data["class"]:
        if class_name in highlighted_classes:
            bar_colors.append(selected_color)
        else:
            bar_colors.append(GRAY)

    chart = go.Figure(
        go.Bar(
            x=chart_data["class"],
            y=chart_data[value_column],
            marker_color=bar_colors,
            text=chart_data[value_column].apply(text_formatter),
            textposition="outside",
        )
    )
    chart.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(t=40, b=20),
    )
    if yaxis_format:
        chart.update_yaxes(tickformat=yaxis_format)
    chart.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return chart


def demand_weighted_fill(policy_results):
    """Calculate fulfilled units divided by demanded units."""
    total_demand = policy_results["total_demand"].sum()
    if not total_demand:
        return np.nan
    return policy_results["total_sales"].sum() / total_demand


def summarize_policy(policy_results):
    """Add SKU-level simulation results into one policy summary."""
    return {
        "fill_rate": demand_weighted_fill(policy_results),
        "avg_on_hand_total": policy_results["avg_inventory"].sum(),
        "total_lost": policy_results["total_lost"].sum(),
        "total_cost": policy_results["total_cost"].sum(),
    }


def get_uncertainty_estimate(
    uncertainty,
    metric,
    fallback,
):
    """Read a bootstrap estimate, or use the deterministic fallback."""
    if metric not in uncertainty.index:
        return fallback

    result = uncertainty.loc[metric]
    return float(result["estimate"])


def choose_recommended_action(row):
    """Translate a reorder-point difference into a planner action."""
    material_change = max(1.0, abs(row["current_rop"]) * 0.05)
    if row["rop_delta_units"] > material_change:
        return "Increase ROP"
    if row["rop_delta_units"] < -material_change:
        return "Reduce ROP"
    return "Hold / monitor"


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(manifest_version):
    """Load one internally consistent artifact run.

    ``manifest_version`` is deliberately used only as the cache key. Passing
    the manifest modification time forces Streamlit to reload after a completed
    backend run instead of serving stale CSVs indefinitely.
    """

    del manifest_version
    loader = DataLoader(mode="dashboard")
    loader.load()
    return loader

_MANIFEST_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "metadata",
        "artifact_manifest.json",
    )
)
_MANIFEST_VERSION = (
    os.path.getmtime(_MANIFEST_PATH) if os.path.exists(_MANIFEST_PATH) else 0.0
)
loader = load_data(_MANIFEST_VERSION)

FINAL_TEST_START = pd.Timestamp(APP_CONFIG["data"]["final_test_start"])
HISTORY_START = loader.inventory["date"].min()
HISTORY_END = FINAL_TEST_START - pd.Timedelta(days=1)
EVALUATION_START = loader.forecast["date"].min()
EVALUATION_END = loader.forecast["date"].max()
TOTAL_SKUS = int(loader.sku_metric["sku_id"].nunique())
IN_SCOPE_SKUS = int(loader.policy_sku["sku_id"].nunique())
OUT_OF_SCOPE_SKUS = max(TOTAL_SKUS - IN_SCOPE_SKUS, 0)

EVALUATION_LABEL = (
    f"{EVALUATION_START.date()} to {EVALUATION_END.date()}"
    if pd.notna(EVALUATION_START) and pd.notna(EVALUATION_END)
    else "N/A"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="
            font-size:18px;
            font-weight:700;
            color:#ffffff;
            padding: 10px 0 6px 0;
            white-space: nowrap;
        ">
            📦 Inventory Optimization
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 ABC-XYZ Analysis",
            "📈 Forecast Performance",
            "🎯 Inventory Policy"
        ],
        label_visibility="collapsed"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏠 Overview")

    sku_metric = loader.sku_metric
    sku_class  = loader.sku_class

    st.caption(
        f"Historical performance for all {TOTAL_SKUS} SKUs through "
        f"{HISTORY_END.date()}."
    )

    # Policy exists only for optimization-scope SKUs. Portfolio inventory value
    # and turns therefore use the complete inventory master instead.
    unit_cost_by_sku = loader.inventory.groupby("sku_id")["unit_cost"].median()
    portfolio_metric = sku_metric.set_index("sku_id").copy()
    portfolio_metric["unit_cost"] = unit_cost_by_sku
    portfolio_metric["avg_inventory_value"] = (
        portfolio_metric["avg_inventory"] * portfolio_metric["unit_cost"]
    )
    if "num_days" in portfolio_metric:
        observed_days = portfolio_metric["num_days"].replace(0, np.nan)
    else:
        observed_days = pd.Series(
            (HISTORY_END - HISTORY_START).days + 1,
            index=portfolio_metric.index,
            dtype=float,
        )
    portfolio_metric["avg_daily_demand"] = (
        portfolio_metric["total_demand"] / observed_days
    )
    portfolio_metric["annual_cogs"] = (
        portfolio_metric["avg_daily_demand"]
        * 365
        * portfolio_metric["unit_cost"]
    )
    total_value = portfolio_metric["avg_inventory_value"].sum()
    total_avg_inventory = portfolio_metric["avg_inventory"].sum()
    total_daily_demand = portfolio_metric["avg_daily_demand"].sum()
    portfolio_doi = (
        total_avg_inventory / total_daily_demand if total_daily_demand else np.nan
    )
    value_turnover = (
        portfolio_metric["annual_cogs"].sum() / total_value
        if total_value
        else np.nan
    )
    total_sku      = len(sku_metric)
    slow_moving    = len(sku_metric[sku_metric["DOI"] > 60])
    high_risk      = len(
        sku_metric[sku_metric["fill_rate"] < UNDERSTOCK_FILL_RATE_MAX]
    )
    fill_rate_avg  = sku_metric["total_sales"].sum() / sku_metric["total_demand"].sum()
    render_kpi_cards(
        [
            {
                "label": "Avg on-hand inventory value",
                "value": f"${total_value:,.0f}",
            },
            {
                "label": "Portfolio SKUs",
                "value": f"{total_sku:,}",
            },
            {
                "label": "Days of inventory",
                "value": f"{portfolio_doi:.2f} days",
            },
            {
                "label": "Value-weighted inventory turns",
                "value": f"{value_turnover:.2f}×/yr",
            },
            {
                "label": "Historical actual fill rate",
                "value": f"{fill_rate_avg:.2%}",
            },
        ]
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [
            "Portfolio inventory trend",
            "Inventory by class",
            "Top 10 SKUs by avg on-hand",
        ]
    )

    # ── Tab 1 — Demand trend + alerts + detail tables ─────────────────────────
    with tab1:
        trend = loader.get_inventory_trend()
        fig = px.line(trend, x="date", y="total_inventory", color_discrete_sequence=[BLUE])
        fig.update_layout(
            xaxis_title="", yaxis_title="Total Inventory",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20), hovermode="x unified"
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        # Alert row
        al1, al2 = st.columns(2)
        with al1:
            st.error(f"**{slow_moving}** SKUs with DOI > 60 days — potential slow-moving / excess stock")
        with al2:
            st.warning(
                f"**{high_risk}** SKUs with fill rate < "
                f"{UNDERSTOCK_FILL_RATE_LABEL} — high stockout risk"
            )

        # Detail tables
        tbl1, tbl2 = st.columns(2)

        with tbl1:
            st.markdown(
                '<div class="section-header">🔴 Slow-moving SKUs</div>',
                unsafe_allow_html=True,
            )
            slow_df = (
                sku_metric[sku_metric["DOI"] > 60][["sku_id", "DOI", "avg_inventory", "fill_rate"]]
                .merge(sku_class, on="sku_id", how="left")
                .sort_values("DOI", ascending=False)
                .reset_index(drop=True)
            )
            slow_df["fill_rate"]     = slow_df["fill_rate"].map("{:.2%}".format)
            slow_df["DOI"]           = slow_df["DOI"].round(2)
            slow_df["avg_inventory"] = slow_df["avg_inventory"].round(2)
            st.dataframe(
                slow_df.rename(columns={
                    "sku_id": "SKU", "DOI": "DOI (days)",
                    "avg_inventory": "Avg Inventory", "fill_rate": "Fill Rate",
                    "class": "Class"
                }),
                width="stretch",
                hide_index=True
            )

        with tbl2:
            st.markdown(
                '<div class="section-header">🟡 Stockout-risk SKUs</div>',
                unsafe_allow_html=True,
            )
            risk_df = (
                sku_metric[
                    sku_metric["fill_rate"] < UNDERSTOCK_FILL_RATE_MAX
                ][["sku_id", "fill_rate", "DOI", "lost_sales"]]
                .merge(sku_class, on="sku_id", how="left")
                .sort_values("fill_rate", ascending=True)
                .reset_index(drop=True)
            )
            risk_df["fill_rate"]  = risk_df["fill_rate"].map("{:.2%}".format)
            risk_df["DOI"]        = risk_df["DOI"].round(2)
            risk_df["lost_sales"] = risk_df["lost_sales"].round(2)
            st.dataframe(
                risk_df.rename(columns={
                    "sku_id": "SKU", "fill_rate": "Fill Rate",
                    "DOI": "DOI (days)", "lost_sales": "Lost Demand",
                    "class": "Class"
                }),
                width="stretch",
                hide_index=True
            )

# ── Tab 2 — Distribution by Class ─────────────────────────────────────────
    with tab2:
        # Pre-compute inventory value by class (needed for both charts)
        merged = sku_metric.merge(sku_class, on="sku_id", how="left")
        merged["unit_cost"] = merged["sku_id"].map(unit_cost_by_sku)
        merged["inv_value"] = merged["avg_inventory"] * merged["unit_cost"]
        class_val = (
            merged.groupby("class")["inv_value"]
            .sum()
            .reset_index()
            .sort_values("inv_value", ascending=False)
            .reset_index(drop=True)
        )

        col_a, col_b = st.columns(2)

        with col_a:
            class_count = sku_class["class"].value_counts().reset_index()
            class_count.columns = ["class", "count"]
            class_count = class_count.sort_values("count", ascending=False).reset_index(drop=True)

            n = len(class_count)
            pie_colors = [
                interpolate_hex_color(
                    "#dbeafe",
                    "#1E90FF",
                    1 - index / max(n - 1, 1) * 0.85,
                )
                for index in range(n)
            ]

            fig = go.Figure(go.Pie(
                labels=class_count["class"],
                values=class_count["count"],
                marker=dict(colors=pie_colors, line=dict(color="white", width=1.5)),
                textinfo="label+percent",
                textposition="inside",
                textfont=dict(color="#0f1117", size=11),
                insidetextorientation="horizontal",
                hovertemplate="%{label}: %{value} SKUs (%{percent})<extra></extra>",
                sort=False
            ))
            fig.update_layout(
                title="SKU Distribution by Class",
                paper_bgcolor="white",
                margin=dict(t=40, b=10),
                uniformtext=dict(minsize=8, mode="hide")
            )
            st.plotly_chart(fig, width="stretch")

        with col_b:
            fig = go.Figure(go.Bar(
                x=class_val["class"],
                y=class_val["inv_value"],
                marker_color=BLUE,
                text=class_val["inv_value"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside"
            ))
            fig.update_layout(
                title="Total Inventory Value by Class",
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40, b=10),
                xaxis_title="Class", yaxis_title="Inventory Value ($)"
            )
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, width="stretch")

    # ── Tab 3 — Top 10 SKUs ───────────────────────────────────────────────────
    with tab3:
        top10 = (
            sku_metric.nlargest(10, "avg_inventory")[["sku_id", "avg_inventory", "DOI", "fill_rate"]]
            .sort_values("avg_inventory", ascending=False)
            .reset_index(drop=True)
        )

        fig = go.Figure(go.Bar(
            x=top10["sku_id"],
            y=top10["avg_inventory"],
            marker_color=BLUE,
            text=top10["avg_inventory"].round(2),
            textposition="outside"
        ))
        fig.update_layout(
            title="Top 10 SKUs — Average Inventory",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="SKU", yaxis_title="Avg Inventory (units)",
            margin=dict(t=40, b=20),
            xaxis=dict(categoryorder="total descending", showgrid=False)
        )
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            top10.assign(fill_rate=lambda d: d["fill_rate"].map("{:.2%}".format))
            .rename(columns={
                "sku_id": "SKU", "avg_inventory": "Avg Inventory",
                "DOI": "DOI (days)", "fill_rate": "Fill Rate"
            }),
            width="stretch",
            hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ABC-XYZ ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 ABC-XYZ Analysis":
    st.markdown("# 📊 ABC-XYZ Analysis")
    st.caption(
        "ABC groups SKUs by historical gross-margin contribution; "
        "XYZ groups them by one-day-ahead forecastability."
    )

    sku_class  = loader.sku_class
    sku_metric = loader.sku_metric
    df = sku_metric.merge(sku_class, on="sku_id", how="left")
    df["abc"] = df["class"].str[0]
    df["xyz"] = df["class"].str[1]

    # ABC-XYZ matrix
    st.markdown('<div class="section-header">ABC-XYZ Matrix</div>', unsafe_allow_html=True)
    matrix = df.groupby(["abc", "xyz"])["sku_id"].count().reset_index()
    matrix.columns = ["ABC", "XYZ", "SKU Count"]
    pivot = matrix.pivot(index="ABC", columns="XYZ", values="SKU Count").fillna(0)
    fig = px.imshow(pivot, text_auto=True,
                    color_continuous_scale=[[0, "#f0f4ff"], [1, BLUE]],
                    aspect="auto")
    fig.update_layout(margin=dict(t=20, b=20), paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

    # Metrics table
    st.markdown(
        '<div class="section-header">Historical metrics by class</div>',
        unsafe_allow_html=True,
    )
    summary = loader.get_class_summary().copy()
    summary["avg_fill_rate"]  = (summary["avg_fill_rate"] * 100).round(2).astype(str) + "%"
    summary["avg_doi"]        = summary["avg_doi"].round(2)
    summary["avg_lost_sales"] = summary["avg_lost_sales"].round(2)
    st.dataframe(
        summary.rename(
            columns={
                "class": "Class",
                "sku_count": "SKUs",
                "avg_fill_rate": "Historical fill rate",
                "avg_lost_sales": "Avg lost demand",
                "avg_doi": "Avg DOI (days)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # Problem classes
    st.markdown(
        '<div class="section-header">Policy optimization priorities</div>',
        unsafe_allow_html=True,
    )

    optimization_scope = loader.optimization_scope.drop_duplicates("class").copy()
    strategic_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "protect_strategic_value", "class"
    ].tolist()
    understock_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "correct_understock", "class"
    ].tolist()
    overstock_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "reduce_overstock", "class"
    ].tolist()
    problem_classes = optimization_scope["class"].tolist()

    df_p = df[df["class"].isin(problem_classes)].copy()
    agg  = df_p.groupby("class").agg(
        total_sales=("total_sales", "sum"),
        total_demand=("total_demand", "sum"),
        lost_sales =("lost_sales", "mean"),
        DOI        =("DOI",        "mean")
    ).reset_index()
    agg["fill_rate"] = agg["total_sales"] / agg["total_demand"]

    if strategic_classes:
        strategic_label = ", ".join(strategic_classes)
        st.markdown(f"**{strategic_label} — Protect Strategic Value**")

        col1, col2 = st.columns(2)
        strategic = agg[agg["class"].isin(strategic_classes)].copy()

        with col1:
            fig = build_sorted_bar_chart(
                strategic,
                "fill_rate",
                highlighted_classes=strategic_classes,
                title="Historical demand-weighted fill rate",
                text_formatter=lambda value: f"{value:.2%}",
                yaxis_format=".0%",
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = build_sorted_bar_chart(
                agg, "lost_sales",
                highlighted_classes=strategic_classes,
                title="Average lost demand by class",
                text_formatter=lambda value: f"{value:.2f}"
            )
            st.plotly_chart(fig, width="stretch")

    if understock_classes:
        understock_label = ", ".join(understock_classes)
        st.markdown(f"**{understock_label} — Correct Understock**")

        col3, col4 = st.columns(2)
        with col3:
            fig = build_sorted_bar_chart(
                agg, "lost_sales",
                highlighted_classes=understock_classes,
                title="Average lost demand by class",
                text_formatter=lambda value: f"{value:.2f}"
            )
            st.plotly_chart(fig, width="stretch")

        with col4:
            fig = build_sorted_bar_chart(
                agg, "DOI",
                highlighted_classes=understock_classes,
                title="Average DOI by class",
                text_formatter=lambda value: f"{value:.2f}"
            )
            st.plotly_chart(fig, width="stretch")

    if overstock_classes:
        overstock_label = ", ".join(overstock_classes)
        st.markdown(f"**{overstock_label} — Reduce Overstock**")

        col5, col6 = st.columns(2)
        with col5:
            fig = build_sorted_bar_chart(
                agg, "fill_rate",
                highlighted_classes=overstock_classes,
                title="Historical fill rate by class",
                text_formatter=lambda value: f"{value:.2%}",
                yaxis_format=".0%"
            )
            st.plotly_chart(fig, width="stretch")

        with col6:
            fig = build_sorted_bar_chart(
                agg, "DOI",
                highlighted_classes=overstock_classes,
                title="Average DOI by class",
                text_formatter=lambda value: f"{value:.2f}"
            )
            st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecast Performance":
    st.markdown("# 📈 Forecast Performance")

    forecast_df = loader.forecast
    forecast_metrics = loader.forecast_metrics
    sku_class   = loader.sku_class

    portfolio_metric = forecast_metrics[
        (forecast_metrics["level"] == "portfolio")
        & (forecast_metrics["segment"] == "portfolio")
    ]
    st.caption(
        f"One-day-ahead forecast results for {IN_SCOPE_SKUS} policy-scope SKUs, "
        f"evaluated from {EVALUATION_LABEL}."
    )
    if not portfolio_metric.empty:
        portfolio_row = portfolio_metric.iloc[0]
        render_kpi_cards(
            [
                {
                    "label": "Portfolio WAPE",
                    "value": f"{portfolio_row['WAPE']:.2%}",
                },
                {
                    "label": "Portfolio bias (forecast − actual)",
                    "value": f"{portfolio_row['Bias']:+.2f} units/day",
                },
                {
                    "label": "Portfolio MASE",
                    "value": f"{portfolio_row['MASE']:.3f}",
                },
                {
                    "label": "Portfolio RMSSE",
                    "value": f"{portfolio_row['RMSSE']:.3f}",
                },
            ]
        )

    st.markdown('<div class="section-header">SKU drill-down</div>', unsafe_allow_html=True)
    sku_list     = sorted(forecast_df["sku_id"].unique().tolist())
    selected_sku = st.selectbox("Select SKU", sku_list)

    sku_data  = loader.get_forecast_by_sku(selected_sku)
    cls_label = sku_class[sku_class["sku_id"] == selected_sku]["class"].values
    cls_label = cls_label[0] if len(cls_label) > 0 else "N/A"

    y_true   = sku_data["demand"]
    y_pred   = sku_data["forecast"]
    wape_denominator = y_true.abs().sum()
    sku_wape = (
        (y_true - y_pred).abs().sum() / wape_denominator
        if wape_denominator
        else np.nan
    )
    bias     = (y_pred - y_true).mean()
    sku_metric_row = forecast_metrics[
        (forecast_metrics["level"] == "sku")
        & (forecast_metrics["segment"] == selected_sku)
    ]
    sku_mase = sku_metric_row["MASE"].iloc[0] if not sku_metric_row.empty else np.nan
    sku_rmsse = sku_metric_row["RMSSE"].iloc[0] if not sku_metric_row.empty else np.nan

    st.caption(
        f"SKU {selected_sku} · Class {cls_label} · "
        f"{sku_data['date'].min().date()} to {sku_data['date'].max().date()}"
    )

    render_kpi_cards(
        [
            {
                "label": "SKU WAPE",
                "value": f"{sku_wape:.2%}" if pd.notna(sku_wape) else "N/A",
            },
            {
                "label": "SKU bias (forecast − actual)",
                "value": f"{bias:+.2f} units/day",
            },
            {
                "label": "SKU MASE",
                "value": f"{sku_mase:.3f}" if pd.notna(sku_mase) else "N/A",
            },
            {
                "label": "SKU RMSSE",
                "value": f"{sku_rmsse:.3f}" if pd.notna(sku_rmsse) else "N/A",
            },
        ]
    )

    st.markdown('<div class="section-header">Actual vs Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["demand"],
        name="Actual demand", line=dict(color="#0f1117", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["forecast"],
        name="Forecast", line=dict(color=BLUE, width=1.5, dash="dash")
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="", yaxis_title="Daily demand (units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=20), hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POLICY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Inventory Policy":
    st.markdown("# 🎯 Inventory Policy")

    old_df = loader.old_policy
    new_df = loader.new_policy
    st.caption(
        f"Both policies use the same starting inventory and demand from "
        f"{EVALUATION_LABEL}. Current policy uses settings recorded in raw data."
    )

    old_portfolio = summarize_policy(old_df)
    new_portfolio = summarize_policy(new_df)
    uncertainty = (
        loader.policy_uncertainty.set_index("metric")
        if not loader.policy_uncertainty.empty
        else pd.DataFrame()
    )

    fill_delta = get_uncertainty_estimate(
        uncertainty,
        "fill_rate_delta_percentage_points",
        100 * (new_portfolio["fill_rate"] - old_portfolio["fill_rate"]),
    )
    inventory_delta = get_uncertainty_estimate(
        uncertainty,
        "avg_inventory_change_pct",
        safe_pct_change(
            new_portfolio["avg_on_hand_total"],
            old_portfolio["avg_on_hand_total"],
        ),
    )
    cost_delta = get_uncertainty_estimate(
        uncertainty,
        "total_cost_change_pct",
        safe_pct_change(new_portfolio["total_cost"], old_portfolio["total_cost"]),
    )
    lost_delta = safe_pct_change(
        new_portfolio["total_lost"], old_portfolio["total_lost"]
    )

    if not loader.full_policy_summary.empty:
        st.markdown(
            f'<div class="section-header">Full-portfolio policy comparison '
            f'({TOTAL_SKUS} SKUs)</div>',
            unsafe_allow_html=True,
        )
        full_summary_columns = [
            "policy_label",
            "sku_count",
            "total_lost",
            "fill_rate",
            "sum_sku_avg_inventory",
            "total_cost",
        ]
        full_summary_display = loader.full_policy_summary[
            full_summary_columns
        ].copy()
        policy_names = {
            "recorded_policy_all_skus": "Current policy",
            "proposed_in_scope_recorded_out_of_scope": (
                f"Proposed policy ({IN_SCOPE_SKUS} optimized + "
                f"{OUT_OF_SCOPE_SKUS} unchanged)"
            ),
        }
        full_summary_display["policy_label"] = full_summary_display[
            "policy_label"
        ].replace(policy_names)
        full_summary_display["fill_rate"] = pd.to_numeric(
            full_summary_display["fill_rate"], errors="coerce"
        ).map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
        for column in ["sku_count", "total_lost", "sum_sku_avg_inventory"]:
            numeric_values = pd.to_numeric(
                full_summary_display[column], errors="coerce"
            )
            full_summary_display[column] = numeric_values.map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )
        total_cost_values = pd.to_numeric(
            full_summary_display["total_cost"], errors="coerce"
        )
        full_summary_display["total_cost"] = total_cost_values.map(
            lambda value: "" if pd.isna(value) else f"${value:,.0f}"
        )
        st.dataframe(
            full_summary_display.rename(
                columns={
                    "policy_label": "Scenario",
                    "sku_count": "SKUs",
                    "total_lost": "Lost demand",
                    "fill_rate": "Simulated fill rate",
                    "sum_sku_avg_inventory": "Avg portfolio on-hand",
                    "total_cost": "Modeled total cost",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        f'<div class="section-header">Impact on {IN_SCOPE_SKUS} optimized '
        f'SKUs</div>',
        unsafe_allow_html=True,
    )
    render_kpi_cards(
        [
            {
                "label": "Simulated fill rate",
                "value": f"{new_portfolio['fill_rate']:.2%}",
                "delta": f"{fill_delta:+.2f} pts vs current policy",
                "delta_class": (
                    "kpi-delta-pos" if fill_delta >= 0 else "kpi-delta-neg"
                ),
            },
            {
                "label": "Avg portfolio on-hand",
                "value": f"{new_portfolio['avg_on_hand_total']:,.0f} units",
                "delta": f"{inventory_delta:+.2f}% vs current policy",
                "delta_class": (
                    "kpi-delta-pos" if inventory_delta <= 0 else "kpi-delta-neg"
                ),
            },
            {
                "label": "Modeled total cost",
                "value": f"${new_portfolio['total_cost']:,.0f}",
                "delta": f"{cost_delta:+.2f}% vs current policy",
                "delta_class": (
                    "kpi-delta-pos" if cost_delta <= 0 else "kpi-delta-neg"
                ),
            },
            {
                "label": "Lost demand",
                "value": f"{new_portfolio['total_lost']:,.0f}",
                "delta": f"{lost_delta:+.2f}% vs current policy",
                "delta_class": (
                    "kpi-delta-pos" if lost_delta <= 0 else "kpi-delta-neg"
                ),
            },
        ]
    )

    st.caption(
        "Simulation results on synthetic data; values are not realized savings."
    )

    if not loader.scenario_uncertainty.empty:
        scenario_frame = loader.scenario_uncertainty
        st.markdown(
            f'<div class="section-header">Stress-test results '
            f'({len(scenario_frame):,} scenarios)</div>',
            unsafe_allow_html=True,
        )
        scenario_cards = []

        def add_scenario_distribution(
            column,
            label,
            unit,
            positive_is_favorable,
        ):
            if column not in scenario_frame:
                return
            values = pd.to_numeric(
                scenario_frame[column], errors="coerce"
            ).dropna()
            if values.empty:
                return
            median = float(values.median())
            lower = float(values.quantile(0.05))
            upper = float(values.quantile(0.95))
            favorable = (
                median >= 0 if positive_is_favorable else median <= 0
            )
            scenario_cards.append(
                {
                    "label": label,
                    "value": f"{median:+.2f}{unit}",
                    "delta": (
                        f"90% scenario range: {lower:+.2f} to "
                        f"{upper:+.2f}{unit}"
                    ),
                    "delta_class": (
                        "kpi-delta-pos" if favorable else "kpi-delta-neg"
                    ),
                }
            )

        add_scenario_distribution(
            "fill_rate_delta_percentage_points",
            "Median fill-rate change",
            " pts",
            True,
        )
        add_scenario_distribution(
            "avg_inventory_change_pct",
            "Median inventory change",
            "%",
            False,
        )
        add_scenario_distribution(
            "total_cost_change_pct",
            "Median total-cost change",
            "%",
            False,
        )
        if "total_cost_savings" in scenario_frame:
            savings = pd.to_numeric(
                scenario_frame["total_cost_savings"], errors="coerce"
            )
            valid_savings = savings.notna()
            if valid_savings.any():
                favorable_share = float(
                    savings.loc[valid_savings].gt(0).mean()
                )
                if "scenario_probability" in scenario_frame:
                    probabilities = pd.to_numeric(
                        scenario_frame["scenario_probability"],
                        errors="coerce",
                    )
                    valid_probabilities = probabilities.loc[valid_savings]
                    if (
                        valid_probabilities.notna().all()
                        and valid_probabilities.ge(0).all()
                        and valid_probabilities.sum() > 0
                    ):
                        favorable_share = float(
                            valid_probabilities.loc[
                                savings.loc[valid_savings].gt(0)
                            ].sum()
                            / valid_probabilities.sum()
                        )
                scenario_cards.append(
                    {
                        "label": "Scenarios with lower cost",
                        "value": f"{favorable_share:.1%}",
                    }
                )
        if scenario_cards:
            render_kpi_cards(scenario_cards)
        with st.expander(
            f"View {len(scenario_frame):,} stress-test scenarios"
        ):
            st.dataframe(
                scenario_frame,
                width="stretch",
                hide_index=True,
            )

    optional_sensitivity = [
        ("Cost assumptions", loader.policy_sensitivity),
        (
            "Current policy assumptions",
            loader.historical_policy_sensitivity,
        ),
    ]
    if any(not frame.empty for _, frame in optional_sensitivity):
        with st.expander("Sensitivity details"):
            for label, frame in optional_sensitivity:
                if frame.empty:
                    continue
                st.caption(label)
                st.dataframe(frame, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-header">Class summary</div>',
        unsafe_allow_html=True,
    )
    scope_intervention = (
        loader.optimization_scope[["class", "intervention"]]
        .drop_duplicates("class")
        .set_index("class")["intervention"]
        .to_dict()
    )
    class_rows = []
    available_classes = sorted(
        set(old_df["class"]).intersection(set(new_df["class"]))
    )
    for class_name in available_classes:
        old_class = summarize_policy(old_df[old_df["class"] == class_name])
        new_class = summarize_policy(new_df[new_df["class"] == class_name])
        class_rows.append(
            {
                "class": class_name,
                "intervention": scope_intervention.get(class_name, "in_scope"),
                "skus": int(
                    loader.policy_sku.loc[
                        loader.policy_sku["class"] == class_name, "sku_id"
                    ].nunique()
                ),
                "old_fill_rate": old_class["fill_rate"],
                "new_fill_rate": new_class["fill_rate"],
                "fill_delta_pp": 100
                * (new_class["fill_rate"] - old_class["fill_rate"]),
                "inventory_change_pct": safe_pct_change(
                    new_class["avg_on_hand_total"], old_class["avg_on_hand_total"]
                ),
                "cost_change_pct": safe_pct_change(
                    new_class["total_cost"], old_class["total_cost"]
                ),
            }
        )
    class_view = pd.DataFrame(class_rows)
    class_display = class_view.copy()
    intervention_names = {
        "protect_strategic_value": "Protect strategic value",
        "correct_understock": "Correct understock",
        "reduce_overstock": "Reduce overstock",
        "in_scope": "In scope",
    }
    class_display["intervention"] = class_display["intervention"].replace(
        intervention_names
    )
    for column in ["old_fill_rate", "new_fill_rate"]:
        class_display[column] = class_display[column].map("{:.2%}".format)
    for column in [
        "fill_delta_pp",
        "inventory_change_pct",
        "cost_change_pct",
    ]:
        class_display[column] = class_display[column].map("{:+.2f}".format)
    st.dataframe(
        class_display.rename(
            columns={
                "class": "Class",
                "intervention": "Intervention",
                "skus": "SKUs",
                "old_fill_rate": "Current policy fill",
                "new_fill_rate": "Proposed policy fill",
                "fill_delta_pp": "Fill Δ (pts)",
                "inventory_change_pct": "On-hand Δ (%)",
                "cost_change_pct": "Cost Δ (%)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-header">SKU actions</div>',
        unsafe_allow_html=True,
    )
    if not loader.policy_action.empty:
        action_table = loader.policy_action.copy()
    else:
        # Fallback action layer: recorded master parameters describe the current
        # implementation delta, while old/new metric files retain the simulator's
        # recorded-policy comparison.
        current_parameters = (
            loader.inventory[loader.inventory["date"] < FINAL_TEST_START]
            .groupby("sku_id", observed=True)
            .agg(
                current_rop=("reorder_point", "median"),
                current_safety_stock=("safety_stock", "median"),
                current_order_quantity=("order_quantity", "median"),
                unit_cost=("unit_cost", "median"),
            )
            .reset_index()
        )
        action_table = (
            loader.policy_sku[
                [
                    "sku_id",
                    "class",
                    "target_service_level",
                    "safety_stock",
                    "rop",
                    "order_quantity",
                ]
            ]
            .rename(
                columns={
                    "safety_stock": "proposed_safety_stock",
                    "rop": "proposed_rop",
                    "order_quantity": "proposed_order_quantity",
                }
            )
            .merge(current_parameters, on="sku_id", validate="one_to_one")
            .merge(
                old_df[
                    [
                        "sku_id",
                        "fill_rate",
                        "avg_inventory",
                        "total_cost",
                    ]
                ].rename(
                    columns={
                        "fill_rate": "old_fill_rate",
                        "avg_inventory": "old_avg_inventory",
                        "total_cost": "old_total_cost",
                    }
                ),
                on="sku_id",
                validate="one_to_one",
            )
            .merge(
                new_df[
                    [
                        "sku_id",
                        "fill_rate",
                        "avg_inventory",
                        "total_cost",
                    ]
                ].rename(
                    columns={
                        "fill_rate": "new_fill_rate",
                        "avg_inventory": "new_avg_inventory",
                        "total_cost": "new_total_cost",
                    }
                ),
                on="sku_id",
                validate="one_to_one",
            )
        )
        action_table["rop_delta_units"] = (
            action_table["proposed_rop"] - action_table["current_rop"]
        )
        action_table["safety_stock_delta_units"] = (
            action_table["proposed_safety_stock"]
            - action_table["current_safety_stock"]
        )
        action_table["fill_delta_pp"] = 100 * (
            action_table["new_fill_rate"] - action_table["old_fill_rate"]
        )
        action_table["avg_on_hand_delta_units"] = (
            action_table["new_avg_inventory"]
            - action_table["old_avg_inventory"]
        )
        action_table["avg_on_hand_value_delta"] = (
            action_table["avg_on_hand_delta_units"] * action_table["unit_cost"]
        )
        action_table["modeled_cost_saving"] = (
            action_table["old_total_cost"] - action_table["new_total_cost"]
        )

        action_table["recommended_action"] = action_table.apply(
            choose_recommended_action,
            axis=1,
        )
        action_table = action_table.sort_values(
            ["modeled_cost_saving", "fill_delta_pp"],
            ascending=[False, False],
        ).reset_index(drop=True)
    if "selection_status" in action_table.columns:
        review_count = int(
            action_table["selection_status"]
            .ne("minimum_cost_feasible")
            .sum()
        )
        if review_count:
            st.warning(
                f"{review_count} SKUs need planner review because none of the "
                "tested policies reached the required calibration fill rate."
            )
    filter_columns = st.columns(2)
    with filter_columns[0]:
        action_class = st.selectbox(
            "Action class",
            ["All"] + sorted(action_table["class"].dropna().unique().tolist())
            if "class" in action_table
            else ["All"],
        )
    action_column = None
    if "recommended_action" in action_table:
        action_column = "recommended_action"
    elif "action" in action_table:
        action_column = "action"
    with filter_columns[1]:
        action_filter = st.selectbox(
            "Action type",
            ["All"]
            + (
                sorted(action_table[action_column].dropna().unique().tolist())
                if action_column
                else []
            ),
        )
    filtered_actions = action_table.copy()
    if action_class != "All" and "class" in filtered_actions:
        filtered_actions = filtered_actions[
            filtered_actions["class"] == action_class
        ]
    if (
        action_filter != "All"
        and action_column
        and action_column in filtered_actions
    ):
        filtered_actions = filtered_actions[
            filtered_actions[action_column] == action_filter
        ]

    preferred_action_columns = [
        "sku_id",
        "class",
        "recommended_action",
        "current_rop",
        "proposed_rop",
        "current_order_quantity",
        "proposed_order_quantity",
        "fill_delta_pp",
        "modeled_cost_saving",
        "selection_status",
    ]
    visible_action_columns = [
        column
        for column in preferred_action_columns
        if column in filtered_actions.columns
    ]
    action_display = (
        filtered_actions[visible_action_columns].copy()
        if visible_action_columns
        else filtered_actions.copy()
    )
    numeric_action_columns = action_display.select_dtypes(include="number").columns
    action_display[numeric_action_columns] = action_display[
        numeric_action_columns
    ].round(2)
    if "selection_status" in action_display:
        action_display["selection_status"] = action_display[
            "selection_status"
        ].replace(
            {
                "minimum_cost_feasible": "Ready",
                "no_feasible_candidate_best_service": "Planner review",
            }
        )
    action_display = action_display.rename(
        columns={
            "sku_id": "SKU",
            "class": "Class",
            "recommended_action": "Recommended action",
            "current_rop": "Current ROP",
            "proposed_rop": "Proposed ROP",
            "current_order_quantity": "Current order qty",
            "proposed_order_quantity": "Proposed order qty",
            "fill_delta_pp": "Fill change (pts)",
            "modeled_cost_saving": "Modeled cost saving",
            "selection_status": "Status",
        }
    )
    st.dataframe(action_display, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered SKU actions",
        data=filtered_actions.to_csv(index=False).encode("utf-8"),
        file_name="inventory_policy_sku_actions.csv",
        mime="text/csv",
        width="stretch",
    )