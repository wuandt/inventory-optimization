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

[data-testid="stSidebar"] { background-color: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
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
.section-header { font-size: 16px; font-weight: 600; color: #0f1117; margin: 20px 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

BLUE   = "#1E90FF"
GRAY   = "#A1A6AB"
RED    = "#EF4444"

SERVICE_LEVEL = {"AX": 99, "AY": 97, "CX": 85, "BZ": 85, "CZ": 75}

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    loader = DataLoader(mode="dashboard")
    loader.load()
    return loader

loader = load_data()

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
    policy_sku = loader.policy_sku
    sku_class  = loader.sku_class

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    policy_indexed = policy_sku.set_index("sku_id")
    total_value = (
        sku_metric.set_index("sku_id")["avg_inventory"]
        .mul(policy_indexed["unit_cost"])
        .sum()
    )
    total_sku      = len(sku_metric)
    slow_moving    = len(sku_metric[sku_metric["DOI"] > 60])
    high_risk      = len(sku_metric[sku_metric["fill_rate"] < 0.95])
    fill_rate_avg  = sku_metric["fill_rate"].mean()
    turnover       = 365 / sku_metric["DOI"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Total Inventory Value", f"${total_value:,.0f}",                   None, None),
        (c2, "Total SKUs",            f"{total_sku:,}",                          None, None),
        (c3, "Avg DOI",               f"{sku_metric['DOI'].mean():.2f} days",    None, None),
        (c4, "Inventory Turnover",    f"{turnover:.2f}×/yr",                     None, None),
        (c5, "Avg Fill Rate",         f"{fill_rate_avg:.2%}",                    None, None)
    ]
    for col, label, value, delta, delta_type in kpi_data:
        with col:
            delta_html = ""
            if delta:
                css_class = "kpi-delta-pos" if delta_type == "pos" else "kpi-delta-neg"
                delta_html = f'<div class="{css_class}">{delta}</div>'
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Inventory Trend", "Distribution by Class", "Top avg inventory SKUs"])

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
        st.plotly_chart(fig, use_container_width=True)

        # Alert row
        al1, al2 = st.columns(2)
        with al1:
            st.error(f"**{slow_moving}** SKUs with DOI > 60 days — potential slow-moving / excess stock")
        with al2:
            st.warning(f"**{high_risk}** SKUs with fill rate < 95% — high stockout risk")

        st.markdown("")

        # Detail tables
        tbl1, tbl2 = st.columns(2)

        with tbl1:
            st.markdown('<div class="section-header">🔴 Slow-Moving SKUs (DOI > 60 days)</div>', unsafe_allow_html=True)
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
                use_container_width=True,
                hide_index=True
            )

        with tbl2:
            st.markdown('<div class="section-header">🟡 High Stockout-Risk SKUs (Fill Rate < 95%)</div>', unsafe_allow_html=True)
            risk_df = (
                sku_metric[sku_metric["fill_rate"] < 0.95][["sku_id", "fill_rate", "DOI", "lost_sales"]]
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
                    "DOI": "DOI (days)", "lost_sales": "Lost Sales",
                    "class": "Class"
                }),
                use_container_width=True,
                hide_index=True
            )

# ── Tab 2 — Distribution by Class ─────────────────────────────────────────
    with tab2:
        # Pre-compute inventory value by class (needed for both charts)
        merged = sku_metric.merge(sku_class, on="sku_id", how="left")
        merged = merged.merge(policy_sku[["sku_id", "unit_cost"]], on="sku_id", how="left")
        merged["inv_value"] = merged["avg_inventory"] * merged["unit_cost"]
        class_val = (
            merged.groupby("class")["inv_value"]
            .sum()
            .reset_index()
            .sort_values("inv_value", ascending=False)
            .reset_index(drop=True)
        )

        def lerp_hex(c1_hex, c2_hex, t):
            def h2r(h): return tuple(int(h.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
            def r2h(r, g, b): return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            r1, g1, b1 = h2r(c1_hex)
            r2, g2, b2 = h2r(c2_hex)
            return r2h(r1 + (r2-r1)*t, g1 + (g2-g1)*t, b1 + (b2-b1)*t)

        col_a, col_b = st.columns(2)

        with col_a:
            class_count = sku_class["class"].value_counts().reset_index()
            class_count.columns = ["class", "count"]
            class_count = class_count.sort_values("count", ascending=False).reset_index(drop=True)

            n = len(class_count)
            pie_colors = [
                lerp_hex("#dbeafe", "#1E90FF", 1 - i / max(n - 1, 1) * 0.85)
                for i in range(n)
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top10.assign(fill_rate=lambda d: d["fill_rate"].map("{:.2%}".format))
            .rename(columns={
                "sku_id": "SKU", "avg_inventory": "Avg Inventory",
                "DOI": "DOI (days)", "fill_rate": "Fill Rate"
            }),
            use_container_width=True,
            hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ABC-XYZ ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 ABC-XYZ Analysis":
    st.markdown("# 📊 ABC-XYZ Analysis")

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
                    aspect="auto", title="Number of SKUs per Class")
    fig.update_layout(margin=dict(t=40, b=20), paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown('<div class="section-header">Metrics by Class</div>', unsafe_allow_html=True)
    summary = loader.get_class_summary().copy()
    summary["avg_fill_rate"]  = (summary["avg_fill_rate"] * 100).round(2).astype(str) + "%"
    summary["avg_doi"]        = summary["avg_doi"].round(2)
    summary["avg_lost_sales"] = summary["avg_lost_sales"].round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Problem classes
    st.markdown('<div class="section-header">⚠️ Classes Requiring Optimization</div>', unsafe_allow_html=True)

    problem_classes = ["AX", "AY", "CX", "BZ", "CZ"]
    df_p = df[df["class"].isin(problem_classes)].copy()
    agg  = df_p.groupby("class").agg(
        fill_rate  =("fill_rate",  "mean"),
        lost_sales =("lost_sales", "mean"),
        DOI        =("DOI",        "mean")
    ).reset_index()

    # ── Helper: build sorted bar chart ────────────────────────────────────────
    def sorted_bar(data, y_col, highlight_classes, title, text_fmt, yaxis_fmt=None, color_override=None):
        """Return a bar chart sorted descending by y_col, with highlight_classes in BLUE."""
        sorted_data = data.sort_values(y_col, ascending=False).copy()
        colors = [
            (color_override if color_override else BLUE) if c in highlight_classes else GRAY
            for c in sorted_data["class"]
        ]
        texts = sorted_data[y_col].apply(text_fmt)
        fig = go.Figure(go.Bar(
            x=sorted_data["class"], y=sorted_data[y_col],
            marker_color=colors,
            text=texts, textposition="outside"
        ))
        layout_kwargs = dict(
            title=title,
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, margin=dict(t=40, b=20)
        )
        if yaxis_fmt:
            layout_kwargs["yaxis_tickformat"] = yaxis_fmt
        fig.update_layout(**layout_kwargs)
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        return fig

    # Row 1 — AX, AY
    st.markdown("**AX, AY — High Value, Low Service**")
    st.markdown("These SKUs contribute most to revenue but have fill rates below target service levels.")

    col1, col2 = st.columns(2)
    ax_ay = agg[agg["class"].isin(["AX", "AY"])].copy()
    ax_ay["target"] = ax_ay["class"].map(SERVICE_LEVEL).astype(float) / 100

    with col1:
        ax_ay_sorted = ax_ay.sort_values("fill_rate", ascending=False).reset_index(drop=True)
        fig = go.Figure()
        for i, row in ax_ay_sorted.iterrows():
            fig.add_trace(go.Bar(
                x=[row["class"]], y=[row["fill_rate"]],
                name=row["class"],
                marker_color=BLUE,
                text=f"{row['fill_rate']:.2%}", textposition="outside",
                showlegend=False
            ))
            fig.add_shape(
                type="line",
                x0=i - 0.4, x1=i + 0.4,
                y0=row["target"], y1=row["target"],
                line=dict(color=RED, width=2, dash="dash"),
                xref="x", yref="y"
            )
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                  line=dict(color=RED, dash="dash"), name="Target SL"))
        fig.update_layout(
            title="Fill Rate vs Target Service Level",
            yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = sorted_bar(
            agg, "lost_sales",
            highlight_classes=["AX", "AY"],
            title="Avg Lost Sales by Class",
            text_fmt=lambda v: f"{v:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — CX
    st.markdown("**CX — Understock**")
    st.markdown("CX has the highest lost sales and low DOI — insufficient inventory leads to frequent stockouts.")

    col3, col4 = st.columns(2)
    with col3:
        fig = sorted_bar(
            agg, "lost_sales",
            highlight_classes=["CX"],
            title="Avg Lost Sales — CX has the highest",
            text_fmt=lambda v: f"{v:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = sorted_bar(
            agg, "DOI",
            highlight_classes=["CX"],
            title="Avg DOI — CX is among the lowest",
            text_fmt=lambda v: f"{v:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — BZ, CZ
    st.markdown("**BZ, CZ — Overstock**")
    st.markdown("BZ and CZ have high fill rates but the highest DOI — excess inventory is tied up unnecessarily.")

    col5, col6 = st.columns(2)
    with col5:
        fig = sorted_bar(
            agg, "fill_rate",
            highlight_classes=["BZ", "CZ"],
            title="Fill Rate — BZ, CZ higher than AX, AY",
            text_fmt=lambda v: f"{v:.2%}",
            yaxis_fmt=".0%"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = sorted_bar(
            agg, "DOI",
            highlight_classes=["BZ", "CZ"],
            title="DOI — CZ highest, followed by BZ",
            text_fmt=lambda v: f"{v:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecast Performance":
    st.markdown("# 📈 Forecast Performance")

    forecast_df = loader.forecast
    sku_class   = loader.sku_class

    sku_list     = sorted(forecast_df["sku_id"].unique().tolist())
    selected_sku = st.selectbox("Select SKU", sku_list)

    sku_data  = loader.get_forecast_by_sku(selected_sku)
    cls_label = sku_class[sku_class["sku_id"] == selected_sku]["class"].values
    cls_label = cls_label[0] if len(cls_label) > 0 else "N/A"

    y_true   = sku_data["demand"]
    y_pred   = sku_data["forecast"]
    mean_    = y_true.mean()
    mae      = (y_true - y_pred).abs().mean()
    rmse     = ((y_true - y_pred) ** 2).mean() ** 0.5
    bias     = (y_pred - y_true).mean()
    acc      = (1 - mae / mean_) * 100
    pct_mae  = mae  / mean_ * 100
    pct_rmse = rmse / mean_ * 100
    pct_bias = bias / mean_ * 100

    st.markdown(f"**SKU: `{selected_sku}` · Class: `{cls_label}`**")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, fmt in zip(
        [c1, c2, c3, c4],
        ["Forecast Accuracy", "%MAE", "%RMSE", "%Bias"],
        [acc, pct_mae, pct_rmse, pct_bias],
        ["{:.2f}%", "{:.2f}%", "{:.2f}%", "{:+.2f}%"]
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{fmt.format(value)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="section-header">Actual vs Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["demand"],
        name="Actual", line=dict(color="#0f1117", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=sku_data["date"], y=sku_data["forecast"],
        name="Forecast", line=dict(color=BLUE, width=1.5, dash="dash")
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="", yaxis_title="Demand",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=20), hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POLICY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Inventory Policy":
    st.markdown("# 🎯 Inventory Policy")

    old_df = loader.old_policy
    new_df = loader.new_policy

    available_classes = sorted(old_df["class"].unique().tolist())
    selected_class    = st.selectbox("Select Class", available_classes)

    old_cls = old_df[old_df["class"] == selected_class]
    new_cls = new_df[new_df["class"] == selected_class]

    METRICS = [
        ("fill_rate",     "Fill Rate",          "{:.2%}", 1,    True),
        ("avg_inventory", "Avg Inventory",       "{:.2f}", 1,    True),
        ("holding_cost",  "Holding Cost ($)",    "{:,.0f}", 1,   False),
        ("ordering_cost", "Ordering Cost ($)",   "{:,.0f}", 1,   False),
        ("stockout_cost", "Stockout Cost ($)",   "{:,.0f}", 1,   False),
        ("total_cost",    "Total Cost ($)",      "{:,.0f}", 1,   False)
    ]

    cols = st.columns(len(METRICS))

    for col, (metric, label, fmt, scale, higher_is_better) in zip(cols, METRICS):
        old_val = old_cls[metric].mean() * scale
        new_val = new_cls[metric].mean() * scale
        if metric == "fill_rate":
            delta       = (new_val - old_val) * 100
            improved    = (delta > 0) == higher_is_better
            arrow       = "▲" if delta > 0 else "▼"
            delta_class = "kpi-delta-pos" if improved else "kpi-delta-neg"
            delta_str   = f"{arrow} {abs(delta):.2f} pts"
        else:
            delta       = (new_val - old_val) / abs(old_val) * 100 if old_val != 0 else 0
            improved    = (delta > 0) == higher_is_better
            arrow       = "▲" if delta > 0 else "▼"
            delta_class = "kpi-delta-pos" if improved else "kpi-delta-neg"
            delta_str   = f"{arrow} {abs(delta):.2f}% vs old"
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{fmt.format(new_val)}</div>
                <div class="{delta_class}">{delta_str}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="section-header">Old vs New Policy Comparison</div>', unsafe_allow_html=True)
    cols2 = st.columns(len(METRICS))

    for col, (metric, label, fmt, scale, _) in zip(cols2, METRICS):
        old_val = old_cls[metric].mean() * scale
        new_val = new_cls[metric].mean() * scale

        fig = go.Figure(go.Bar(
            x=["Old", "New"],
            y=[old_val, new_val],
            marker_color=[GRAY, BLUE],
            text=[fmt.format(old_val), fmt.format(new_val)],
            textposition="outside"
        ))
        fig.update_layout(
            title=label,
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", visible=False),
            xaxis=dict(showgrid=False),
            height=280
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">New Policy Parameters</div>', unsafe_allow_html=True)
    display_cols = ["sku_id", "class", "target_service_level", "safety_stock", "rop", "order_quantity"]
    st.dataframe(
        loader.policy_sku[loader.policy_sku["class"] == selected_class][display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )