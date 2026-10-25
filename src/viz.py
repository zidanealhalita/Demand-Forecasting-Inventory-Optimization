"""Shared chart helpers: one colour-blind-safe palette (Okabe-Ito), one layout, insight-style titles.

Principles: a single y-axis per chart, thin marks, recessive grid, colour follows the *entity* (category /
policy) and never its rank, every chart gets a title that states the insight plus labelled axes with units.
"""
from __future__ import annotations

import textwrap

import plotly.graph_objects as go

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, PINK, SKY, YELLOW, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)
GREY = "#8A8F98"
CATEGORY_COLORS = {
    "Water": BLUE,
    "Dairy": ORANGE,
    "Dairy-Chilled": SKY,
    "Nutrition": GREEN,
    "Plant-based": PINK,
}
POLICY_COLORS = {"Current policy": VERMILION, "Proposed policy": BLUE}
SEQ_SCALE = [[0, "#F2F7FB"], [0.5, "#6FA8D6"], [1, "#08306B"]]  # single-hue sequential (blue)
DIV_SCALE = [[0, BLUE], [0.5, "#F5F5F5"], [1, VERMILION]]  # diverging: blue (low) - neutral - vermilion (high)
RISK_SCALE = [[0, "#FFF5EB"], [0.5, "#F5A25D"], [1, "#8C2D04"]]  # single-hue sequential (orange) for risk

FOOTER_TEXT = "Synthetic data for portfolio purposes - not real company data."


def style(
    fig: go.Figure,
    title: str,
    xtitle: str | None = None,
    ytitle: str | None = None,
    height: int = 380,
    legend: bool = True,
    wrap: int = 68,
) -> go.Figure:
    """Apply the project theme: insight title, axis labels with units, light grid, legend on top."""
    lines = textwrap.wrap(title, wrap) or [title]
    top = 38 + 21 * len(lines)
    bottom = 100 if legend else 10
    fig.update_layout(
        title=dict(text="<br>".join(lines), x=0.0, xanchor="left", y=1.0, yanchor="bottom", yref="paper", pad=dict(b=10), font=dict(size=15)),
        height=height + 21 * (len(lines) - 1) + (50 if legend else 0),
        margin=dict(l=10, r=10, t=top, b=bottom),
        template="plotly_white",
        font=dict(size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, yref="container", xanchor="left", x=0.0, title_text=""),
        hovermode="x unified" if any(getattr(t, "type", "") == "scatter" for t in fig.data) else "closest",
    )
    fig.update_xaxes(title_text=xtitle, showgrid=False, linecolor="#C9CDD2", ticks="outside", tickcolor="#C9CDD2")
    fig.update_yaxes(title_text=ytitle, gridcolor="#E9ECEF", zeroline=False)
    return fig


def idr_m(x: float) -> str:
    """Format IDR as millions / billions (e.g. 'IDR 1.69 bn')."""
    a = abs(x)
    if a >= 1e11:
        return f"IDR {x / 1e9:,.0f} bn"
    if a >= 1e10:
        return f"IDR {x / 1e9:,.1f} bn"
    if a >= 1e9:
        return f"IDR {x / 1e9:,.2f} bn"
    if a >= 1e6:
        return f"IDR {x / 1e6:,.1f} m"
    return f"IDR {x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    """Format a fraction as a percentage string."""
    return f"{x * 100:.{d}f}%"
