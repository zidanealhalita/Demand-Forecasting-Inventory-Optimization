"""Entry point: page routing + shared sidebar filters.

Run with ``streamlit run app.py``. Every page lives in ``pages/`` and imports its logic from ``src/``.
"""
from __future__ import annotations

import streamlit as st

from src import data, ui

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon=":material/inventory_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Analyse": [
        st.Page("pages/1_Executive_Overview.py", title="1 - Executive Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/2_Demand_Patterns.py", title="2 - Demand Patterns", icon=":material/insights:"),
        st.Page("pages/3_Forecasting.py", title="3 - Forecasting", icon=":material/show_chart:"),
        st.Page("pages/4_Inventory_Optimizer.py", title="4 - Inventory Policy Optimizer", icon=":material/tune:"),
        st.Page("pages/5_Supply_Risk.py", title="5 - Supply Risk & Lead Time", icon=":material/local_shipping:"),
    ],
    "Decide": [
        st.Page("pages/6_Recommendations.py", title="6 - Recommendations", icon=":material/task_alt:"),
    ],
    "Appendix": [
        st.Page("pages/7_Validation_Appendix.py", title="Appendix - Validation", icon=":material/fact_check:"),
    ],
}

panel = data.load_panel()
ui.render_sidebar(panel)
st.navigation(PAGES).run()
