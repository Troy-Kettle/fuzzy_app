import streamlit as st
from src.ui.type2_fuzzy_ui_sections import (
    render_upload_config_section,
    render_presets_section,
    render_variable_section,
    render_fuzzy_sets_section,
    render_visualization_section,
    render_rules_section,
    render_inference_section,
    render_export_section,
)

st.set_page_config(page_title="Type 2 Fuzzy Inference System", layout="wide")
st.title("Type 2 Fuzzy Inference System (IT2-FIS)")

render_upload_config_section()
render_presets_section()
render_variable_section()
render_fuzzy_sets_section()
render_visualization_section()
render_rules_section()
render_inference_section()
render_export_section()

st.info("Type 2 fuzzy logic UI is coming soon! This page will allow you to define and run interval type-2 fuzzy inference systems, similar to the Type 1 page.") 