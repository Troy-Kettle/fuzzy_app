import streamlit as st
from fuzzy_core.type1_fuzzy_ui_sections import (
    render_upload_config_section,
    render_presets_section,
    render_variable_section,
    render_fuzzy_sets_section,
    render_visualization_section,
    render_rules_section,
    render_inference_section,
    render_export_section,
)

st.set_page_config(page_title="Fuzzy Inference System", layout="wide")
st.title("Fuzzy Inference System (FIS)")

render_upload_config_section()
render_presets_section()
render_variable_section()
render_fuzzy_sets_section()
render_visualization_section()
render_rules_section()
render_inference_section()
render_export_section()

