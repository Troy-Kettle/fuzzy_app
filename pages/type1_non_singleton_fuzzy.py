import streamlit as st
from src.ui.type1_non_singleton_ui_sections import (
    render_upload_config_section,
    render_presets_section,
    render_variable_section,
    render_fuzzy_sets_section,
    render_visualization_section,
    render_non_singleton_visualization_section,
    render_rules_section,
    render_inference_section,
    render_export_section,
)

st.set_page_config(page_title="Non-Singleton Type-1 Fuzzy Inference System", layout="wide")
st.title("Non-Singleton Type-1 Fuzzy Inference System")

st.markdown("""
This system allows you to create and test **non-singleton Type-1 fuzzy logic systems**. 
Non-singleton fuzzification accounts for input uncertainty by modelling inputs as fuzzy sets 
rather than crisp values, making the system more robust to measurement noise and uncertainty.

**Features:**
- Input uncertainty modelling with standard deviation parameters
- Non-singleton fuzzification for enhanced robustness
- Real-time visualisation of input uncertainty effects
- Interactive comparison of singleton vs non-singleton membership values
""")

render_upload_config_section()
render_presets_section()
render_variable_section()
render_fuzzy_sets_section()
render_visualization_section()
render_non_singleton_visualization_section()
render_rules_section()
render_inference_section()
render_export_section() 