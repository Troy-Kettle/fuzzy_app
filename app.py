import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Fuzzy Logic Inference System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("🧠 Fuzzy Logic Inference System")
st.markdown("---")

# Introduction
st.markdown("""
### Welcome to the Interactive Fuzzy Logic System

Build, visualise, and experiment with both **Type-1** and **Interval Type-2 (IT2)** fuzzy logic systems. 
Perfect for education, research, and rapid prototyping of fuzzy inference systems.
""")

# Two-column layout for features
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Features
    
    - **Interactive System Builder**
    - **Real-time Visualisation**
    - **Export to Python Code**
    - **Save/Load Configurations**
    - **Type-1 & Type-2 Support**
    - **Educational Examples**
    """)

with col2:
    st.markdown("""
    ### 🚀 Quick Start
    
    1. **Select a System Type** from the sidebar
    2. **Define Variables** and fuzzy sets
    3. **Create Rules** for inference
    4. **Run Inference** and see results
    5. **Export** your system
    """)

st.markdown("---")

# System types explanation
st.subheader("📚 Understanding Fuzzy Logic Systems")

tab1, tab2, tab3 = st.tabs(["Type-1 Fuzzy", "Type-2 Fuzzy", "Applications"])

with tab1:
    st.markdown("""
    **Type-1 Fuzzy Logic** is the classic approach where membership functions are crisp values between 0 and 1.
    
    **Advantages:**
    - Simple to understand and implement
    - Fast computation
    - Widely used in control systems
    
    **Use Cases:**
    - Temperature control systems
    - Washing machine controllers
    - Simple decision-making systems
    """)

with tab2:
    st.markdown("""
    **Interval Type-2 (IT2) Fuzzy Logic** extends Type-1 by introducing uncertainty in membership functions.
    
    **Advantages:**
    - Handles uncertainty better
    - More robust to noise
    - Provides confidence intervals
    
    **Use Cases:**
    - Noisy sensor data
    - Medical diagnosis systems
    - Financial forecasting
    """)

with tab3:
    st.markdown("""
    **Common Applications:**
    
    - **Control Systems:** Temperature, pressure, speed control
    - **Decision Making:** Medical diagnosis, risk assessment
    - **Pattern Recognition:** Image processing, speech recognition
    - **Optimisation:** Resource allocation, scheduling
    """)

st.markdown("---")

# Navigation section
st.subheader("🎮 Get Started")

st.markdown("""
Choose your fuzzy system type from the sidebar and start building:

- **[Type-1 Fuzzy System](./pages/type1_fuzzy.py)** - Classic fuzzy logic with crisp membership functions
- **[Non-Singleton Type-1 Fuzzy System](./pages/type1_non_singleton_fuzzy.py)** - Type-1 fuzzy logic with input uncertainty modelling
- **[Type-2 Fuzzy System](./pages/type2_fuzzy.py)** - Advanced fuzzy logic with uncertainty handling
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>For more information, see the <a href='https://en.wikipedia.org/wiki/Fuzzy_logic' target='_blank'>Wikipedia article on Fuzzy Logic</a></p>
</div>
""", unsafe_allow_html=True)
