import streamlit as st

st.set_page_config(page_title="Fuzzy Inference System Home", layout="wide")

st.markdown("""
# Welcome to the Fuzzy Inference System App!

This app lets you build, visualize, and experiment with fuzzy logic systems interactively.

---

## What is a Fuzzy Inference System?
A fuzzy inference system (FIS) is a framework for reasoning with uncertain or imprecise information, using fuzzy logic. FISs are widely used in control systems, decision-making, and artificial intelligence.

- **Define input and output variables and their fuzzy sets**
- **Create and edit fuzzy rules**
- **Visualize membership functions and inference results**
- **Save and load your fuzzy system configurations**

---

## How to Use This App
1. Go to the sidebar and select a page (e.g., **Type 1 Fuzzy**) to start building your system.
2. Use the provided tools to define variables, sets, and rules.
3. Run inference and visualize the results.
4. Save or load configurations as needed.

---

## Get Started
- [Type 1 Fuzzy System](./pages/type1_fuzzy.py)

For more information, see the [Wikipedia article on Fuzzy Logic](https://en.wikipedia.org/wiki/Fuzzy_logic).

---

*Created with ❤️ using Streamlit.*
""")
