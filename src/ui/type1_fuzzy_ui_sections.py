"""
Type-1 Fuzzy Logic System User Interface Components

This module provides Streamlit-based UI components for constructing and interacting with
Type-1 Fuzzy Inference Systems (FIS). It supports variable definition, fuzzy set creation,
rules management, inference execution, and export functionality.

The interface is designed for clarity and ease of use, suitable for both educational and
professional applications in fuzzy logic.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.core.fuzzy_utils import safe_div
from src.core.type1.fuzzy_fuzzification import fuzzify
from src.core.type1.fuzzy_inference import run_fuzzy_inference
from src.export.fuzzy_export import generate_python_code
import json

def load_tipper():
    """
    Loads a pre-configured Type-1 Fuzzy Inference System for restaurant tipping.

    This example demonstrates a classic fuzzy logic application where food quality
    and service level determine the appropriate tip percentage. The system utilises
    triangular membership functions to model linguistic variables.

    Variables:
        - Food Quality (Input): 0-10 scale
        - Service (Input): 0-10 scale
        - Tip (Output): 0-25% range

    Returns:
        None: Updates session state with FIS configuration
    """
    st.session_state.fis_vars = [
        {
            "name": "Food Quality",
            "role": "Input",
            "range": [0, 10],
            "sets": [
                {"name": "bad", "type": "Triangular", "params": "0, 0, 5"},
                {"name": "average", "type": "Triangular", "params": "0, 5, 10"},
                {"name": "good", "type": "Triangular", "params": "5, 10, 10"}
            ]
        },
        {
            "name": "Service",
            "role": "Input",
            "range": [0, 10],
            "sets": [
                {"name": "poor", "type": "Triangular", "params": "0, 0, 5"},
                {"name": "good", "type": "Triangular", "params": "0, 5, 10"},
                {"name": "excellent", "type": "Triangular", "params": "5, 10, 10"}
            ]
        },
        {
            "name": "Tip",
            "role": "Output",
            "range": [0, 25],
            "sets": [
                {"name": "low", "type": "Triangular", "params": "0, 0, 13"},
                {"name": "medium", "type": "Triangular", "params": "0, 13, 25"},
                {"name": "high", "type": "Triangular", "params": "13, 25, 25"}
            ]
        }
    ]
    st.session_state.fis_rules = [
        {"if": [("Food Quality", "bad"), ("Service", "poor")], "then": ("Tip", "low")},
        {"if": [("Food Quality", "bad"), ("Service", "good")], "then": ("Tip", "low")},
        {"if": [("Food Quality", "bad"), ("Service", "excellent")], "then": ("Tip", "low")},
        {"if": [("Food Quality", "average"), ("Service", "poor")], "then": ("Tip", "low")},
        {"if": [("Food Quality", "good"), ("Service", "poor")], "then": ("Tip", "low")},
        {"if": [("Food Quality", "average"), ("Service", "good")], "then": ("Tip", "medium")},
        {"if": [("Food Quality", "average"), ("Service", "excellent")], "then": ("Tip", "high")},
        {"if": [("Food Quality", "good"), ("Service", "good")], "then": ("Tip", "high")},
        {"if": [("Food Quality", "good"), ("Service", "excellent")], "then": ("Tip", "high")}
    ]
    st.session_state.edit_rule_idx = None
    st.success("Loaded classic Tipper example!")
    st.rerun()

def load_washing_machine():
    """
    Loads a pre-configured Type-1 Fuzzy Inference System for washing machine control.

    This example demonstrates an industrial control application where dirtiness level
    and load size determine the optimal wash cycle duration. The system employs
    triangular membership functions to model uncertainty in sensor readings and
    laundry load conditions.

    Variables:
        - Dirtiness (Input): 0-10 scale representing soil level
        - Load Size (Input): 0-10 scale representing laundry quantity
        - Wash Time (Output): 0-60 minutes cycle duration

    Returns:
        None: Updates session state with FIS configuration
    """
    st.session_state.fis_vars = [
        {"name": "Dirtiness", "role": "Input", "range": [0, 10], "sets": [
            {"name": "low", "type": "Triangular", "params": "0, 0, 5"},
            {"name": "medium", "type": "Triangular", "params": "0, 5, 10"},
            {"name": "high", "type": "Triangular", "params": "5, 10, 10"}
        ]},
        {"name": "Load Size", "role": "Input", "range": [0, 10], "sets": [
            {"name": "small", "type": "Triangular", "params": "0, 0, 5"},
            {"name": "medium", "type": "Triangular", "params": "0, 5, 10"},
            {"name": "large", "type": "Triangular", "params": "5, 10, 10"}
        ]},
        {"name": "Wash Time", "role": "Output", "range": [0, 60], "sets": [
            {"name": "short", "type": "Triangular", "params": "0, 0, 30"},
            {"name": "medium", "type": "Triangular", "params": "0, 30, 60"},
            {"name": "long", "type": "Triangular", "params": "30, 60, 60"}
        ]}
    ]
    st.session_state.fis_rules = [
        {"if": [("Dirtiness", "low"), ("Load Size", "small")], "then": ("Wash Time", "short")},
        {"if": [("Dirtiness", "low"), ("Load Size", "medium")], "then": ("Wash Time", "short")},
        {"if": [("Dirtiness", "low"), ("Load Size", "large")], "then": ("Wash Time", "medium")},
        {"if": [("Dirtiness", "medium"), ("Load Size", "small")], "then": ("Wash Time", "medium")},
        {"if": [("Dirtiness", "medium"), ("Load Size", "medium")], "then": ("Wash Time", "medium")},
        {"if": [("Dirtiness", "medium"), ("Load Size", "large")], "then": ("Wash Time", "long")},
        {"if": [("Dirtiness", "high"), ("Load Size", "small")], "then": ("Wash Time", "long")},
        {"if": [("Dirtiness", "high"), ("Load Size", "medium")], "then": ("Wash Time", "long")},
        {"if": [("Dirtiness", "high"), ("Load Size", "large")], "then": ("Wash Time", "long")}
    ]
    st.session_state.edit_rule_idx = None
    st.success("Loaded Washing Machine example!")
    st.rerun()

def load_room_heater():
    """
    Loads a pre-configured Type-1 Fuzzy Inference System for room heating control.

    This example demonstrates a climate control application where ambient temperature
    determines heater power output. The system utilises triangular membership functions
    to model environmental and sensor uncertainty.

    Variables:
        - Temperature (Input): 0-40°C ambient temperature
        - Heater Power (Output): 0-100% power output

    Returns:
        None: Updates session state with FIS configuration
    """
    st.session_state.fis_vars = [
        {"name": "Temperature", "role": "Input", "range": [0, 40], "sets": [
            {"name": "cold", "type": "Triangular", "params": "0, 0, 20"},
            {"name": "comfortable", "type": "Triangular", "params": "10, 20, 30"},
            {"name": "hot", "type": "Triangular", "params": "20, 40, 40"}
        ]},
        {"name": "Heater Power", "role": "Output", "range": [0, 100], "sets": [
            {"name": "low", "type": "Triangular", "params": "0, 0, 50"},
            {"name": "medium", "type": "Triangular", "params": "0, 50, 100"},
            {"name": "high", "type": "Triangular", "params": "50, 100, 100"}
        ]}
    ]
    st.session_state.fis_rules = [
        {"if": [("Temperature", "cold")], "then": ("Heater Power", "high")},
        {"if": [("Temperature", "comfortable")], "then": ("Heater Power", "medium")},
        {"if": [("Temperature", "hot")], "then": ("Heater Power", "low")}
    ]
    st.session_state.edit_rule_idx = None
    st.success("Loaded Room Heater example!")
    st.rerun()

def render_presets_section():
    """
    Renders the preset examples section for Type-1 Fuzzy Inference Systems.

    Provides users with three pre-configured FIS examples to demonstrate
    different application domains: service industry (tipping), industrial control
    (washing machine), and climate control (room heating).

    Returns:
        None: Updates session state with selected example configuration
    """
    st.subheader("Preset Examples")
    col1, col2, col3 = st.columns(3)
    if col1.button("Load Tipper Example"):
        load_tipper()
    if col2.button("Load Washing Machine Example"):
        load_washing_machine()
    if col3.button("Load Room Heater Example"):
        load_room_heater()

def render_variable_section():
    """
    Renders the variable definition section for Type-1 Fuzzy Inference Systems.

    Allows users to define input and output variables with their respective ranges.
    Variables serve as the foundation for the fuzzy system, defining the domain
    of discourse for each linguistic variable in the FIS.

    Features:
        - Variable name specification
        - Role assignment (Input/Output)
        - Range definition with validation
        - Variable management (add/remove)

    Returns:
        None: Updates session state with variable definitions
    """
    st.header("1. Define Variables")
    if "fis_vars" not in st.session_state:
        st.session_state.fis_vars = []
    with st.form("add_var_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            var_name = st.text_input("Variable Name")
        with col2:
            var_role = st.selectbox("Role", ["Input", "Output"])
        with col3:
            var_range = st.text_input("Range (min, max)", "0, 1")
        submitted = st.form_submit_button("Add Variable")
        if submitted and var_name and var_range:
            try:
                rng = [float(x.strip()) for x in var_range.split(",")]
                assert len(rng) == 2 and rng[0] < rng[1]
            except:
                st.warning("Invalid range. Use format: min, max")
                rng = None
            if rng:
                st.session_state.fis_vars.append({
                    "name": var_name,
                    "role": var_role,
                    "range": rng,
                    "sets": []
                })
                st.success(f"Added variable: {var_name}")
    if st.session_state.fis_vars:
        for idx, var in enumerate(st.session_state.fis_vars):
            st.markdown(f"**{var['role']}: {var['name']}**  Range: {var['range']}")
            if st.button(f"Remove {var['name']}", key=f"delvar_{idx}"):
                st.session_state.fis_vars.pop(idx)
                st.rerun()
    else:
        st.info("No variables defined yet.")

def render_fuzzy_sets_section():
    """
    Renders the fuzzy set definition section for Type-1 Fuzzy Inference Systems.

    Provides tools for creating and managing fuzzy sets for each variable. Supports
    triangular, trapezoidal, and Gaussian membership functions with parameter validation.

    Features:
        - Interactive fuzzy set editor
        - Parameter validation and user feedback
        - Set management (add/remove)

    Returns:
        None: Updates session state with fuzzy set definitions
    """
    st.header("2. Define Fuzzy Sets for Variables")
    for vidx, var in enumerate(st.session_state.fis_vars):
        st.subheader(f"{var['role']}: {var['name']}")
        with st.form(f"add_set_form_{vidx}"):
            col1, col2 = st.columns(2)
            with col1:
                set_name = st.text_input(f"Set Name for {var['name']}", key=f"setname_{vidx}")
                set_type = st.selectbox(f"Type", ["Triangular", "Trapezoidal", "Gaussian"], key=f"settype_{vidx}")
            with col2:
                if set_type == "Triangular":
                    params = st.text_input(f"Params (a, b, c)", "0, 0.5, 1", key=f"params_{vidx}")
                elif set_type == "Trapezoidal":
                    params = st.text_input(f"Params (a, b, c, d)", "0, 0.3, 0.7, 1", key=f"params_{vidx}")
                else:
                    params = st.text_input(f"Params (mean, sigma)", "0.5, 0.1", key=f"params_{vidx}")
            addset = st.form_submit_button(f"Add Fuzzy Set to {var['name']}")
            if addset and set_name and params:
                var['sets'].append({
                    "name": set_name,
                    "type": set_type,
                    "params": params
                })
                st.success(f"Added set {set_name} to {var['name']}")
        if var['sets']:
            for sidx, fset in enumerate(var['sets']):
                st.markdown(f"- **{fset['name']}** ({fset['type']}, params: {fset['params']}) ")
                if st.button(f"Remove {fset['name']} from {var['name']}", key=f"delset_{vidx}_{sidx}"):
                    var['sets'].pop(sidx)
                    st.rerun()
        else:
            st.info(f"No sets defined for {var['name']}.")

def render_visualization_section():
    """
    Renders the visualisation section for Type-1 membership functions.

    Displays visualisations of all defined fuzzy sets for each variable, including
    their membership functions. Provides clear visual feedback for system design.

    Features:
        - Membership function visualisation
        - Multi-set comparison within variables
        - Professional plotting with legends

    Returns:
        None: Displays visualisations in Streamlit interface
    """
    st.header("3. Visualise Membership Functions")
    for var in st.session_state.fis_vars:
        if not var['sets']:
            continue
        with st.expander(f"{var['role']}: {var['name']} (show/hide plot)", expanded=False):
            rng = np.linspace(var['range'][0], var['range'][1], 500)
            fig, ax = plt.subplots(figsize=(3, 2.2), dpi=60)
            for idx, fset in enumerate(var['sets']):
                params = [float(p.strip()) for p in fset['params'].split(",")]
                y = np.zeros_like(rng)
                if fset['type'] == "Triangular" and len(params) == 3:
                    a, b, c = params
                    left = np.maximum((rng - a) / (b - a) if b > a else 1, 0)
                    right = np.maximum((c - rng) / (c - b) if c > b else 1, 0)
                    y = np.maximum(np.minimum(left, right), 0)
                elif fset['type'] == "Trapezoidal" and len(params) == 4:
                    a, b, c, d = params
                    left = np.maximum((rng - a) / (b - a) if b > a else 1, 0)
                    right = np.maximum((d - rng) / (d - c) if d > c else 1, 0)
                    y = np.maximum(np.minimum(np.minimum(left, 1), right), 0)
                elif fset['type'] == "Gaussian" and len(params) == 2:
                    mean, sigma = params
                    y = np.exp(-0.5*((rng-mean)/sigma)**2)
                ax.plot(rng, y, linewidth=2, label=fset['name'])
            ax.set_xlabel(var['name'])
            ax.set_ylabel("Membership")
            ax.set_title(f"Membership Functions for {var['name']}")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

def render_rules_section():
    """
    Renders the rule definition and management section for Type-1 Fuzzy Inference Systems.

    Allows users to view, add, edit, and remove fuzzy rules. Rules are displayed in a
    tabular format for clarity. The editor supports both creation and modification of rules.

    Features:
        - Rule table with all current rules
        - Rule editor for adding and editing
        - Rule removal functionality

    Returns:
        None: Updates session state with rule definitions
    """
    st.header("4. Define Fuzzy Rules")
    with st.expander("Show/Hide Rule Table and Editor", expanded=False):
        if "fis_rules" not in st.session_state:
            st.session_state.fis_rules = []
        if "edit_rule_idx" not in st.session_state:
            st.session_state.edit_rule_idx = None
        input_vars = [v for v in st.session_state.fis_vars if v['role']=="Input"]
        output_vars = [v for v in st.session_state.fis_vars if v['role']=="Output"]
        if st.session_state.fis_rules:
            rule_rows = []
            for ridx, rule in enumerate(st.session_state.fis_rules):
                row = {v['name']: '' for v in input_vars}
                for v, s in rule['if']:
                    row[v] = s
                then_var, then_set = rule['then']
                row['Output Variable'] = then_var
                row['Output Set'] = then_set
                rule_rows.append(row)
            df = pd.DataFrame(rule_rows)
            st.dataframe(df, use_container_width=True, height=200)
            for ridx, rule in enumerate(st.session_state.fis_rules):
                cols = st.columns([1,1])
                if cols[0].button("Edit", key=f"edit_rule_{ridx}"):
                    st.session_state.edit_rule_idx = ridx
                if cols[1].button("Delete", key=f"delrule_{ridx}"):
                    st.session_state.fis_rules.pop(ridx)
                    st.rerun()
        else:
            st.info("No rules defined yet.")
        add_mode = st.session_state.edit_rule_idx is None
        if add_mode:
            st.subheader("Add New Rule")
        else:
            st.subheader(f"Edit Rule #{st.session_state.edit_rule_idx+1}")
        with st.form("rule_form"):
            rule_conds = []
            for var in input_vars:
                set_names = [s['name'] for s in var['sets']]
                default_idx = 0
                if not add_mode and st.session_state.fis_rules[st.session_state.edit_rule_idx]:
                    for vname, sname in st.session_state.fis_rules[st.session_state.edit_rule_idx]['if']:
                        if vname == var['name'] and sname in set_names:
                            default_idx = set_names.index(sname)
                cond = st.selectbox(f"IF {var['name']} is", set_names, index=default_idx if set_names else 0, key=f"ruleformif_{var['name']}") if set_names else None
                rule_conds.append((var['name'], cond))
            if output_vars:
                out_var_names = [v['name'] for v in output_vars]
                default_outvar = 0
                default_outset = 0
                if not add_mode and st.session_state.fis_rules[st.session_state.edit_rule_idx]:
                    then_var, then_set = st.session_state.fis_rules[st.session_state.edit_rule_idx]['then']
                    if then_var in out_var_names:
                        default_outvar = out_var_names.index(then_var)
                        out_sets = [s['name'] for s in output_vars[default_outvar]['sets']]
                        if then_set in out_sets:
                            default_outset = out_sets.index(then_set)
                sel_out_var = st.selectbox("THEN output variable", out_var_names, index=default_outvar, key="ruleformthenvar")
                out_sets = [s['name'] for s in output_vars[out_var_names.index(sel_out_var)]['sets']]
                sel_out_set = st.selectbox("THEN output set", out_sets, index=default_outset if out_sets else 0, key="ruleformthenset") if out_sets else None
                can_submit = all(sel for _, sel in rule_conds) and sel_out_set
                submit_label = "Add Rule" if add_mode else "Update Rule"
                submitted = st.form_submit_button(submit_label, disabled=not can_submit)
                cancel = st.form_submit_button("Cancel Edit") if not add_mode else False
                if submitted and can_submit:
                    rule_obj = {"if": rule_conds, "then": (sel_out_var, sel_out_set)}
                    if add_mode:
                        st.session_state.fis_rules.append(rule_obj)
                        st.success("Rule added.")
                    else:
                        st.session_state.fis_rules[st.session_state.edit_rule_idx] = rule_obj
                        st.session_state.edit_rule_idx = None
                        st.success("Rule updated.")
                    st.rerun()
                elif cancel:
                    st.session_state.edit_rule_idx = None
                    st.rerun()
            else:
                st.form_submit_button("Add Rule", disabled=True)
                st.warning("Define at least one output variable and set before adding rules.")

def render_inference_section():
    """
    Renders the inference and result section for Type-1 Fuzzy Inference Systems.

    Allows users to input crisp values for all input variables and computes the
    fuzzy inference result using the defined rules and sets. Displays the output
    value(s) and provides visual feedback for the output membership functions.

    Features:
        - Input fields for all input variables
        - Inference execution and result display
        - Output membership function visualisation

    Returns:
        None: Displays inference results in Streamlit interface
    """
    st.header("5. Fuzzy Inference & Result")
    inputs = {}
    for var in st.session_state.fis_vars:
        if var['role'] == "Input":
            val = st.number_input(
                f"Input value for {var['name']}",
                min_value=float(var['range'][0]),
                max_value=float(var['range'][1]),
                value=float(np.mean(var['range'])),
                step=0.1,
                key=f"input_{var['name']}"
            )
            inputs[var['name']] = val
    if st.button("Run Inference", key="run_inference") and st.session_state.fis_rules:
        try:
            output_results, rule_trace = run_fuzzy_inference(st.session_state.fis_vars, st.session_state.fis_rules, inputs)
            for out_var in [v for v in st.session_state.fis_vars if v['role']=="Output"]:
                if out_var['name'] in output_results:
                    y = output_results[out_var['name']]
                    st.success(f"Output {out_var['name']} = {y:.4f}")
                    # Output MF plot
                    rng = np.linspace(out_var['range'][0], out_var['range'][1], 500)
                    agg = np.zeros(500)
                    for rule in st.session_state.fis_rules:
                        if rule['then'][0] != out_var['name']:
                            continue
                        strength = 1.0
                        for vname, sname in rule['if']:
                            var = next(v for v in st.session_state.fis_vars if v['name']==vname)
                            memberships = fuzzify(inputs[vname], var['sets'])
                            mu = memberships.get(sname, 0.0)
                            strength = min(strength, mu)
                        setname = rule['then'][1]
                        fset = next(s for s in out_var['sets'] if s['name']==setname)
                        params = [float(p.strip()) for p in fset['params'].split(",")]
                        yset = np.zeros_like(rng)
                        if fset['type'] == "Triangular" and len(params) == 3:
                            a, b, c = params
                            left = np.maximum((rng - a) / (b - a) if b > a else 1, 0)
                            right = np.maximum((c - rng) / (c - b) if c > b else 1, 0)
                            yset = np.maximum(np.minimum(left, right), 0)
                        elif fset['type'] == "Trapezoidal" and len(params) == 4:
                            a, b, c, d = params
                            left = np.maximum((rng - a) / (b - a) if b > a else 1, 0)
                            right = np.maximum((d - rng) / (d - c) if d > c else 1, 0)
                            yset = np.maximum(np.minimum(np.minimum(left, 1), right), 0)
                        elif fset['type'] == "Gaussian" and len(params) == 2:
                            mean, sigma = params
                            yset = np.exp(-0.5*((rng-mean)/sigma)**2)
                        agg = np.maximum(agg, np.minimum(strength, yset))
                    fig, ax = plt.subplots(figsize=(3, 1.2), dpi=60)
                    ax.plot(rng, agg, color='blue', linewidth=2, label="Aggregated Output MF")
                    ax.axvline(y, color="green", linestyle="--", linewidth=2, label=f"Defuzzified Output: {y:.3f}")
                    ax.set_xlabel(out_var['name'], fontsize=12)
                    ax.set_ylabel("Membership", fontsize=12)
                    ax.set_title(f"Output for {out_var['name']}", fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=10, loc='best')
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
            if rule_trace:
                st.subheader("Rule Firing Trace")
                st.write("The following rules fired and contributed to the output:")
                st.dataframe(pd.DataFrame(rule_trace))
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
    elif st.session_state.fis_rules:
        st.info("Press 'Run Inference' to compute output.")
    else:
        st.info("Define at least one rule to run inference.")

def render_export_section():
    """
    Renders the export section for Type-1 Fuzzy Inference Systems.

    Allows users to export the current FIS configuration as Python code or JSON.
    Facilitates sharing, reproducibility, and further development outside the UI.

    Features:
        - Python code export
        - JSON configuration export

    Returns:
        None: Provides download buttons in Streamlit interface
    """
    if st.button("Export to Python Code", key="export_py_code"):
        code = generate_python_code(st.session_state.get("fis_vars", []), st.session_state.get("fis_rules", []))
        st.download_button(
            label="Download Python Code",
            data=code,
            file_name="fuzzy_system_export.py",
            mime="text/x-python"
        )
    if st.button("Save FIS Configuration to JSON", key="save_fis_config"):
        fis_config = {
            "variables": st.session_state.get("fis_vars", []),
            "rules": st.session_state.get("fis_rules", [])
        }
        fis_json = json.dumps(fis_config, indent=2)
        st.download_button(
            label="Download FIS Configuration as JSON",
            data=fis_json,
            file_name="fuzzy_system_config.json",
            mime="application/json"
        )

def render_upload_config_section():
    """
    Renders the upload section for FIS configuration JSON files.

    Allows users to upload a previously saved FIS configuration in JSON format.
    The configuration is validated and loaded into the session state for further editing
    or inference.

    Returns:
        None: Updates session state with uploaded configuration
    """
    uploaded_file = st.file_uploader("Upload FIS Configuration JSON", type=["json"], key="upload_fis_config")
    if uploaded_file is not None and "fis_config_loaded" not in st.session_state:
        try:
            config = json.load(uploaded_file)
            if "variables" in config and "rules" in config:
                st.session_state.fis_vars = config["variables"]
                st.session_state.fis_rules = config["rules"]
                st.session_state.edit_rule_idx = None
                st.session_state.fis_config_loaded = True
                st.success("FIS configuration loaded from JSON!")
                st.rerun()
            else:
                st.error("Invalid configuration file: missing 'variables' or 'rules' keys.")
        except Exception as e:
            st.error(f"Failed to load configuration: {str(e)}")
    elif "fis_config_loaded" in st.session_state:
        del st.session_state["fis_config_loaded"] 