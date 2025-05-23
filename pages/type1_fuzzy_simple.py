import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Fuzzy Inference System", layout="wide")
st.title("Fuzzy Inference System (FIS)")

# Initialize session state
if "fis_vars" not in st.session_state:
    st.session_state.fis_vars = []
if "fis_rules" not in st.session_state:
    st.session_state.fis_rules = []
if "edit_rule_idx" not in st.session_state:
    st.session_state.edit_rule_idx = None

# --- Load Tipper Example ---
def load_tipper():
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
    # Rules: OR is represented by two rules (since only AND is supported)
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

# Add buttons for loading example or clearing
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Tipper Example"):
        load_tipper()
with col2:
    if st.button("Clear All"):
        st.session_state.fis_vars = []
        st.session_state.fis_rules = []
        st.session_state.edit_rule_idx = None
        st.rerun()

# --- Section 1: Define Variables (Inputs & Outputs) ---
st.header("1. Define Variables")

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
                "sets": []  # Each set: {name, type, params}
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

# --- Section 2: Define Fuzzy Sets for Each Variable ---
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
            st.write(f"- {fset['name']} ({fset['type']}): {fset['params']}")
            if st.button(f"Remove {fset['name']} from {var['name']}", key=f"delset_{vidx}_{sidx}"):
                var['sets'].pop(sidx)
                st.rerun()
    else:
        st.info(f"No sets defined for {var['name']}.")

# --- Section 3: Visualize Membership Functions ---
st.header("3. Visualize Membership Functions")
for var in st.session_state.fis_vars:
    if not var['sets']:
        continue
    st.subheader(f"{var['role']}: {var['name']}")
    
    # Plot membership functions
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(var['range'][0], var['range'][1], 1000)
    
    for fset in var['sets']:
        y = np.zeros_like(x)
        params = [float(p.strip()) for p in fset['params'].split(",")]
        
        if fset['type'] == "Triangular" and len(params) == 3:
            a, b, c = params
            # Left side
            mask_left = (x >= a) & (x < b)
            if b > a:
                y[mask_left] = (x[mask_left] - a) / (b - a)
            
            # Right side
            mask_right = (x > b) & (x <= c)
            if c > b:
                y[mask_right] = (c - x[mask_right]) / (c - b)
            
            # Peak
            y[x == b] = 1.0
            
        elif fset['type'] == "Trapezoidal" and len(params) == 4:
            a, b, c, d = params
            # Left side
            mask_left = (x >= a) & (x < b)
            if b > a:
                y[mask_left] = (x[mask_left] - a) / (b - a)
            
            # Flat top
            mask_flat = (x >= b) & (x <= c)
            y[mask_flat] = 1.0
            
            # Right side
            mask_right = (x > c) & (x <= d)
            if d > c:
                y[mask_right] = (d - x[mask_right]) / (d - c)
            
        elif fset['type'] == "Gaussian" and len(params) == 2:
            mean, sigma = params
            if sigma > 0:
                y = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        
        ax.plot(x, y, label=fset['name'])
    
    ax.set_xlabel(var['name'])
    ax.set_ylabel("Membership")
    ax.set_title(f"Membership Functions for {var['name']}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

# --- Section 4: Define Fuzzy Rules ---
st.header("4. Define Fuzzy Rules")
if "fis_rules" not in st.session_state:
    st.session_state.fis_rules = []  # Each rule: {if: [(var, set)], then: (var, set)}
if "edit_rule_idx" not in st.session_state:
    st.session_state.edit_rule_idx = None

# Helper: get variable/set lists
input_vars = [v for v in st.session_state.fis_vars if v['role'] == "Input"]
output_vars = [v for v in st.session_state.fis_vars if v['role'] == "Output"]

# Display existing rules
if st.session_state.fis_rules:
    for idx, rule in enumerate(st.session_state.fis_rules):
        if_part = " AND ".join([f"{var} is {set_name}" for var, set_name in rule['if']])
        then_part = f"{rule['then'][0]} is {rule['then'][1]}"
        rule_text = f"Rule {idx+1}: IF {if_part} THEN {then_part}"
        
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.write(rule_text)
        with col2:
            if st.button("Edit", key=f"edit_rule_{idx}"):
                st.session_state.edit_rule_idx = idx
                st.rerun()
        with col3:
            if st.button("Delete", key=f"del_rule_{idx}"):
                st.session_state.fis_rules.pop(idx)
                st.rerun()
else:
    st.info("No rules defined yet.")

# --- Add/Edit Rule Builder ---
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
            for v, s in st.session_state.fis_rules[st.session_state.edit_rule_idx]['if']:
                if v == var['name'] and s in set_names:
                    default_idx = set_names.index(s)
                    break
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
                out_var = next(v for v in output_vars if v['name'] == then_var)
                out_set_names = [s['name'] for s in out_var['sets']]
                if then_set in out_set_names:
                    default_outset = out_set_names.index(then_set)
        
        sel_out_var = st.selectbox("THEN", out_var_names, index=default_outvar)
        out_var = next(v for v in output_vars if v['name'] == sel_out_var)
        out_set_names = [s['name'] for s in out_var['sets']]
        sel_out_set = st.selectbox("is", out_set_names, index=default_outset if out_set_names else 0) if out_set_names else None
        
        can_submit = all(cond for _, cond in rule_conds) and sel_out_set is not None
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

# --- Section 5: Inference ---
st.header("5. Fuzzy Inference & Result")

# Simple membership function calculation
def calculate_membership(val, fset_type, params):
    """Calculate membership value for a given input and fuzzy set"""
    if fset_type == "Triangular":
        a, b, c = params
        if val <= a or val >= c:
            return 0.0
        elif val == b:
            return 1.0
        elif val < b:
            return (val - a) / (b - a) if b > a else 1.0
        else:  # val > b
            return (c - val) / (c - b) if c > b else 1.0
    
    elif fset_type == "Trapezoidal":
        a, b, c, d = params
        if val <= a or val >= d:
            return 0.0
        elif b <= val <= c:
            return 1.0
        elif val < b:
            return (val - a) / (b - a) if b > a else 1.0
        else:  # val > c
            return (d - val) / (d - c) if d > c else 1.0
    
    elif fset_type == "Gaussian":
        mean, sigma = params
        if sigma <= 0:
            return 1.0 if val == mean else 0.0
        return np.exp(-0.5 * ((val - mean) / sigma) ** 2)
    
    return 0.0

# Input section
inputs = {}
debug_mode = st.checkbox("Debug Mode", value=False)

for var in st.session_state.fis_vars:
    if var['role'] == "Input":
        val = st.slider(
            f"Input value for {var['name']}",
            min_value=float(var['range'][0]),
            max_value=float(var['range'][1]),
            value=float(np.mean(var['range'])),
            step=0.1,
            key=f"input_{var['name']}"
        )
        inputs[var['name']] = val
        
        # Show membership values in debug mode
        if debug_mode and var['sets']:
            st.write(f"Membership values for {var['name']} = {val}:")
            for fset in var['sets']:
                params = [float(p.strip()) for p in fset['params'].split(",")]
                mu = calculate_membership(val, fset['type'], params)
                st.write(f"  - {fset['name']}: {mu:.4f}")

# Run inference button
if st.button("Run Inference") and st.session_state.fis_rules:
    try:
        # Check if we have all required inputs
        input_vars = [v for v in st.session_state.fis_vars if v['role'] == "Input"]
        if not all(var['name'] in inputs for var in input_vars):
            st.error("Missing input values for some variables.")
        else:
            # Process each output variable
            output_results = {}
            
            for out_var in [v for v in st.session_state.fis_vars if v['role'] == "Output"]:
                if not out_var['sets']:
                    st.warning(f"Output variable '{out_var['name']}' has no fuzzy sets defined.")
                    continue
                
                # Create a discrete universe of discourse for the output
                x_out = np.linspace(out_var['range'][0], out_var['range'][1], 500)
                aggregated = np.zeros_like(x_out)
                
                # Track if any rules fired
                any_rule_fired = False
                
                # Process each rule
                for rule_idx, rule in enumerate(st.session_state.fis_rules):
                    # Skip rules that don't apply to this output
                    if rule['then'][0] != out_var['name']:
                        continue
                    
                    # Calculate rule firing strength (min of all antecedents)
                    firing_strength = 1.0
                    rule_applies = True
                    
                    for var_name, set_name in rule['if']:
                        # Find the variable
                        var = next((v for v in st.session_state.fis_vars if v['name'] == var_name), None)
                        if not var or var_name not in inputs:
                            rule_applies = False
                            break
                        
                        # Find the fuzzy set
                        fset = next((s for s in var['sets'] if s['name'] == set_name), None)
                        if not fset:
                            rule_applies = False
                            break
                        
                        # Calculate membership
                        params = [float(p.strip()) for p in fset['params'].split(",")]
                        mu = calculate_membership(inputs[var_name], fset['type'], params)
                        
                        # Update firing strength (min operator for AND)
                        firing_strength = min(firing_strength, mu)
                    
                    if not rule_applies or firing_strength <= 0:
                        continue
                    
                    # Rule fired with non-zero strength
                    any_rule_fired = True
                    
                    # Get consequent fuzzy set
                    consequent_set_name = rule['then'][1]
                    consequent_set = next((s for s in out_var['sets'] if s['name'] == consequent_set_name), None)
                    
                    if not consequent_set:
                        continue
                    
                    # Calculate consequent membership function
                    params = [float(p.strip()) for p in consequent_set['params'].split(",")]
                    y_out = np.zeros_like(x_out)
                    
                    for i, x in enumerate(x_out):
                        y_out[i] = calculate_membership(x, consequent_set['type'], params)
                    
                    # Apply firing strength (min operator)
                    y_out = np.minimum(y_out, firing_strength)
                    
                    # Aggregate (max operator)
                    aggregated = np.maximum(aggregated, y_out)
                    
                    if debug_mode:
                        st.write(f"Rule {rule_idx+1} fired with strength: {firing_strength:.4f}")
                
                # Defuzzify using centroid method if any rule fired
                if any_rule_fired and np.sum(aggregated) > 0:
                    # Centroid defuzzification
                    centroid = np.sum(x_out * aggregated) / np.sum(aggregated)
                    output_results[out_var['name']] = centroid
                    
                    # Plot the result
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.fill_between(x_out, aggregated, alpha=0.3, color='blue')
                    ax.plot(x_out, aggregated, label="Aggregated Output", color='blue')
                    ax.axvline(centroid, color='red', linestyle='--', 
                               label=f"Defuzzified: {centroid:.2f}")
                    
                    ax.set_xlabel(out_var['name'])
                    ax.set_ylabel("Membership")
                    ax.set_title(f"Output for {out_var['name']}")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    # No rules fired
                    centroid = float(np.mean(out_var['range']))
                    output_results[out_var['name']] = centroid
                    st.warning(f"No rules fired for {out_var['name']}. Using default value: {centroid:.2f}")
            
            # Show final results
            if output_results:
                st.success("Inference complete!")
                for var_name, value in output_results.items():
                    st.write(f"{var_name}: {value:.2f}")
            else:
                st.warning("No output was produced. Check your rules and variables.")
                
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
elif st.session_state.fis_rules:
    st.info("Press 'Run Inference' to compute output.")
else:
    st.info("Define at least one rule to run inference.")
