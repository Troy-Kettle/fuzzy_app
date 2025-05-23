import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time  # For generating unique keys

def safe_div(n, d):
    # Handle both scalar and array division safely
    if isinstance(n, np.ndarray) or isinstance(d, np.ndarray):
        # For array division
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(n, d)
            result[~np.isfinite(result)] = 0  # Replace inf/NaN with 0
        return result
    else:
        # For scalar division
        return n / d if d != 0 else 0

st.set_page_config(page_title="Fuzzy Inference System", layout="wide")
st.title("Fuzzy Inference System (FIS)")

# --- Load Tipper Example ---
def load_tipper():
    # Clear any existing variables and rules
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
    # Don't use st.rerun() here as it can cause issues

# Add a clear button to reset everything
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Tipper Example", key="load_tipper"):
        load_tipper()
with col2:
    if st.button("Clear All", key="clear_all"):
        st.session_state.fis_vars = []
        st.session_state.fis_rules = []
        st.session_state.edit_rule_idx = None
        st.rerun()

# --- Section 1: Define Variables (Inputs & Outputs) ---
st.header("1. Define Variables")
if "fis_vars" not in st.session_state:
    st.session_state.fis_vars = []  # List of dicts: {name, role, range, sets}

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
    # Show sets for variable
    if var['sets']:
        for sidx, fset in enumerate(var['sets']):
            st.markdown(f"- **{fset['name']}** ({fset['type']}, params: {fset['params']}) ")
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
    rng = np.linspace(var['range'][0], var['range'][1], 500)
    fig, ax = plt.subplots(figsize=(7, 3))
    for fset in var['sets']:
        params = [float(p.strip()) for p in fset['params'].split(",")]
        y = np.zeros_like(rng)
        if fset['type'] == "Triangular" and len(params) == 3:
            a, b, c = params
            left = safe_div(rng - a, b - a)
            right = safe_div(c - rng, c - b)
            y = np.maximum(np.minimum(left, right), 0)
        elif fset['type'] == "Trapezoidal" and len(params) == 4:
            a, b, c, d = params
            left = safe_div(rng - a, b - a)
            right = safe_div(d - rng, d - c)
            y = np.maximum(np.minimum(np.minimum(left, 1), right), 0)
        elif fset['type'] == "Gaussian" and len(params) == 2:
            mean, sigma = params
            y = np.exp(-0.5*((rng-mean)/sigma)**2)
        ax.plot(rng, y, label=fset['name'])
    ax.set_xlabel(var['name'])
    ax.set_ylabel("Membership")
    ax.set_title(f"Membership Functions for {var['name']}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- Section 4: Define Fuzzy Rules ---
st.header("4. Define Fuzzy Rules")
if "fis_rules" not in st.session_state:
    st.session_state.fis_rules = []  # Each rule: {if: [(var, set)], then: (var, set)}
if "edit_rule_idx" not in st.session_state:
    st.session_state.edit_rule_idx = None

# Helper: get variable/set lists
input_vars = [v for v in st.session_state.fis_vars if v['role']=="Input"]
output_vars = [v for v in st.session_state.fis_vars if v['role']=="Output"]

# --- Rule Table ---
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
    st.dataframe(df, use_container_width=True)
    for ridx, rule in enumerate(st.session_state.fis_rules):
        cols = st.columns([1,1])
        if cols[0].button("Edit", key=f"edit_rule_{ridx}"):
            st.session_state.edit_rule_idx = ridx
        if cols[1].button("Delete", key=f"delrule_{ridx}"):
            st.session_state.fis_rules.pop(ridx)
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
            # Pre-fill with current value
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

# --- Section 5: Inference ---
st.header("5. Fuzzy Inference & Result")
inputs = {}

# Add a debug toggle
debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed calculation steps")

# Simple input section
for var in st.session_state.fis_vars:
    if var['role'] == "Input":
        # Use a slider for input values
        val = st.slider(
            f"Input value for {var['name']}",
            min_value=float(var['range'][0]),
            max_value=float(var['range'][1]),
            value=float(np.mean(var['range'])),
            step=0.1,
            key=f"input_{var['name']}_{time.time()}"
        )
        inputs[var['name']] = val
        
        # Show membership values in debug mode
        if debug_mode and var['sets']:
            memberships = fuzzify(val, var['sets'])
            st.write(f"Membership values for {var['name']} = {val}:")
            for set_name, mu in memberships.items():
                st.write(f"  - {set_name}: {mu:.4f}")

def safe_div(n, d):
    # Handle both scalar and array division safely
    if isinstance(n, np.ndarray) or isinstance(d, np.ndarray):
        # For array division
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(n, d)
            result[~np.isfinite(result)] = 0  # Replace inf/NaN with 0
        return result
    else:
        # For scalar division
        return n / d if d != 0 else 0

def fuzzify(val, sets):
    memberships = {}
    for s in sets:
        try:
            params = [float(p.strip()) for p in s['params'].split(",")]
            if s['type'] == "Triangular" and len(params) == 3:
                a, b, c = params
                # Simple triangular membership function
                if val <= a or val >= c:
                    mu = 0.0
                elif val == b:
                    mu = 1.0
                elif val < b:
                    mu = (val - a) / (b - a) if b > a else 1.0
                else:  # val > b
                    mu = (c - val) / (c - b) if c > b else 1.0
            elif s['type'] == "Trapezoidal" and len(params) == 4:
                a, b, c, d = params
                # Simple trapezoidal membership function
                if val <= a or val >= d:
                    mu = 0.0
                elif b <= val <= c:
                    mu = 1.0
                elif val < b:
                    mu = (val - a) / (b - a) if b > a else 1.0
                else:  # val > c
                    mu = (d - val) / (d - c) if d > c else 1.0
            elif s['type'] == "Gaussian" and len(params) == 2:
                mean, sigma = params
                # Simple Gaussian membership function
                if sigma <= 0:
                    mu = 1.0 if val == mean else 0.0
                else:
                    mu = np.exp(-0.5 * ((val - mean) / sigma) ** 2)
            else:
                mu = 0.0
            # Ensure membership is in [0,1]
            mu = max(0.0, min(1.0, mu))
            memberships[s['name']] = mu
        except Exception as e:
            st.error(f"Error in fuzzify: {str(e)}")
            memberships[s['name']] = 0.0
    return memberships

# Simple run button
if st.button("Run Inference", key=f"run_{time.time()}") and st.session_state.fis_rules:
    try:
        # Validate inputs
        input_vars = [v for v in st.session_state.fis_vars if v['role']=="Input"]
        if not all(var['name'] in inputs for var in input_vars):
            st.error("Missing input values for some variables.")
        else:
            # Mamdani-style: min for AND, max for aggregation, centroid defuzzification
            output_results = {}
            for out_var in [v for v in st.session_state.fis_vars if v['role']=="Output"]:
                if not out_var['sets']:
                    st.warning(f"Output variable '{out_var['name']}' has no fuzzy sets defined.")
                    continue
                    
                agg_y = np.zeros(500)
                rng = np.linspace(out_var['range'][0], out_var['range'][1], 500)
                
                # Track if any rules fired for this output
                rules_fired = False
                
                for rule in st.session_state.fis_rules:
                    # Skip rules that don't apply to this output variable
                    if rule['then'][0] != out_var['name']:
                        continue
                        
                    # Check if all input variables in this rule exist
                    valid_rule = True
                    for vname, sname in rule['if']:
                        if vname not in inputs:
                            valid_rule = False
                            st.warning(f"Rule contains undefined input variable: {vname}")
                            break
                    
                    if not valid_rule:
                        continue
                        
                    # Compute firing strength
                    strength = 1.0
                    rule_conditions = []
                    
                    for vname, sname in rule['if']:
                        try:
                            var = next(v for v in st.session_state.fis_vars if v['name']==vname)
                            # Check if the set exists
                            if not any(s['name'] == sname for s in var['sets']):
                                st.warning(f"Rule references undefined fuzzy set '{sname}' for variable '{vname}'")
                                valid_rule = False
                                break
                                
                            memberships = fuzzify(inputs[vname], var['sets'])
                            if sname not in memberships:
                                st.warning(f"Fuzzy set '{sname}' not found in variable '{vname}'")
                                valid_rule = False
                                break
                                
                            mu = memberships[sname]
                            rule_conditions.append((vname, sname, mu))
                            strength = min(strength, mu)
                        except Exception as e:
                            st.error(f"Error evaluating rule condition: {str(e)}")
                            valid_rule = False
                            break
                    
                    # Show rule firing details in debug mode
                    if debug_mode and valid_rule:
                        rule_desc = " AND ".join([f"{vname} is {sname} ({mu:.4f})" for vname, sname, mu in rule_conditions])
                        st.write(f"Rule: IF {rule_desc} THEN {rule['then'][0]} is {rule['then'][1]}")
                        st.write(f"  - Firing strength: {strength:.4f}")
                    
                    if not valid_rule:
                        continue
                        
                    # Get output MF
                    try:
                        setname = rule['then'][1]
                        # Check if the output set exists
                        if not any(s['name'] == setname for s in out_var['sets']):
                            st.warning(f"Rule references undefined output fuzzy set '{setname}'")
                            continue
                            
                        fset = next(s for s in out_var['sets'] if s['name']==setname)
                        params = [float(p.strip()) for p in fset['params'].split(",")]
                        
                        if fset['type'] == "Triangular" and len(params) == 3:
                            a, b, c = params
                            # Simplified triangular membership function
                            y = np.zeros_like(rng)
                            
                            # Only calculate within range
                            mask = (rng >= a) & (rng <= c)
                            
                            # Left side
                            mask_left = (rng >= a) & (rng < b) & mask
                            if b > a and np.any(mask_left):
                                y[mask_left] = (rng[mask_left] - a) / (b - a)
                            
                            # Right side
                            mask_right = (rng > b) & (rng <= c) & mask
                            if c > b and np.any(mask_right):
                                y[mask_right] = (c - rng[mask_right]) / (c - b)
                            
                            # Peak
                            y[rng == b] = 1.0
                        elif fset['type'] == "Trapezoidal" and len(params) == 4:
                            a, b, c, d = params
                            # Simplified trapezoidal membership function
                            y = np.zeros_like(rng)
                            
                            # Left slope (a to b)
                            mask_left = (rng >= a) & (rng < b)
                            if b > a and np.any(mask_left):
                                y[mask_left] = (rng[mask_left] - a) / (b - a)
                            
                            # Flat top (b to c)
                            mask_flat = (rng >= b) & (rng <= c)
                            y[mask_flat] = 1.0
                            
                            # Right slope (c to d)
                            mask_right = (rng > c) & (rng <= d)
                            if d > c and np.any(mask_right):
                                y[mask_right] = (d - rng[mask_right]) / (d - c)
                        elif fset['type'] == "Gaussian" and len(params) == 2:
                            mean, sigma = params
                            if sigma == 0:  # Handle division by zero
                                y = np.zeros_like(rng)
                                y[rng == mean] = 1.0
                            else:
                                y = np.exp(-0.5*((rng-mean)/sigma)**2)
                        else:
                            y = np.zeros_like(rng)
                            
                        agg_y = np.maximum(agg_y, np.minimum(strength, y))
                        rules_fired = True
                    except Exception as e:
                        st.error(f"Error applying rule consequent: {str(e)}")
                        continue
                # Defuzzify (centroid)
                if rules_fired and np.sum(agg_y) > 0:
                    # Calculate centroid (center of gravity)
                    centroid = np.sum(rng * agg_y) / np.sum(agg_y)
                    output_results[out_var['name']] = centroid
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Plot individual output membership functions for each rule
                    if debug_mode:
                        for i, rule in enumerate(st.session_state.fis_rules):
                            if rule['then'][0] == out_var['name']:
                                # Get the output set
                                setname = rule['then'][1]
                                fset = next((s for s in out_var['sets'] if s['name']==setname), None)
                                if fset:
                                    # Calculate the rule strength
                                    strength = 1.0
                                    for vname, sname in rule['if']:
                                        var = next((v for v in st.session_state.fis_vars if v['name']==vname), None)
                                        if var and vname in inputs:
                                            memberships = fuzzify(inputs[vname], var['sets'])
                                            if sname in memberships:
                                                strength = min(strength, memberships[sname])
                                    
                                    # Only show rules with non-zero strength
                                    if strength > 0:
                                        # Calculate the output membership function
                                        params = [float(p.strip()) for p in fset['params'].split(",")]
                                        y = np.zeros_like(rng)
                                        
                                        if fset['type'] == "Triangular" and len(params) == 3:
                                            a, b, c = params
                                            mask = (rng >= a) & (rng <= c)
                                            left = np.ones_like(rng)
                                            right = np.ones_like(rng)
                                            
                                            mask_left = (rng >= a) & (rng < b) & (b != a)
                                            if np.any(mask_left):
                                                left[mask_left] = safe_div(rng[mask_left] - a, b - a)
                                            
                                            mask_right = (rng > b) & (rng <= c) & (c != b)
                                            if np.any(mask_right):
                                                right[mask_right] = safe_div(c - rng[mask_right], c - b)
                                                
                                            y[mask] = np.minimum(left[mask], right[mask])
                                            
                                        elif fset['type'] == "Trapezoidal" and len(params) == 4:
                                            a, b, c, d = params
                                            mask = (rng >= a) & (rng <= d)
                                            left = np.ones_like(rng)
                                            right = np.ones_like(rng)
                                            
                                            mask_left = (rng >= a) & (rng < b) & (b != a)
                                            if np.any(mask_left):
                                                left[mask_left] = safe_div(rng[mask_left] - a, b - a)
                                            
                                            mask_right = (rng > c) & (rng <= d) & (d != c)
                                            if np.any(mask_right):
                                                right[mask_right] = safe_div(d - rng[mask_right], d - c)
                                            
                                            mask_core = (rng >= b) & (rng <= c)
                                            left[mask_core] = 1.0
                                            right[mask_core] = 1.0
                                            
                                            y[mask] = np.minimum(left[mask], right[mask])
                                            
                                        elif fset['type'] == "Gaussian" and len(params) == 2:
                                            mean, sigma = params
                                            if sigma > 0:
                                                y = np.exp(-0.5*((rng-mean)/sigma)**2)
                                        
                                        # Apply rule strength
                                        y_clipped = np.minimum(y, strength)
                                        
                                        # Plot with low alpha to not obscure the aggregated result
                                        ax.plot(rng, y_clipped, '--', alpha=0.3, 
                                                label=f"Rule {i+1}: {setname} (strength={strength:.2f})")
                    
                    # Plot the aggregated membership function
                    ax.fill_between(rng, agg_y, alpha=0.3, color='blue')
                    ax.plot(rng, agg_y, label="Aggregated Output MF", color='blue', linewidth=2)
                    
                    # Plot the defuzzified value (centroid)
                    ax.axvline(centroid, color="red", linestyle="--", linewidth=2, 
                               label=f"Defuzzified: {centroid:.3f}")
                    
                    # Add labels and title
                    ax.set_xlabel(out_var['name'], fontsize=12)
                    ax.set_ylabel("Membership", fontsize=12)
                    ax.set_title(f"Output for {out_var['name']}", fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=10, loc='best')
                    
                    # Show the plot
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show detailed output information
                    st.success(f"Output {out_var['name']} = {centroid:.4f}")
                    
                    # In debug mode, show more details about the defuzzification
                    if debug_mode:
                        st.write("Defuzzification details:")
                        st.write(f"  - Sum of membership values: {np.sum(agg_y):.4f}")
                        st.write(f"  - Weighted sum: {np.sum(rng * agg_y):.4f}")
                        st.write(f"  - Centroid: {centroid:.4f}")
                else:
                    # No rules fired for this output
                    centroid = float(np.mean(out_var['range']))
                    output_results[out_var['name']] = centroid
                    st.warning(f"No rules fired for output variable '{out_var['name']}'. Using default value: {centroid:.3f}")
            
            if output_results:
                st.success(f"Inference complete!")
                
                # Create a summary table for all outputs
                if len(output_results) > 1:
                    output_df = pd.DataFrame({
                        'Output Variable': list(output_results.keys()),
                        'Defuzzified Value': list(output_results.values())
                    })
                    st.write("Summary of all outputs:")
                    st.dataframe(output_df)
            else:
                st.warning("No output variables processed. Check your rules and variables.")
                
            # Add a section to explain the inference process
            with st.expander("How Fuzzy Inference Works"):
                st.markdown("""
                The fuzzy inference process follows these steps:
                1. **Fuzzification**: Convert crisp input values to fuzzy membership values
                2. **Rule Evaluation**: Apply fuzzy operators (AND/OR) to rule antecedents
                3. **Aggregation**: Combine the outputs of all rules
                4. **Defuzzification**: Convert the fuzzy output to a crisp value (using centroid method)
                
                If your results seem incorrect, check:
                - Are your membership functions defined correctly?
                - Do your rules cover the input space adequately?
                - Are the input values within the expected ranges?
                """)
                
                # Show a visual explanation
                st.image("https://www.researchgate.net/publication/330712496/figure/fig2/AS:720124235583488@1548654845316/Mamdani-fuzzy-inference-system.png", 
                         caption="Mamdani Fuzzy Inference Process")
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
elif st.session_state.fis_rules:
    st.info("Press 'Run Inference' to compute output.")
else:
    st.info("Define at least one rule to run inference.")

