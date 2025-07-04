import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fuzzy_core.fuzzy_fuzzification_it2 import fuzzify_it2
from fuzzy_core.fuzzy_inference_it2 import run_fuzzy_inference_it2
from fuzzy_core.fuzzy_export_it2 import generate_python_code_it2
import json

def load_tipper_it2():
    st.session_state.fis_vars = [
        {
            "name": "Food Quality",
            "role": "Input",
            "range": [0, 10],
            "sets": [
                {"name": "bad", "type": "Triangular", "lower_params": "0, 0, 4", "upper_params": "0, 0, 6"},
                {"name": "average", "type": "Triangular", "lower_params": "2, 5, 8", "upper_params": "0, 5, 10"},
                {"name": "good", "type": "Triangular", "lower_params": "6, 10, 10", "upper_params": "4, 10, 10"}
            ]
        },
        {
            "name": "Service",
            "role": "Input",
            "range": [0, 10],
            "sets": [
                {"name": "poor", "type": "Triangular", "lower_params": "0, 0, 4", "upper_params": "0, 0, 6"},
                {"name": "good", "type": "Triangular", "lower_params": "2, 5, 8", "upper_params": "0, 5, 10"},
                {"name": "excellent", "type": "Triangular", "lower_params": "6, 10, 10", "upper_params": "4, 10, 10"}
            ]
        },
        {
            "name": "Tip",
            "role": "Output",
            "range": [0, 25],
            "sets": [
                {"name": "low", "type": "Triangular", "lower_params": "0, 0, 10", "upper_params": "0, 0, 16"},
                {"name": "medium", "type": "Triangular", "lower_params": "8, 13, 20", "upper_params": "0, 13, 25"},
                {"name": "high", "type": "Triangular", "lower_params": "16, 25, 25", "upper_params": "10, 25, 25"}
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
    st.success("Loaded IT2 Tipper example!")
    st.rerun()

def render_presets_section():
    st.subheader("Preset IT2 Examples")
    if st.button("Load IT2 Tipper Example"):
        load_tipper_it2()
    # Add more IT2 examples as needed

def render_variable_section():
    st.header("1. Define Variables (IT2)")
    if "fis_vars" not in st.session_state:
        st.session_state.fis_vars = []
    with st.form("add_var_form_it2"):
        col1, col2, col3 = st.columns(3)
        with col1:
            var_name = st.text_input("Variable Name (IT2)")
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
            if st.button(f"Remove {var['name']}", key=f"delvar_it2_{idx}"):
                st.session_state.fis_vars.pop(idx)
                st.rerun()
    else:
        st.info("No variables defined yet.")

def render_fuzzy_sets_section():
    st.header("2. Define Interval Type-2 Fuzzy Sets")
    for vidx, var in enumerate(st.session_state.fis_vars):
        st.subheader(f"{var['role']}: {var['name']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            set_name = st.text_input(f"Set Name for {var['name']}", key=f"setname_it2_{vidx}")
            set_type = st.selectbox(f"Type", ["Triangular", "Trapezoidal", "Gaussian"], key=f"settype_it2_{vidx}")
        with col2:
            rng = [float(x) for x in var['range']]
            fou_width = st.number_input(f"FOU width (absolute, e.g. 0.1)", min_value=0.0, max_value=(rng[1]-rng[0]), value=st.session_state.get(f"fou_width_it2_{vidx}", 0.1), step=0.01, key=f"fou_width_it2_{vidx}")
            warning = None
            if set_type == "Triangular":
                aL = st.slider(f"a", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"la_it2_{vidx}", float(rng[0])), step=0.01, key=f"la_it2_{vidx}")
                bL = st.slider(f"b", min_value=aL, max_value=float(rng[1]), value=st.session_state.get(f"lb_it2_{vidx}", float((rng[0]+rng[1])/2)), step=0.01, key=f"lb_it2_{vidx}")
                cL = st.slider(f"c", min_value=bL, max_value=float(rng[1]), value=st.session_state.get(f"lc_it2_{vidx}", float(rng[1])), step=0.01, key=f"lc_it2_{vidx}")
                aU = max(rng[0], aL - fou_width)
                bU = bL
                cU = min(rng[1], cL + fou_width)
                lower_params = f"{aL}, {bL}, {cL}"
                upper_params = f"{aU}, {bU}, {cU}"
                if not (aU <= aL <= bL <= cL <= cU):
                    warning = "Upper MF must contain lower MF: ensure aU <= aL <= bL <= cL <= cU."
            elif set_type == "Trapezoidal":
                aL = st.slider(f"a", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"la_it2_{vidx}", float(rng[0])), step=0.01, key=f"la_it2_{vidx}")
                bL = st.slider(f"b", min_value=aL, max_value=float(rng[1]), value=st.session_state.get(f"lb_it2_{vidx}", float(rng[0]+(rng[1]-rng[0])/3)), step=0.01, key=f"lb_it2_{vidx}")
                cL = st.slider(f"c", min_value=bL, max_value=float(rng[1]), value=st.session_state.get(f"lc_it2_{vidx}", float(rng[0]+2*(rng[1]-rng[0])/3)), step=0.01, key=f"lc_it2_{vidx}")
                dL = st.slider(f"d", min_value=cL, max_value=float(rng[1]), value=st.session_state.get(f"ld_it2_{vidx}", float(rng[1])), step=0.01, key=f"ld_it2_{vidx}")
                aU = max(rng[0], aL - fou_width)
                bU = bL
                cU = cL
                dU = min(rng[1], dL + fou_width)
                lower_params = f"{aL}, {bL}, {cL}, {dL}"
                upper_params = f"{aU}, {bU}, {cU}, {dU}"
                if not (aU <= aL <= bL <= cL <= dL <= dU):
                    warning = "Upper MF must contain lower MF: ensure aU <= aL <= bL <= cL <= dL <= dU."
            else:
                meanL = st.slider(f"mean", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"lmean_it2_{vidx}", float((rng[0]+rng[1])/2)), step=0.01, key=f"lmean_it2_{vidx}")
                sigmaL = st.slider(f"sigma", min_value=0.01, max_value=float(rng[1]-rng[0]), value=st.session_state.get(f"lsigma_it2_{vidx}", 0.1), step=0.01, key=f"lsigma_it2_{vidx}")
                meanU = meanL
                sigmaU = sigmaL + fou_width
                lower_params = f"{meanL}, {sigmaL}"
                upper_params = f"{meanU}, {sigmaU}"
                if not (sigmaU >= sigmaL):
                    warning = "Upper sigma must be >= lower sigma."
        with col3:
            plot_rng = np.linspace(float(rng[0]), float(rng[1]), 500)
            fig, ax = plt.subplots(figsize=(3, 1.2), dpi=60)
            if set_type == "Triangular":
                leftL = np.maximum((plot_rng - aL) / (bL - aL) if bL > aL else 1, 0)
                rightL = np.maximum((cL - plot_rng) / (cL - bL) if cL > bL else 1, 0)
                yL = np.maximum(np.minimum(leftL, rightL), 0)
                leftU = np.maximum((plot_rng - aU) / (bU - aU) if bU > aU else 1, 0)
                rightU = np.maximum((cU - plot_rng) / (cU - bU) if cU > bU else 1, 0)
                yU = np.maximum(np.minimum(leftU, rightU), 0)
            elif set_type == "Trapezoidal":
                leftL = np.maximum((plot_rng - aL) / (bL - aL) if bL > aL else 1, 0)
                rightL = np.maximum((dL - plot_rng) / (dL - cL) if dL > cL else 1, 0)
                yL = np.maximum(np.minimum(np.minimum(leftL, 1), rightL), 0)
                leftU = np.maximum((plot_rng - aU) / (bU - aU) if bU > aU else 1, 0)
                rightU = np.maximum((dU - plot_rng) / (dU - cU) if dU > cU else 1, 0)
                yU = np.maximum(np.minimum(np.minimum(leftU, 1), rightU), 0)
            else:
                yL = np.exp(-0.5*((plot_rng-meanL)/sigmaL)**2)
                yU = np.exp(-0.5*((plot_rng-meanU)/sigmaU)**2)
            # Lower and upper MF lines (plot first)
            ax.plot(plot_rng, yL, color='blue', linewidth=1, label="Lower MF")
            ax.plot(plot_rng, yU, color='red', linewidth=1, label="Upper MF")
            # FOU region (plot last, on top, lighter color)
            ax.fill_between(plot_rng, yL, yU, alpha=0.15, color='#b266b2', label="FOU")
            ax.set_xlabel(var['name'])
            ax.set_ylabel("Membership")
            ax.set_title(f"FOU for {set_name if set_name else 'Set'} (width: {fou_width:.3f})")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        if warning:
            st.warning(warning)
        with st.form(f"add_set_form_button_it2_{vidx}"):
            addset = st.form_submit_button(f"Add IT2 Fuzzy Set to {var['name']}")
            if addset and set_name and not warning:
                var['sets'].append({
                    "name": set_name,
                    "type": set_type,
                    "lower_params": lower_params,
                    "upper_params": upper_params
                })
                st.success(f"Added IT2 set {set_name} to {var['name']}")
        # Show sets for this variable only
        if var['sets']:
            for sidx, fset in enumerate(var['sets']):
                st.markdown(f"- **{fset['name']}** ({fset['type']}, lower: {fset['lower_params']}, upper: {fset['upper_params']}) ")
                if st.button(f"Remove {fset['name']} from {var['name']}", key=f"delset_it2_{vidx}_{sidx}"):
                    var['sets'].pop(sidx)
                    st.rerun()
        else:
            st.info(f"No sets defined for {var['name']}.")

def symmetric_expand_params(params, width):
    # For Triangular: [a, b, c] => [a-width/2, b, c+width/2]
    # For Trapezoidal: [a, b, c, d] => [a-width/2, b, c, d+width/2]
    n = len(params)
    if n == 3:
        a, b, c = params
        return [a - width/2, b, c + width/2]
    elif n == 4:
        a, b, c, d = params
        return [a - width/2, b, c, d + width/2]
    else:
        return params

def render_visualization_section():
    st.header("3. Visualize IT2 Membership Functions")
    for var in st.session_state.fis_vars:
        if not var['sets']:
            continue
        with st.expander(f"{var['role']}: {var['name']} (show/hide plot)", expanded=False):
            rng = np.linspace(var['range'][0], var['range'][1], 500)
            fig, ax = plt.subplots(figsize=(3, 2.2), dpi=60)
            for idx, fset in enumerate(var['sets']):
                lparams = [float(p.strip()) for p in fset['lower_params'].split(",")]
                uparams = [float(p.strip()) for p in fset['upper_params'].split(",")]
                print(f"DEBUG: Set {fset['name']} lower_params: {lparams}, upper_params: {uparams}")
                yL = np.zeros_like(rng)
                yU = np.zeros_like(rng)
                if fset['type'] == "Triangular" and len(lparams) == 3 and len(uparams) == 3:
                    aL, bL, cL = lparams
                    aU, bU, cU = uparams
                    leftL = np.maximum((rng - aL) / (bL - aL) if bL > aL else 1, 0)
                    rightL = np.maximum((cL - rng) / (cL - bL) if cL > bL else 1, 0)
                    yL = np.maximum(np.minimum(leftL, rightL), 0)
                    leftU = np.maximum((rng - aU) / (bU - aU) if bU > aU else 1, 0)
                    rightU = np.maximum((cU - rng) / (cU - bU) if cU > bU else 1, 0)
                    yU = np.maximum(np.minimum(leftU, rightU), 0)
                elif fset['type'] == "Trapezoidal" and len(lparams) == 4 and len(uparams) == 4:
                    aL, bL, cL, dL = lparams
                    aU, bU, cU, dU = uparams
                    leftL = np.maximum((rng - aL) / (bL - aL) if bL > aL else 1, 0)
                    rightL = np.maximum((dL - rng) / (dL - cL) if dL > cL else 1, 0)
                    yL = np.maximum(np.minimum(np.minimum(leftL, 1), rightL), 0)
                    leftU = np.maximum((rng - aU) / (bU - aU) if bU > aU else 1, 0)
                    rightU = np.maximum((dU - rng) / (dU - cU) if dU > cU else 1, 0)
                    yU = np.maximum(np.minimum(np.minimum(leftU, 1), rightU), 0)
                elif fset['type'] == "Gaussian" and len(lparams) == 2 and len(uparams) == 2:
                    meanL, sigmaL = lparams
                    meanU, sigmaU = uparams
                    yL = np.exp(-0.5*((rng-meanL)/sigmaL)**2)
                    yU = np.exp(-0.5*((rng-meanU)/sigmaU)**2)
                # Lower and upper MF lines (plot first)
                ax.plot(rng, yL, color='blue', linewidth=2, label="Lower MF" if idx==0 else None, zorder=2)
                ax.plot(rng, yU, color='red', linewidth=2, linestyle='--', label="Upper MF" if idx==0 else None, zorder=2)
                # FOU region (plot last, on top, lighter color)
                ax.fill_between(rng, yL, yU, alpha=0.15, color='#b266b2', label="FOU" if idx==0 else "_nolegend_", zorder=3)
            ax.set_xlabel(var['name'])
            ax.set_ylabel("Membership")
            ax.set_title(f"IT2 Membership Functions for {var['name']}")
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            new_handles, new_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen and l != "_nolegend_":
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            ax.legend(new_handles, new_labels, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

def render_rules_section():
    st.header("4. Define IT2 Fuzzy Rules")
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
                if cols[0].button("Edit", key=f"edit_rule_it2_{ridx}"):
                    st.session_state.edit_rule_idx = ridx
                if cols[1].button("Delete", key=f"delrule_it2_{ridx}"):
                    st.session_state.fis_rules.pop(ridx)
                    st.rerun()
        else:
            st.info("No rules defined yet.")
        add_mode = st.session_state.edit_rule_idx is None
        if add_mode:
            st.subheader("Add New Rule")
        else:
            st.subheader(f"Edit Rule #{st.session_state.edit_rule_idx+1}")
        with st.form("rule_form_it2"):
            rule_conds = []
            for var in input_vars:
                set_names = [s['name'] for s in var['sets']]
                default_idx = 0
                if not add_mode and st.session_state.fis_rules[st.session_state.edit_rule_idx]:
                    for vname, sname in st.session_state.fis_rules[st.session_state.edit_rule_idx]['if']:
                        if vname == var['name'] and sname in set_names:
                            default_idx = set_names.index(sname)
                cond = st.selectbox(f"IF {var['name']} is", set_names, index=default_idx if set_names else 0, key=f"ruleformif_it2_{var['name']}") if set_names else None
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
                sel_out_var = st.selectbox("THEN output variable", out_var_names, index=default_outvar, key="ruleformthenvar_it2")
                out_sets = [s['name'] for s in output_vars[out_var_names.index(sel_out_var)]['sets']]
                sel_out_set = st.selectbox("THEN output set", out_sets, index=default_outset if out_sets else 0, key="ruleformthenset_it2") if out_sets else None
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
    st.header("5. IT2 Fuzzy Inference & Result")
    inputs = {}
    debug_mode = st.checkbox("Debug Mode (IT2)", value=False, help="Show detailed calculation steps for IT2")
    for var in st.session_state.fis_vars:
        if var['role'] == "Input":
            val = st.number_input(
                f"Input value for {var['name']}",
                min_value=float(var['range'][0]),
                max_value=float(var['range'][1]),
                value=float(np.mean(var['range'])),
                step=0.1,
                key=f"input_it2_{var['name']}"
            )
            inputs[var['name']] = val
            if debug_mode and var['sets']:
                memberships = fuzzify_it2(val, var['sets'])
                st.write(f"IT2 Membership values for {var['name']} = {val}:")
                for set_name, (muL, muU) in memberships.items():
                    st.write(f"  - {set_name}: lower={muL:.4f}, upper={muU:.4f}")
    if st.button("Run IT2 Inference", key="run_inference_it2") and st.session_state.fis_rules:
        try:
            output_results, rule_trace = run_fuzzy_inference_it2(st.session_state.fis_vars, st.session_state.fis_rules, inputs, debug_mode)
            for out_var in [v for v in st.session_state.fis_vars if v['role']=="Output"]:
                if out_var['name'] in output_results:
                    yl, yu = output_results[out_var['name']]
                    st.success(f"Output {out_var['name']} = [{yl:.4f}, {yu:.4f}] (interval)")
                    # Output MF plot
                    rng = np.linspace(out_var['range'][0], out_var['range'][1], 500)
                    lower_agg = np.zeros(500)
                    upper_agg = np.zeros(500)
                    for rule in st.session_state.fis_rules:
                        if rule['then'][0] != out_var['name']:
                            continue
                        strengthL = 1.0
                        strengthU = 1.0
                        for vname, sname in rule['if']:
                            var = next(v for v in st.session_state.fis_vars if v['name']==vname)
                            memberships = fuzzify_it2(inputs[vname], var['sets'])
                            muL, muU = memberships.get(sname, (0.0, 0.0))
                            strengthL = min(strengthL, muL)
                            strengthU = min(strengthU, muU)
                        setname = rule['then'][1]
                        fset = next(s for s in out_var['sets'] if s['name']==setname)
                        lparams = [float(p.strip()) for p in fset['lower_params'].split(",")]
                        uparams = [float(p.strip()) for p in fset['upper_params'].split(",")]
                        # Lower MF
                        if fset['type'] == "Triangular" and len(lparams) == 3:
                            a, b, c = lparams
                            yL = np.zeros_like(rng)
                            mask = (rng >= a) & (rng <= c)
                            mask_left = (rng >= a) & (rng < b) & mask
                            if b > a and np.any(mask_left):
                                yL[mask_left] = (rng[mask_left] - a) / (b - a)
                            mask_right = (rng > b) & (rng <= c) & mask
                            if c > b and np.any(mask_right):
                                yL[mask_right] = (c - rng[mask_right]) / (c - b)
                            yL[rng == b] = 1.0
                        elif fset['type'] == "Trapezoidal" and len(lparams) == 4:
                            a, b, c, d = lparams
                            yL = np.zeros_like(rng)
                            mask_left = (rng >= a) & (rng < b)
                            if b > a and np.any(mask_left):
                                yL[mask_left] = (rng[mask_left] - a) / (b - a)
                            mask_flat = (rng >= b) & (rng <= c)
                            yL[mask_flat] = 1.0
                            mask_right = (rng > c) & (rng <= d)
                            if d > c and np.any(mask_right):
                                yL[mask_right] = (d - rng[mask_right]) / (d - c)
                        elif fset['type'] == "Gaussian" and len(lparams) == 2:
                            mean, sigma = lparams
                            if sigma == 0:
                                yL = np.zeros_like(rng)
                                yL[rng == mean] = 1.0
                            else:
                                yL = np.exp(-0.5*((rng-mean)/sigma)**2)
                        else:
                            yL = np.zeros_like(rng)
                        # Upper MF
                        if fset['type'] == "Triangular" and len(uparams) == 3:
                            a, b, c = uparams
                            yU = np.zeros_like(rng)
                            mask = (rng >= a) & (rng <= c)
                            mask_left = (rng >= a) & (rng < b) & mask
                            if b > a and np.any(mask_left):
                                yU[mask_left] = (rng[mask_left] - a) / (b - a)
                            mask_right = (rng > b) & (rng <= c) & mask
                            if c > b and np.any(mask_right):
                                yU[mask_right] = (c - rng[mask_right]) / (c - b)
                            yU[rng == b] = 1.0
                        elif fset['type'] == "Trapezoidal" and len(uparams) == 4:
                            a, b, c, d = uparams
                            yU = np.zeros_like(rng)
                            mask_left = (rng >= a) & (rng < b)
                            if b > a and np.any(mask_left):
                                yU[mask_left] = (rng[mask_left] - a) / (b - a)
                            mask_flat = (rng >= b) & (rng <= c)
                            yU[mask_flat] = 1.0
                            mask_right = (rng > c) & (rng <= d)
                            if d > c and np.any(mask_right):
                                yU[mask_right] = (d - rng[mask_right]) / (d - c)
                        elif fset['type'] == "Gaussian" and len(uparams) == 2:
                            mean, sigma = uparams
                            if sigma == 0:
                                yU = np.zeros_like(rng)
                                yU[rng == mean] = 1.0
                            else:
                                yU = np.exp(-0.5*((rng-mean)/sigma)**2)
                        else:
                            yU = np.zeros_like(rng)
                        lower_agg = np.maximum(lower_agg, np.minimum(strengthL, yL))
                        upper_agg = np.maximum(upper_agg, np.minimum(strengthU, yU))
                    fig, ax = plt.subplots(figsize=(3, 1.2), dpi=60)
                    ax.fill_between(rng, lower_agg, upper_agg, alpha=0.3, color='blue')
                    ax.plot(rng, lower_agg, color='blue', linewidth=2, label="Lower MF")
                    ax.plot(rng, upper_agg, color='red', linewidth=2, label="Upper MF")
                    ax.axvline(yl, color="green", linestyle="--", linewidth=2, label=f"Type-Reduced Lower: {yl:.3f}")
                    ax.axvline(yu, color="orange", linestyle="--", linewidth=2, label=f"Type-Reduced Upper: {yu:.3f}")
                    ax.set_xlabel(out_var['name'], fontsize=12)
                    ax.set_ylabel("Membership", fontsize=12)
                    ax.set_title(f"IT2 Output for {out_var['name']}", fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=10, loc='best')
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
            if rule_trace:
                st.subheader("IT2 Rule Firing Trace")
                st.write("The following rules fired and contributed to the output:")
                st.dataframe(pd.DataFrame(rule_trace))
        except Exception as e:
            st.error(f"Error during IT2 inference: {str(e)}")
    elif st.session_state.fis_rules:
        st.info("Press 'Run IT2 Inference' to compute output.")
    else:
        st.info("Define at least one rule to run IT2 inference.")

def render_export_section():
    if st.button("Export IT2 to Python Code", key="export_py_code_it2"):
        code = generate_python_code_it2(st.session_state.get("fis_vars", []), st.session_state.get("fis_rules", []))
        st.download_button(
            label="Download IT2 Python Code",
            data=code,
            file_name="it2_fuzzy_system_export.py",
            mime="text/x-python"
        )
    if st.button("Save IT2 FIS Configuration to JSON", key="save_fis_config_it2"):
        fis_config = {
            "variables": st.session_state.get("fis_vars", []),
            "rules": st.session_state.get("fis_rules", [])
        }
        fis_json = json.dumps(fis_config, indent=2)
        st.download_button(
            label="Download IT2 FIS Configuration as JSON",
            data=fis_json,
            file_name="it2_fuzzy_system_config.json",
            mime="application/json"
        )

def render_upload_config_section():
    uploaded_file = st.file_uploader("Upload IT2 FIS Configuration JSON", type=["json"], key="upload_fis_config_it2")
    if uploaded_file is not None and "fis_config_loaded_it2" not in st.session_state:
        try:
            config = json.load(uploaded_file)
            if "variables" in config and "rules" in config:
                st.session_state.fis_vars = config["variables"]
                st.session_state.fis_rules = config["rules"]
                st.session_state.edit_rule_idx = None
                st.session_state.fis_config_loaded_it2 = True
                st.success("IT2 FIS configuration loaded from JSON!")
                st.rerun()
            else:
                st.error("Invalid configuration file: missing 'variables' or 'rules' keys.")
        except Exception as e:
            st.error(f"Failed to load IT2 configuration: {str(e)}")
    elif "fis_config_loaded_it2" in st.session_state:
        del st.session_state["fis_config_loaded_it2"] 