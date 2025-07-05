"""
Interval Type-2 Fuzzy Logic System User Interface Components

This module provides comprehensive UI components for building and interacting with
Interval Type-2 Fuzzy Inference Systems (IT2 FIS). It includes functionality for
variable definition, fuzzy set creation, rule management, inference execution,
and system export capabilities.

The module utilises Streamlit for the web interface and integrates with the core
IT2 fuzzy logic engine for fuzzification, inference, and defuzzification processes.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.core.type2.fuzzy_fuzzification_it2 import fuzzify_it2
from src.core.type2.fuzzy_inference_it2 import run_fuzzy_inference_it2
from src.export.fuzzy_export_it2 import generate_python_code_it2
import json

def load_tipper_it2():
    """
    Loads a pre-configured Interval Type-2 Fuzzy Inference System for restaurant tipping.
    
    This example demonstrates a classic fuzzy logic application where food quality
    and service level determine the appropriate tip percentage. The system utilises
    triangular membership functions with Footprint of Uncertainty (FOU) to model
    uncertainty in human decision-making processes.
    
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

def load_washing_machine_it2():
    """
    Loads a pre-configured Interval Type-2 Fuzzy Inference System for washing machine control.
    
    This example demonstrates an industrial control application where dirtiness level
    and load size determine optimal wash cycle duration. The system employs triangular
    membership functions with FOU to handle sensor uncertainty and varying load conditions.
    
    Variables:
        - Dirtiness (Input): 0-10 scale representing soil level
        - Load Size (Input): 0-10 scale representing laundry quantity
        - Wash Time (Output): 0-60 minutes cycle duration
        
    Returns:
        None: Updates session state with FIS configuration
    """
    st.session_state.fis_vars = [
        {"name": "Dirtiness", "role": "Input", "range": [0, 10], "sets": [
            {"name": "low", "type": "Triangular", "lower_params": "0, 0, 5", "upper_params": "0, 0, 7"},
            {"name": "medium", "type": "Triangular", "lower_params": "0, 5, 10", "upper_params": "0, 3, 10"},
            {"name": "high", "type": "Triangular", "lower_params": "5, 10, 10", "upper_params": "3, 10, 10"}
        ]},
        {"name": "Load Size", "role": "Input", "range": [0, 10], "sets": [
            {"name": "small", "type": "Triangular", "lower_params": "0, 0, 5", "upper_params": "0, 0, 7"},
            {"name": "medium", "type": "Triangular", "lower_params": "0, 5, 10", "upper_params": "0, 3, 10"},
            {"name": "large", "type": "Triangular", "lower_params": "5, 10, 10", "upper_params": "3, 10, 10"}
        ]},
        {"name": "Wash Time", "role": "Output", "range": [0, 60], "sets": [
            {"name": "short", "type": "Triangular", "lower_params": "0, 0, 30", "upper_params": "0, 0, 40"},
            {"name": "medium", "type": "Triangular", "lower_params": "0, 30, 60", "upper_params": "0, 20, 60"},
            {"name": "long", "type": "Triangular", "lower_params": "30, 60, 60", "upper_params": "20, 60, 60"}
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
    st.success("Loaded IT2 Washing Machine example!")
    st.rerun()

def load_room_heater_it2():
    """
    Loads a pre-configured Interval Type-2 Fuzzy Inference System for room heating control.
    
    This example demonstrates a climate control application where ambient temperature
    determines heater power output. The system utilises triangular membership functions
    with FOU to account for temperature sensor uncertainty and environmental variations.
    
    Variables:
        - Temperature (Input): 0-40°C ambient temperature
        - Heater Power (Output): 0-100% power output
        
    Returns:
        None: Updates session state with FIS configuration
    """
    st.session_state.fis_vars = [
        {"name": "Temperature", "role": "Input", "range": [0, 40], "sets": [
            {"name": "cold", "type": "Triangular", "lower_params": "0, 0, 20", "upper_params": "0, 0, 25"},
            {"name": "comfortable", "type": "Triangular", "lower_params": "10, 20, 30", "upper_params": "8, 20, 32"},
            {"name": "hot", "type": "Triangular", "lower_params": "20, 40, 40", "upper_params": "15, 40, 40"}
        ]},
        {"name": "Heater Power", "role": "Output", "range": [0, 100], "sets": [
            {"name": "low", "type": "Triangular", "lower_params": "0, 0, 50", "upper_params": "0, 0, 60"},
            {"name": "medium", "type": "Triangular", "lower_params": "0, 50, 100", "upper_params": "0, 40, 100"},
            {"name": "high", "type": "Triangular", "lower_params": "50, 100, 100", "upper_params": "40, 100, 100"}
        ]}
    ]
    st.session_state.fis_rules = [
        {"if": [("Temperature", "cold")], "then": ("Heater Power", "high")},
        {"if": [("Temperature", "comfortable")], "then": ("Heater Power", "medium")},
        {"if": [("Temperature", "hot")], "then": ("Heater Power", "low")}
    ]
    st.session_state.edit_rule_idx = None
    st.success("Loaded IT2 Room Heater example!")
    st.rerun()

def render_presets_section():
    """
    Renders the preset examples section for Interval Type-2 Fuzzy Inference Systems.
    
    Provides users with three pre-configured IT2 FIS examples to demonstrate
    different application domains: service industry (tipping), industrial control
    (washing machine), and climate control (room heating).
    
    Returns:
        None: Updates session state with selected example configuration
    """
    st.subheader("Preset IT2 Examples")
    col1, col2, col3 = st.columns(3)
    if col1.button("Load IT2 Tipper Example"):
        load_tipper_it2()
    if col2.button("Load IT2 Washing Machine Example"):
        load_washing_machine_it2()
    if col3.button("Load IT2 Room Heater Example"):
        load_room_heater_it2()

def render_variable_section():
    """
    Renders the variable definition section for Interval Type-2 Fuzzy Inference Systems.
    
    Allows users to define input and output variables with their respective ranges.
    Variables serve as the foundation for the fuzzy system, defining the domain
    of discourse for each linguistic variable in the IT2 FIS.
    
    Features:
        - Variable name specification
        - Role assignment (Input/Output)
        - Range definition with validation
        - Variable management (add/remove)
        
    Returns:
        None: Updates session state with variable definitions
    """
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
    """
    Renders the fuzzy set definition section for Interval Type-2 Fuzzy Inference Systems.
    
    Provides comprehensive tools for creating and managing IT2 fuzzy sets with
    Footprint of Uncertainty (FOU). Supports triangular, trapezoidal, and Gaussian
    membership functions with real-time visualisation and parameter validation.
    
    Features:
        - Interactive fuzzy set editor with real-time preview
        - FOU width configuration for uncertainty modelling
        - Parameter validation ensuring proper IT2 constraints
        - Visual feedback for membership function shapes
        - Set management (add/remove) with summary display
        
    Returns:
        None: Updates session state with fuzzy set definitions
    """
    st.header("2. Define Interval Type-2 Fuzzy Sets")
    for vidx, var in enumerate(st.session_state.fis_vars):
        st.subheader(f"{var['role']}: {var['name']}")
        
        # Determine if this variable has fuzzy sets defined
        has_sets = len(var['sets']) > 0
        
        if has_sets:
            # Display minimised editor for variables with existing sets
            with st.expander(f"Show/hide fuzzy set editor for {var['name']}", expanded=False):
                render_fuzzy_set_editor(var, vidx)
        else:
            # Display full editor for variables without sets
            render_fuzzy_set_editor(var, vidx)
    
    # Display comprehensive summary of all defined fuzzy sets
    all_sets_defined = any(len(var['sets']) > 0 for var in st.session_state.fis_vars)
    if all_sets_defined:
        st.subheader("Defined Fuzzy Sets Summary")
        for var_idx, var in enumerate(st.session_state.fis_vars):
            if var['sets']:
                st.write(f"**{var['role']}: {var['name']}**")
                for set_idx, fset in enumerate(var['sets']):
                    if 'lower_params' in fset and 'upper_params' in fset:
                        st.write(f"  - {fset['name']} ({fset['type']}, lower: {fset['lower_params']}, upper: {fset['upper_params']})")
                    elif 'params' in fset:
                        st.write(f"  - {fset['name']} ({fset['type']}, params: {fset['params']})")
                    else:
                        st.write(f"  - {fset['name']} ({fset['type']}, unknown params format)")
                    if st.button(f"Remove {fset['name']} from {var['name']}", key=f"delset_summary_{var_idx}_{set_idx}_{var['name']}_{fset['name']}"):
                        var['sets'].remove(fset)
                        st.rerun()

def render_fuzzy_set_editor(var, vidx):
    """
    Renders the interactive fuzzy set editor for Interval Type-2 membership functions.
    
    Provides a comprehensive interface for creating and configuring IT2 fuzzy sets
    with real-time visualisation. Supports triangular, trapezoidal, and Gaussian
    membership functions with Footprint of Uncertainty (FOU) parameterisation.
    
    Args:
        var (dict): Variable dictionary containing name, role, range, and sets
        vidx (int): Variable index for unique key generation
        
    Features:
        - Parameter sliders with range validation
        - FOU width configuration for uncertainty modelling
        - Real-time membership function visualisation
        - Parameter constraint validation
        - Boundary case handling for edge conditions
        
    Returns:
        None: Updates variable's fuzzy sets in session state
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        set_name = st.text_input(f"Set Name for {var['name']}", key=f"setname_it2_{vidx}")
        set_type = st.selectbox(f"Type", ["Triangular", "Trapezoidal", "Gaussian"], key=f"settype_it2_{vidx}")
    with col2:
        rng = [float(x) for x in var['range']]
        fou_width = st.number_input(
            f"FOU width (absolute, e.g. 0.1)",
            min_value=0.0,
            max_value=(rng[1]-rng[0]),
            value=st.session_state.get(f"fou_width_it2_{vidx}_{set_name}", 0.1),
            step=0.01,
            key=f"fou_width_it2_{vidx}_{set_name}"
        )
        warning = None
        
        # Calculate membership function parameters based on selected set type
        if set_type == "Triangular":
            aL = st.slider(f"a", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"la_it2_{vidx}", float(rng[0])), step=0.01, key=f"la_it2_{vidx}")
            bL = st.slider(f"b", min_value=aL, max_value=float(rng[1]), value=st.session_state.get(f"lb_it2_{vidx}", float((rng[0]+rng[1])/2)), step=0.01, key=f"lb_it2_{vidx}")
            cL = st.slider(f"c", min_value=bL, max_value=float(rng[1]), value=st.session_state.get(f"lc_it2_{vidx}", float(rng[1])), step=0.01, key=f"lc_it2_{vidx}")
            # Ensure FOU creates proper separation and handle boundary conditions
            range_width = rng[1] - rng[0]
            fou_adjustment = min(fou_width, range_width * 0.3)  # Maximum 30% of range width
            
            # Special handling for boundary conditions where parameters are at range limits
            if aL == rng[0] and cL == rng[1]:
                # When both a and c are at boundaries, create separation by adjusting b
                separation = fou_adjustment  # Separation proportional to FOU width
                aU = rng[0]
                bU = bL + separation
                cU = rng[1]
            else:
                # Standard FOU calculation for non-boundary cases
                aU = max(rng[0], aL - fou_adjustment)
                bU = bL  # Maintain peak position at centre
                cU = min(rng[1], cL + fou_adjustment)
            
            lower_params = f"{aL}, {bL}, {cL}"
            upper_params = f"{aU}, {bU}, {cU}"
            if not (aU <= aL <= bL <= cL <= cU):
                warning = "Upper MF must contain lower MF: ensure aU <= aL <= bL <= cL <= cU."
        elif set_type == "Trapezoidal":
            aL = st.slider(f"a", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"la_it2_{vidx}", float(rng[0])), step=0.01, key=f"la_it2_{vidx}")
            bL = st.slider(f"b", min_value=aL, max_value=float(rng[1]), value=st.session_state.get(f"lb_it2_{vidx}", float(rng[0]+(rng[1]-rng[0])/3)), step=0.01, key=f"lb_it2_{vidx}")
            cL = st.slider(f"c", min_value=bL, max_value=float(rng[1]), value=st.session_state.get(f"lc_it2_{vidx}", float(rng[0]+2*(rng[1]-rng[0])/3)), step=0.01, key=f"lc_it2_{vidx}")
            # Handle boundary condition for dL slider: when cL equals range maximum
            if cL == rng[1]:
                dL = rng[1]
                st.slider(f"d", min_value=cL, max_value=float(rng[1]), value=dL, step=0.01, key=f"ld_it2_{vidx}", disabled=True)
            else:
                dL = st.slider(f"d", min_value=cL, max_value=float(rng[1]), value=st.session_state.get(f"ld_it2_{vidx}", float(rng[1])), step=0.01, key=f"ld_it2_{vidx}")
            # Ensure FOU creates proper separation and handle boundary conditions
            range_width = rng[1] - rng[0]
            fou_adjustment = min(fou_width, range_width * 0.3)  # Maximum 30% of range width
            
            # Special handling for boundary conditions where parameters are at range limits
            if aL == rng[0] and dL == rng[1]:
                # When both a and d are at boundaries, create separation by adjusting b and c
                separation = fou_adjustment  # Separation proportional to FOU width
                aU = rng[0]
                bU = bL + separation
                cU = cL - separation
                dU = rng[1]
            else:
                # Standard FOU calculation for non-boundary cases
                aU = max(rng[0], aL - fou_adjustment)
                bU = bL  # Maintain left shoulder position
                cU = cL  # Maintain right shoulder position
                dU = min(rng[1], dL + fou_adjustment)
            
            lower_params = f"{aL}, {bL}, {cL}, {dL}"
            upper_params = f"{aU}, {bU}, {cU}, {dU}"
            if not (aU <= aL <= bL <= cL <= dL <= dU):
                warning = "Upper MF must contain lower MF: ensure aU <= aL <= bL <= cL <= dL <= dU."
        else:
            meanL = st.slider(f"mean", min_value=float(rng[0]), max_value=float(rng[1]), value=st.session_state.get(f"lmean_it2_{vidx}", float((rng[0]+rng[1])/2)), step=0.01, key=f"lmean_it2_{vidx}")
            sigmaL = st.slider(f"sigma", min_value=0.01, max_value=float(rng[1]-rng[0]), value=st.session_state.get(f"lsigma_it2_{vidx}", 0.1), step=0.01, key=f"lsigma_it2_{vidx}")
            meanU = meanL  # Maintain mean position at centre
            sigmaU = sigmaL + fou_width  # Expand sigma by FOU width for uncertainty
            lower_params = f"{meanL}, {sigmaL}"
            upper_params = f"{meanU}, {sigmaU}"
            if not (sigmaU >= sigmaL):
                warning = "Upper sigma must be >= lower sigma."
        
        # Real-time visualisation of membership functions
        with col3:
            plot_rng = np.linspace(float(rng[0]), float(rng[1]), 500)
            fig, ax = plt.subplots(figsize=(3, 1.2), dpi=60)
            
            # Recalculate upper parameters based on current FOU width for real-time visualisation
            if set_type == "Triangular":
                # Recalculate upper parameters for visualisation
                range_width = rng[1] - rng[0]
                fou_adjustment = min(fou_width, range_width * 0.3)
                
                if aL == rng[0] and cL == rng[1]:
                    # Boundary condition handling
                    separation = range_width * 0.1
                    aU_viz = rng[0]
                    bU_viz = bL + separation
                    cU_viz = rng[1]
                else:
                    # Standard FOU calculation for visualisation
                    aU_viz = max(rng[0], aL - fou_adjustment)
                    bU_viz = bL
                    cU_viz = min(rng[1], cL + fou_adjustment)
                
                leftL = np.maximum((plot_rng - aL) / (bL - aL) if bL > aL else 1, 0)
                rightL = np.maximum((cL - plot_rng) / (cL - bL) if cL > bL else 1, 0)
                yL = np.maximum(np.minimum(leftL, rightL), 0)
                leftU = np.maximum((plot_rng - aU_viz) / (bU_viz - aU_viz) if bU_viz > aU_viz else 1, 0)
                rightU = np.maximum((cU_viz - plot_rng) / (cU_viz - bU_viz) if cU_viz > bU_viz else 1, 0)
                yU = np.maximum(np.minimum(leftU, rightU), 0)
            elif set_type == "Trapezoidal":
                # Recalculate upper parameters for visualisation
                range_width = rng[1] - rng[0]
                fou_adjustment = min(fou_width, range_width * 0.3)
                
                if aL == rng[0] and dL == rng[1]:
                    # Boundary condition handling
                    separation = range_width * 0.1
                    aU_viz = rng[0]
                    bU_viz = bL + separation
                    cU_viz = cL - separation
                    dU_viz = rng[1]
                else:
                    # Standard FOU calculation for visualisation
                    aU_viz = max(rng[0], aL - fou_adjustment)
                    bU_viz = bL
                    cU_viz = cL
                    dU_viz = min(rng[1], dL + fou_adjustment)
                
                leftL = np.maximum((plot_rng - aL) / (bL - aL) if bL > aL else 1, 0)
                rightL = np.maximum((dL - plot_rng) / (dL - cL) if dL > cL else 1, 0)
                yL = np.maximum(np.minimum(np.minimum(leftL, 1), rightL), 0)
                leftU = np.maximum((plot_rng - aU_viz) / (bU_viz - aU_viz) if bU_viz > aU_viz else 1, 0)
                rightU = np.maximum((dU_viz - plot_rng) / (dU_viz - cU_viz) if dU_viz > cU_viz else 1, 0)
                yU = np.maximum(np.minimum(np.minimum(leftU, 1), rightU), 0)
            else:
                # Recalculate upper parameters for Gaussian membership functions
                meanU_viz = meanL
                sigmaU_viz = sigmaL + fou_width
                yL = np.exp(-0.5*((plot_rng-meanL)/sigmaL)**2)
                yU = np.exp(-0.5*((plot_rng-meanU_viz)/sigmaU_viz)**2)
            
            # Plot lower and upper membership function lines (rendered first)
            ax.plot(plot_rng, yL, color='blue', linewidth=1, label="Lower MF")
            ax.plot(plot_rng, yU, color='red', linewidth=1, label="Upper MF")
            # Render FOU region (plotted last, on top, with lighter colour)
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

def symmetric_expand_params(params, width):
    """
    Symmetrically expands membership function parameters to create FOU.
    
    Applies symmetric expansion to membership function parameters to generate
    the Footprint of Uncertainty (FOU) for Interval Type-2 fuzzy sets.
    
    Args:
        params (list): Original membership function parameters
        width (float): FOU width for symmetric expansion
        
    Returns:
        list: Expanded parameters for upper membership function
        
    Note:
        - Triangular: [a, b, c] => [a-width/2, b, c+width/2]
        - Trapezoidal: [a, b, c, d] => [a-width/2, b, c, d+width/2]
    """
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
    """
    Renders the visualisation section for Interval Type-2 membership functions.
    
    Displays comprehensive visualisations of all defined IT2 fuzzy sets with
    their lower and upper membership functions, along with the Footprint of
    Uncertainty (FOU) regions. Provides clear visual feedback for system design.
    
    Features:
        - Complete membership function visualisation
        - FOU region highlighting
        - Multi-set comparison within variables
        - Professional plotting with proper legends
        
    Returns:
        None: Displays visualisations in Streamlit interface
    """
    st.header("3. Visualise IT2 Membership Functions")
    for var in st.session_state.fis_vars:
        if not var['sets']:
            continue
        # Display visualisations without expander for immediate visibility
        st.subheader(f"{var['role']}: {var['name']}")
        rng = np.linspace(var['range'][0], var['range'][1], 500)
        fig, ax = plt.subplots(figsize=(3, 2.2), dpi=60)
        for idx, fset in enumerate(var['sets']):
            lparams = [float(p.strip()) for p in fset['lower_params'].split(",")]
            uparams = [float(p.strip()) for p in fset['upper_params'].split(",")]
            # Debug output for parameter verification
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
            # FOU region (plot last, on top, lighter colour)
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