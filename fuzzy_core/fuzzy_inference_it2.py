import numpy as np
from fuzzy_core.fuzzy_fuzzification_it2 import fuzzify_it2

def karnik_mendel(rng, lower, upper):
    # Simple centroid type-reduction (Karnik-Mendel algorithm, iterative)
    yl = lower
    yu = upper
    # Initial guess
    yl_sum = np.sum(rng * yl)
    yl_total = np.sum(yl)
    yu_sum = np.sum(rng * yu)
    yu_total = np.sum(yu)
    if yl_total > 0:
        yl_centroid = yl_sum / yl_total
    else:
        yl_centroid = float(np.mean(rng))
    if yu_total > 0:
        yu_centroid = yu_sum / yu_total
    else:
        yu_centroid = float(np.mean(rng))
    return yl_centroid, yu_centroid

def run_fuzzy_inference_it2(fis_vars, fis_rules, inputs, debug_mode=False):
    """
    Perform interval type-2 fuzzy inference.
    TODO: Implement IT2 inference (footprint of uncertainty, type-reduction, etc.)
    """
    output_results = {}
    rule_trace = []
    for out_var in [v for v in fis_vars if v['role']=="Output"]:
        if not out_var['sets']:
            continue
        lower_agg = np.zeros(500)
        upper_agg = np.zeros(500)
        rng = np.linspace(out_var['range'][0], out_var['range'][1], 500)
        rules_fired = False
        for rule in fis_rules:
            if rule['then'][0] != out_var['name']:
                continue
            valid_rule = True
            for vname, sname in rule['if']:
                if vname not in inputs:
                    valid_rule = False
                    break
            if not valid_rule:
                continue
            strengthL = 1.0
            strengthU = 1.0
            rule_conditions = []
            for vname, sname in rule['if']:
                var = next(v for v in fis_vars if v['name']==vname)
                if not any(s['name'] == sname for s in var['sets']):
                    valid_rule = False
                    break
                memberships = fuzzify_it2(inputs[vname], var['sets'])
                if sname not in memberships:
                    valid_rule = False
                    break
                muL, muU = memberships[sname]
                rule_conditions.append((vname, sname, muL, muU))
                strengthL = min(strengthL, muL)
                strengthU = min(strengthU, muU)
            if not valid_rule:
                continue
            setname = rule['then'][1]
            if not any(s['name'] == setname for s in out_var['sets']):
                continue
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
            rules_fired = True
            if valid_rule:
                rule_trace.append({
                    'Rule': f"IF {' AND '.join([f'{vname} is {sname}' for vname, sname, _, _ in rule_conditions])} THEN {rule['then'][0]} is {rule['then'][1]}",
                    'Firing Strength Lower': strengthL,
                    'Firing Strength Upper': strengthU,
                    'Output Variable': rule['then'][0],
                    'Output Set': rule['then'][1]
                })
        if rules_fired and (np.sum(lower_agg) > 0 and np.sum(upper_agg) > 0):
            yl_centroid, yu_centroid = karnik_mendel(rng, lower_agg, upper_agg)
            output_results[out_var['name']] = (yl_centroid, yu_centroid)
        else:
            mean_val = float(np.mean(out_var['range']))
            output_results[out_var['name']] = (mean_val, mean_val)
    return output_results, rule_trace 