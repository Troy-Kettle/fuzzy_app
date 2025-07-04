import numpy as np
from .fuzzy_fuzzification import fuzzify
from ..fuzzy_utils import safe_div

def run_fuzzy_inference(fis_vars, fis_rules, inputs, debug_mode=False):
    output_results = {}
    rule_trace = []
    plots = []
    for out_var in [v for v in fis_vars if v['role']=="Output"]:
        if not out_var['sets']:
            continue
        agg_y = np.zeros(500)
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
            strength = 1.0
            rule_conditions = []
            for vname, sname in rule['if']:
                var = next(v for v in fis_vars if v['name']==vname)
                if not any(s['name'] == sname for s in var['sets']):
                    valid_rule = False
                    break
                memberships = fuzzify(inputs[vname], var['sets'])
                if sname not in memberships:
                    valid_rule = False
                    break
                mu = memberships[sname]
                rule_conditions.append((vname, sname, mu))
                strength = min(strength, mu)
            if not valid_rule:
                continue
            setname = rule['then'][1]
            if not any(s['name'] == setname for s in out_var['sets']):
                continue
            fset = next(s for s in out_var['sets'] if s['name']==setname)
            params = [float(p.strip()) for p in fset['params'].split(",")]
            if fset['type'] == "Triangular" and len(params) == 3:
                a, b, c = params
                y = np.zeros_like(rng)
                mask = (rng >= a) & (rng <= c)
                mask_left = (rng >= a) & (rng < b) & mask
                if b > a and np.any(mask_left):
                    y[mask_left] = (rng[mask_left] - a) / (b - a)
                mask_right = (rng > b) & (rng <= c) & mask
                if c > b and np.any(mask_right):
                    y[mask_right] = (c - rng[mask_right]) / (c - b)
                y[rng == b] = 1.0
            elif fset['type'] == "Trapezoidal" and len(params) == 4:
                a, b, c, d = params
                y = np.zeros_like(rng)
                mask_left = (rng >= a) & (rng < b)
                if b > a and np.any(mask_left):
                    y[mask_left] = (rng[mask_left] - a) / (b - a)
                mask_flat = (rng >= b) & (rng <= c)
                y[mask_flat] = 1.0
                mask_right = (rng > c) & (rng <= d)
                if d > c and np.any(mask_right):
                    y[mask_right] = (d - rng[mask_right]) / (d - c)
            elif fset['type'] == "Gaussian" and len(params) == 2:
                mean, sigma = params
                if sigma == 0:
                    y = np.zeros_like(rng)
                    y[rng == mean] = 1.0
                else:
                    y = np.exp(-0.5*((rng-mean)/sigma)**2)
            else:
                y = np.zeros_like(rng)
            agg_y = np.maximum(agg_y, np.minimum(strength, y))
            rules_fired = True
            if valid_rule:
                rule_trace.append({
                    'Rule': f"IF {' AND '.join([f'{vname} is {sname}' for vname, sname, _ in rule_conditions])} THEN {rule['then'][0]} is {rule['then'][1]}",
                    'Firing Strength': strength,
                    'Output Variable': rule['then'][0],
                    'Output Set': rule['then'][1]
                })
        if rules_fired and np.sum(agg_y) > 0:
            centroid = np.sum(rng * agg_y) / np.sum(agg_y)
            output_results[out_var['name']] = centroid
        else:
            centroid = float(np.mean(out_var['range']))
            output_results[out_var['name']] = centroid
    return output_results, rule_trace 