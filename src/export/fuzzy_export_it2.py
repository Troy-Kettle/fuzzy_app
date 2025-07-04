import numpy as np

def generate_python_code_it2(fis_vars, fis_rules):
    """
    Generate Python code for interval type-2 fuzzy inference system.
    """
    # Membership function definitions
    mf_funcs = '''
import numpy as np

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a) if b != a else 1, (c-x)/(c-b) if c != b else 1))

def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a) if b != a else 1, 1), (d-x)/(d-c) if d != c else 1))

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) if sigma > 0 else float(x == mean)
'''
    # Fuzzification for IT2
    fuzzify_blocks = []
    for var in fis_vars:
        if not var['sets']:
            continue
        lines = [f'def fuzzify_{var["name"].replace(" ", "_").lower()}(x):', '    return {']
        for s in var['sets']:
            lparams = ', '.join(s['lower_params'].split(','))
            uparams = ', '.join(s['upper_params'].split(','))
            if s['type'] == 'Triangular':
                mfL = f'triangular(x, {lparams})'
                mfU = f'triangular(x, {uparams})'
            elif s['type'] == 'Trapezoidal':
                mfL = f'trapezoidal(x, {lparams})'
                mfU = f'trapezoidal(x, {uparams})'
            elif s['type'] == 'Gaussian':
                mfL = f'gaussian(x, {lparams})'
                mfU = f'gaussian(x, {uparams})'
            else:
                mfL = mfU = '0.0'
            lines.append(f'        "{s["name"]}": ({mfL}, {mfU}),')
        lines.append('    }')
        fuzzify_blocks.append('\n'.join(lines))
    # Karnik-Mendel type-reduction
    km_func = '''
def karnik_mendel(rng, lower, upper):
    yl_sum = np.sum(rng * lower)
    yl_total = np.sum(lower)
    yu_sum = np.sum(rng * upper)
    yu_total = np.sum(upper)
    if yl_total > 0:
        yl_centroid = yl_sum / yl_total
    else:
        yl_centroid = float(np.mean(rng))
    if yu_total > 0:
        yu_centroid = yu_sum / yu_total
    else:
        yu_centroid = float(np.mean(rng))
    return yl_centroid, yu_centroid
'''
    # Inference function
    infer_lines = [
        'def run_fuzzy_inference_it2(fis_vars, fis_rules, inputs):',
        '    output_results = {}',
        '    for out_var in [v for v in fis_vars if v["role"]=="Output"]:',
        '        if not out_var["sets"]:',
        '            continue',
        '        lower_agg = np.zeros(500)',
        '        upper_agg = np.zeros(500)',
        '        rng = np.linspace(out_var["range"][0], out_var["range"][1], 500)',
        '        for rule in fis_rules:',
        '            if rule["then"][0] != out_var["name"]:',
        '                continue',
        '            strengthL = 1.0',
        '            strengthU = 1.0',
        '            for vname, sname in rule["if"]:',
        '                var = next(v for v in fis_vars if v["name"]==vname)',
        '                memberships = globals()[f"fuzzify_{vname.replace(\' \', \'_\').lower()}"](inputs[vname])',
        '                muL, muU = memberships.get(sname, (0.0, 0.0))',
        '                strengthL = min(strengthL, muL)',
        '                strengthU = min(strengthU, muU)',
        '            setname = rule["then"][1]',
        '            fset = next(s for s in out_var["sets"] if s["name"]==setname)',
        '            lparams = [float(p.strip()) for p in fset["lower_params"].split(",")]',
        '            uparams = [float(p.strip()) for p in fset["upper_params"].split(",")]',
        '            # Lower MF',
        '            if fset["type"] == "Triangular" and len(lparams) == 3:',
        '                a, b, c = lparams',
        '                yL = np.zeros_like(rng)',
        '                mask = (rng >= a) & (rng <= c)',
        '                mask_left = (rng >= a) & (rng < b) & mask',
        '                if b > a and np.any(mask_left):',
        '                    yL[mask_left] = (rng[mask_left] - a) / (b - a)',
        '                mask_right = (rng > b) & (rng <= c) & mask',
        '                if c > b and np.any(mask_right):',
        '                    yL[mask_right] = (c - rng[mask_right]) / (c - b)',
        '                yL[rng == b] = 1.0',
        '            elif fset["type"] == "Trapezoidal" and len(lparams) == 4:',
        '                a, b, c, d = lparams',
        '                yL = np.zeros_like(rng)',
        '                mask_left = (rng >= a) & (rng < b)',
        '                if b > a and np.any(mask_left):',
        '                    yL[mask_left] = (rng[mask_left] - a) / (b - a)',
        '                mask_flat = (rng >= b) & (rng <= c)',
        '                yL[mask_flat] = 1.0',
        '                mask_right = (rng > c) & (rng <= d)',
        '                if d > c and np.any(mask_right):',
        '                    yL[mask_right] = (d - rng[mask_right]) / (d - c)',
        '            elif fset["type"] == "Gaussian" and len(lparams) == 2:',
        '                mean, sigma = lparams',
        '                if sigma == 0:',
        '                    yL = np.zeros_like(rng)',
        '                    yL[rng == mean] = 1.0',
        '                else:',
        '                    yL = np.exp(-0.5*((rng-mean)/sigma)**2)',
        '            else:',
        '                yL = np.zeros_like(rng)',
        '            # Upper MF',
        '            if fset["type"] == "Triangular" and len(uparams) == 3:',
        '                a, b, c = uparams',
        '                yU = np.zeros_like(rng)',
        '                mask = (rng >= a) & (rng <= c)',
        '                mask_left = (rng >= a) & (rng < b) & mask',
        '                if b > a and np.any(mask_left):',
        '                    yU[mask_left] = (rng[mask_left] - a) / (b - a)',
        '                mask_right = (rng > b) & (rng <= c) & mask',
        '                if c > b and np.any(mask_right):',
        '                    yU[mask_right] = (c - rng[mask_right]) / (c - b)',
        '                yU[rng == b] = 1.0',
        '            elif fset["type"] == "Trapezoidal" and len(uparams) == 4:',
        '                a, b, c, d = uparams',
        '                yU = np.zeros_like(rng)',
        '                mask_left = (rng >= a) & (rng < b)',
        '                if b > a and np.any(mask_left):',
        '                    yU[mask_left] = (rng[mask_left] - a) / (b - a)',
        '                mask_flat = (rng >= b) & (rng <= c)',
        '                yU[mask_flat] = 1.0',
        '                mask_right = (rng > c) & (rng <= d)',
        '                if d > c and np.any(mask_right):',
        '                    yU[mask_right] = (d - rng[mask_right]) / (d - c)',
        '            elif fset["type"] == "Gaussian" and len(uparams) == 2:',
        '                mean, sigma = uparams',
        '                if sigma == 0:',
        '                    yU = np.zeros_like(rng)',
        '                    yU[rng == mean] = 1.0',
        '                else:',
        '                    yU = np.exp(-0.5*((rng-mean)/sigma)**2)',
        '            else:',
        '                yU = np.zeros_like(rng)',
        '            lower_agg = np.maximum(lower_agg, np.minimum(strengthL, yL))',
        '            upper_agg = np.maximum(upper_agg, np.minimum(strengthU, yU))',
        '        if np.sum(lower_agg) > 0 and np.sum(upper_agg) > 0:',
        '            yl_centroid, yu_centroid = karnik_mendel(rng, lower_agg, upper_agg)',
        '            output_results[out_var["name"]] = (yl_centroid, yu_centroid)',
        '        else:',
        '            mean_val = float(np.mean(out_var["range"]))',
        '            output_results[out_var["name"]] = (mean_val, mean_val)',
        '    return output_results'
    ]
    # Interactive CLI
    input_vars = [v for v in fis_vars if v['role'] == 'Input']
    cli_lines = [
        'if __name__ == "__main__":',
        '    # Define the fuzzy system configuration',
        '    fis_vars = ' + repr(fis_vars),
        '    fis_rules = ' + repr(fis_rules),
        '    input_vars = [v for v in fis_vars if v["role"] == "Input"]',
        '    print("=" * 60)',
        '    print("IT2 FUZZY INFERENCE SYSTEM")',
        '    print("=" * 60)',
        '    print()',
        '    print("System Configuration:")',
        '    print(f"  Input Variables: {len(input_vars)}")',
        '    print(f"  Output Variables: {len([v for v in fis_vars if v[\'role\'] == \'Output\'])}")',
        '    print(f"  Rules: {len(fis_rules)}")',
        '    print()',
        '    print("Input Variables:")',
    ]
    
    # Show input variable details
    for var in input_vars:
        cli_lines.append(f'    print(f"  - {var["name"]}: range [{var["range"][0]}, {var["range"][1]}]")')
        for s in var['sets']:
            cli_lines.append(f'    print(f"    * {s["name"]} ({s["type"]})")')
    
    cli_lines.extend([
        '    print()',
        '    print("Output Variables:")',
    ])
    
    # Show output variable details
    for var in fis_vars:
        if var['role'] == 'Output':
            cli_lines.append(f'    print(f"  - {var["name"]}: range [{var["range"][0]}, {var["range"][1]}]")')
            for s in var['sets']:
                cli_lines.append(f'    print(f"    * {s["name"]} ({s["type"]})")')
    
    cli_lines.extend([
        '    print()',
        '    print("Rules:")',
    ])
    
    # Show rules
    for i, rule in enumerate(fis_rules, 1):
        conditions = " AND ".join([f'{v} is {s}' for v, s in rule['if']])
        then_var, then_set = rule['then']
        cli_lines.append(f'    print(f"  Rule {i}: IF {conditions} THEN {then_var} is {then_set}")')
    
    cli_lines.extend([
        '    print()',
        '    print("=" * 60)',
        '    print("INTERACTIVE INFERENCE")',
        '    print("=" * 60)',
        '    print()',
        '    # Get inputs interactively',
        '    inputs = {}',
    ])
    
    # Interactive input prompts
    for var in input_vars:
        cli_lines.extend([
            f'    while True:',
            f'        try:',
            f'            val = float(input(f"Enter value for {var["name"]} (range [{var["range"][0]}, {var["range"][1]}]): "))',
            f'            if {var["range"][0]} <= val <= {var["range"][1]}:',
            f'                inputs["{var["name"]}"] = val',
            f'                break',
            f'            else:',
            f'                print(f"Error: Value must be between {var["range"][0]} and {var["range"][1]}")',
            f'        except ValueError:',
            f'            print("Error: Please enter a valid number")',
        ])
    
    cli_lines.extend([
        '    print()',
        '    print("Running IT2 Fuzzy Inference...")',
        '    print()',
        '    # Set up fuzzification functions',
    ])
    
    # Set up fuzzification functions
    for var in fis_vars:
        if var['sets']:
            cli_lines.append(f'    globals()["fuzzify_{var["name"].replace(" ", "_").lower()}"] = fuzzify_{var["name"].replace(" ", "_").lower()}')
    
    cli_lines.extend([
        '    # Run inference',
        '    result = run_fuzzy_inference_it2(fis_vars, fis_rules, inputs)',
        '    print("Results:")',
        '    print("-" * 40)',
        '    for k, v in result.items():',
        '        print(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}] (interval)")',
        '        print(f"       Average: {(v[0] + v[1]) / 2:.4f}")',
        '    print()',
        '    print("=" * 60)',
        '    print("Inference Complete!")',
        '    print("=" * 60)',
    ])
    
    code = mf_funcs + '\n\n'.join(fuzzify_blocks) + '\n' + km_func + '\n' + '\n'.join(infer_lines) + '\n\n' + '\n'.join(cli_lines)
    return code 