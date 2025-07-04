import json

def generate_python_code(fis_vars, fis_rules):
    mf_funcs = '''
import numpy as np

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a) if b != a else 1, (c-x)/(c-b) if c != b else 1))

def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a) if b != a else 1, 1), (d-x)/(d-c) if d != c else 1))

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) if sigma > 0 else float(x == mean)
'''
    fuzzify_blocks = []
    input_vars = [var for var in fis_vars if var['role'] == 'Input']
    output_vars = [var for var in fis_vars if var['role'] == 'Output']
    for var in fis_vars:
        if not var['sets']:
            continue
        lines = [f'def fuzzify_{var["name"].replace(" ", "_").lower()}(x):', '    return {']
        for s in var['sets']:
            params = ', '.join(s['params'].split(','))
            if s['type'] == 'Triangular':
                mf_call = f'triangular(x, {params})'
            elif s['type'] == 'Trapezoidal':
                mf_call = f'trapezoidal(x, {params})'
            elif s['type'] == 'Gaussian':
                mf_call = f'gaussian(x, {params})'
            else:
                mf_call = '0'
            lines.append(f'        "{s["name"]}": {mf_call},')
        lines.append('    }')
        fuzzify_blocks.append('\n'.join(lines))
    infer_lines = [
        'def fuzzy_infer(inputs):',
        '    # Fuzzification',
    ]
    for var in input_vars:
        infer_lines.append(f'    {var["name"].replace(" ", "_").lower()}_mf = fuzzify_{var["name"].replace(" ", "_").lower()}(inputs["{var["name"]}"])')
    infer_lines.append('    # Rule evaluation')
    rule_outputs = []
    for i, rule in enumerate(fis_rules):
        conds = []
        for v, s in rule['if']:
            conds.append(f'{v.replace(" ", "_").lower()}_mf["{s}"]')
        rule_strength = f'min({", ".join(conds)})' if conds else '0'
        out_var, out_set = rule['then']
        rule_outputs.append((rule_strength, out_var, out_set))
    out_vars = [v['name'] for v in fis_vars if v['role'] == 'Output']
    infer_lines.append('    results = {}')
    for out_var in out_vars:
        infer_lines.append(f'    # Aggregate for output: {out_var}')
        sets = [s for v in fis_vars if v['name'] == out_var for s in v['sets']]
        infer_lines.append(f'    rng = np.linspace({[v for v in fis_vars if v["name"]==out_var][0]["range"][0]}, {[v for v in fis_vars if v["name"]==out_var][0]["range"][1]}, 500)')
        infer_lines.append(f'    agg_y = np.zeros_like(rng)')
        for i, (rule_strength, r_out_var, r_out_set) in enumerate(rule_outputs):
            if r_out_var == out_var:
                set_obj = [s for v in fis_vars if v['name']==out_var for s in v['sets'] if s['name']==r_out_set][0]
                params = ', '.join(set_obj['params'].split(','))
                if set_obj['type'] == 'Triangular':
                    mf_call = f'triangular(rng, {params})'
                elif set_obj['type'] == 'Trapezoidal':
                    mf_call = f'trapezoidal(rng, {params})'
                elif set_obj['type'] == 'Gaussian':
                    mf_call = f'gaussian(rng, {params})'
                else:
                    mf_call = 'np.zeros_like(rng)'
                infer_lines.append(f'    agg_y = np.maximum(agg_y, np.minimum({rule_strength}, {mf_call}))')
        infer_lines.append(f'    if np.sum(agg_y) > 0:')
        infer_lines.append(f'        centroid = np.sum(rng * agg_y) / np.sum(agg_y)')
        infer_lines.append(f'    else:')
        infer_lines.append(f'        centroid = float(np.mean(rng))')
        infer_lines.append(f'    results["{out_var}"] = centroid')
    infer_lines.append('    return results')
    
    # Interactive CLI
    cli_lines = [
        'if __name__ == "__main__":',
        '    # Define the fuzzy system configuration',
        '    fis_vars = ' + repr(fis_vars),
        '    fis_rules = ' + repr(fis_rules),
        '    input_vars = [v for v in fis_vars if v["role"] == "Input"]',
        '    print("=" * 60)',
        '    print("TYPE-1 FUZZY INFERENCE SYSTEM")',
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
        '    print("Running Type-1 Fuzzy Inference...")',
        '    print()',
        '    # Run inference',
        '    result = fuzzy_infer(inputs)',
        '    print("Results:")',
        '    print("-" * 40)',
        '    for k, v in result.items():',
        '        print(f"  {k}: {v:.4f}")',
        '    print()',
        '    print("=" * 60)',
        '    print("Inference Complete!")',
        '    print("=" * 60)',
    ])
    
    code = mf_funcs + '\n\n'.join(fuzzify_blocks) + '\n\n' + '\n'.join(infer_lines) + '\n\n' + '\n'.join(cli_lines)
    return code 