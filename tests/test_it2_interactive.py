
import numpy as np

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a) if b != a else 1, (c-x)/(c-b) if c != b else 1))

def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a) if b != a else 1, 1), (d-x)/(d-c) if d != c else 1))

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) if sigma > 0 else float(x == mean)
def fuzzify_food_quality(x):
    return {
        "bad": (triangular(x, 0,  0,  4), triangular(x, 0,  0,  6)),
        "average": (triangular(x, 2,  5,  8), triangular(x, 0,  5,  10)),
        "good": (triangular(x, 6,  10,  10), triangular(x, 4,  10,  10)),
    }

def fuzzify_service(x):
    return {
        "poor": (triangular(x, 0,  0,  4), triangular(x, 0,  0,  6)),
        "good": (triangular(x, 2,  5,  8), triangular(x, 0,  5,  10)),
        "excellent": (triangular(x, 6,  10,  10), triangular(x, 4,  10,  10)),
    }

def fuzzify_tip(x):
    return {
        "low": (triangular(x, 0,  0,  10), triangular(x, 0,  0,  16)),
        "medium": (triangular(x, 8,  13,  20), triangular(x, 0,  13,  25)),
        "high": (triangular(x, 16,  25,  25), triangular(x, 10,  25,  25)),
    }

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

def run_fuzzy_inference_it2(fis_vars, fis_rules, inputs):
    output_results = {}
    for out_var in [v for v in fis_vars if v["role"]=="Output"]:
        if not out_var["sets"]:
            continue
        lower_agg = np.zeros(500)
        upper_agg = np.zeros(500)
        rng = np.linspace(out_var["range"][0], out_var["range"][1], 500)
        for rule in fis_rules:
            if rule["then"][0] != out_var["name"]:
                continue
            strengthL = 1.0
            strengthU = 1.0
            for vname, sname in rule["if"]:
                var = next(v for v in fis_vars if v["name"]==vname)
                memberships = globals()[f"fuzzify_{vname.replace(' ', '_').lower()}"](inputs[vname])
                muL, muU = memberships.get(sname, (0.0, 0.0))
                strengthL = min(strengthL, muL)
                strengthU = min(strengthU, muU)
            setname = rule["then"][1]
            fset = next(s for s in out_var["sets"] if s["name"]==setname)
            lparams = [float(p.strip()) for p in fset["lower_params"].split(",")]
            uparams = [float(p.strip()) for p in fset["upper_params"].split(",")]
            # Lower MF
            if fset["type"] == "Triangular" and len(lparams) == 3:
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
            elif fset["type"] == "Trapezoidal" and len(lparams) == 4:
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
            elif fset["type"] == "Gaussian" and len(lparams) == 2:
                mean, sigma = lparams
                if sigma == 0:
                    yL = np.zeros_like(rng)
                    yL[rng == mean] = 1.0
                else:
                    yL = np.exp(-0.5*((rng-mean)/sigma)**2)
            else:
                yL = np.zeros_like(rng)
            # Upper MF
            if fset["type"] == "Triangular" and len(uparams) == 3:
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
            elif fset["type"] == "Trapezoidal" and len(uparams) == 4:
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
            elif fset["type"] == "Gaussian" and len(uparams) == 2:
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
        if np.sum(lower_agg) > 0 and np.sum(upper_agg) > 0:
            yl_centroid, yu_centroid = karnik_mendel(rng, lower_agg, upper_agg)
            output_results[out_var["name"]] = (yl_centroid, yu_centroid)
        else:
            mean_val = float(np.mean(out_var["range"]))
            output_results[out_var["name"]] = (mean_val, mean_val)
    return output_results

if __name__ == "__main__":
    print("=" * 60)
    print("IT2 FUZZY INFERENCE SYSTEM")
    print("=" * 60)
    print()
    print("System Configuration:")
    print(f"  Input Variables: {len(input_vars)}")
    print(f"  Output Variables: {len([v for v in fis_vars if v['role'] == 'Output'])}")
    print(f"  Rules: {len(fis_rules)}")
    print()
    print("Input Variables:")
    print(f"  - Food Quality: range [0, 10]")
    print(f"    * bad (Triangular)")
    print(f"    * average (Triangular)")
    print(f"    * good (Triangular)")
    print(f"  - Service: range [0, 10]")
    print(f"    * poor (Triangular)")
    print(f"    * good (Triangular)")
    print(f"    * excellent (Triangular)")
    print()
    print("Output Variables:")
    print(f"  - Tip: range [0, 25]")
    print(f"    * low (Triangular)")
    print(f"    * medium (Triangular)")
    print(f"    * high (Triangular)")
    print()
    print("Rules:")
    print(f"  Rule 1: IF Food Quality is bad AND Service is poor THEN Tip is low")
    print(f"  Rule 2: IF Food Quality is bad AND Service is good THEN Tip is low")
    print(f"  Rule 3: IF Food Quality is bad AND Service is excellent THEN Tip is low")
    print(f"  Rule 4: IF Food Quality is average AND Service is poor THEN Tip is low")
    print(f"  Rule 5: IF Food Quality is good AND Service is poor THEN Tip is low")
    print(f"  Rule 6: IF Food Quality is average AND Service is good THEN Tip is medium")
    print(f"  Rule 7: IF Food Quality is average AND Service is excellent THEN Tip is high")
    print(f"  Rule 8: IF Food Quality is good AND Service is good THEN Tip is high")
    print(f"  Rule 9: IF Food Quality is good AND Service is excellent THEN Tip is high")
    print()
    print("=" * 60)
    print("INTERACTIVE INFERENCE")
    print("=" * 60)
    print()
    # Define the fuzzy system configuration
    fis_vars = [{'name': 'Food Quality', 'role': 'Input', 'range': [0, 10], 'sets': [{'name': 'bad', 'type': 'Triangular', 'lower_params': '0, 0, 4', 'upper_params': '0, 0, 6'}, {'name': 'average', 'type': 'Triangular', 'lower_params': '2, 5, 8', 'upper_params': '0, 5, 10'}, {'name': 'good', 'type': 'Triangular', 'lower_params': '6, 10, 10', 'upper_params': '4, 10, 10'}]}, {'name': 'Service', 'role': 'Input', 'range': [0, 10], 'sets': [{'name': 'poor', 'type': 'Triangular', 'lower_params': '0, 0, 4', 'upper_params': '0, 0, 6'}, {'name': 'good', 'type': 'Triangular', 'lower_params': '2, 5, 8', 'upper_params': '0, 5, 10'}, {'name': 'excellent', 'type': 'Triangular', 'lower_params': '6, 10, 10', 'upper_params': '4, 10, 10'}]}, {'name': 'Tip', 'role': 'Output', 'range': [0, 25], 'sets': [{'name': 'low', 'type': 'Triangular', 'lower_params': '0, 0, 10', 'upper_params': '0, 0, 16'}, {'name': 'medium', 'type': 'Triangular', 'lower_params': '8, 13, 20', 'upper_params': '0, 13, 25'}, {'name': 'high', 'type': 'Triangular', 'lower_params': '16, 25, 25', 'upper_params': '10, 25, 25'}]}]
    fis_rules = [{'if': [('Food Quality', 'bad'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'bad'), ('Service', 'good')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'bad'), ('Service', 'excellent')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'average'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'good'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'average'), ('Service', 'good')], 'then': ('Tip', 'medium')}, {'if': [('Food Quality', 'average'), ('Service', 'excellent')], 'then': ('Tip', 'high')}, {'if': [('Food Quality', 'good'), ('Service', 'good')], 'then': ('Tip', 'high')}, {'if': [('Food Quality', 'good'), ('Service', 'excellent')], 'then': ('Tip', 'high')}]
    input_vars = [v for v in fis_vars if v["role"] == "Input"]
    # Get inputs interactively
    inputs = {}
    while True:
        try:
            val = float(input(f"Enter value for Food Quality (range [0, 10]): "))
            if 0 <= val <= 10:
                inputs["Food Quality"] = val
                break
            else:
                print(f"Error: Value must be between 0 and 10")
        except ValueError:
            print("Error: Please enter a valid number")
    while True:
        try:
            val = float(input(f"Enter value for Service (range [0, 10]): "))
            if 0 <= val <= 10:
                inputs["Service"] = val
                break
            else:
                print(f"Error: Value must be between 0 and 10")
        except ValueError:
            print("Error: Please enter a valid number")
    print()
    print("Running IT2 Fuzzy Inference...")
    print()
    # Set up fuzzification functions
    globals()["fuzzify_food_quality"] = fuzzify_food_quality
    globals()["fuzzify_service"] = fuzzify_service
    globals()["fuzzify_tip"] = fuzzify_tip
    # Run inference
    result = run_fuzzy_inference_it2(fis_vars, fis_rules, inputs)
    print("Results:")
    print("-" * 40)
    for k, v in result.items():
        print(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}] (interval)")
        print(f"       Average: {(v[0] + v[1]) / 2:.4f}")
    print()
    print("=" * 60)
    print("Inference Complete!")
    print("=" * 60)