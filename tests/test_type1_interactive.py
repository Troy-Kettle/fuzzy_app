
import numpy as np

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a) if b != a else 1, (c-x)/(c-b) if c != b else 1))

def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a) if b != a else 1, 1), (d-x)/(d-c) if d != c else 1))

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) if sigma > 0 else float(x == mean)
def fuzzify_food_quality(x):
    return {
        "bad": triangular(x, 0,  0,  4),
        "average": triangular(x, 2,  5,  8),
        "good": triangular(x, 6,  10,  10),
    }

def fuzzify_service(x):
    return {
        "poor": triangular(x, 0,  0,  4),
        "good": triangular(x, 2,  5,  8),
        "excellent": triangular(x, 6,  10,  10),
    }

def fuzzify_tip(x):
    return {
        "low": triangular(x, 0,  0,  10),
        "medium": triangular(x, 8,  13,  20),
        "high": triangular(x, 16,  25,  25),
    }

def fuzzy_infer(inputs):
    # Fuzzification
    food_quality_mf = fuzzify_food_quality(inputs["Food Quality"])
    service_mf = fuzzify_service(inputs["Service"])
    # Rule evaluation
    results = {}
    # Aggregate for output: Tip
    rng = np.linspace(0, 25, 500)
    agg_y = np.zeros_like(rng)
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["bad"], service_mf["poor"]), triangular(rng, 0,  0,  10)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["bad"], service_mf["good"]), triangular(rng, 0,  0,  10)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["bad"], service_mf["excellent"]), triangular(rng, 0,  0,  10)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["average"], service_mf["poor"]), triangular(rng, 0,  0,  10)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["good"], service_mf["poor"]), triangular(rng, 0,  0,  10)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["average"], service_mf["good"]), triangular(rng, 8,  13,  20)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["average"], service_mf["excellent"]), triangular(rng, 16,  25,  25)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["good"], service_mf["good"]), triangular(rng, 16,  25,  25)))
    agg_y = np.maximum(agg_y, np.minimum(min(food_quality_mf["good"], service_mf["excellent"]), triangular(rng, 16,  25,  25)))
    if np.sum(agg_y) > 0:
        centroid = np.sum(rng * agg_y) / np.sum(agg_y)
    else:
        centroid = float(np.mean(rng))
    results["Tip"] = centroid
    return results

if __name__ == "__main__":
    # Define the fuzzy system configuration
    fis_vars = [{'name': 'Food Quality', 'role': 'Input', 'range': [0, 10], 'sets': [{'name': 'bad', 'type': 'Triangular', 'params': '0, 0, 4'}, {'name': 'average', 'type': 'Triangular', 'params': '2, 5, 8'}, {'name': 'good', 'type': 'Triangular', 'params': '6, 10, 10'}]}, {'name': 'Service', 'role': 'Input', 'range': [0, 10], 'sets': [{'name': 'poor', 'type': 'Triangular', 'params': '0, 0, 4'}, {'name': 'good', 'type': 'Triangular', 'params': '2, 5, 8'}, {'name': 'excellent', 'type': 'Triangular', 'params': '6, 10, 10'}]}, {'name': 'Tip', 'role': 'Output', 'range': [0, 25], 'sets': [{'name': 'low', 'type': 'Triangular', 'params': '0, 0, 10'}, {'name': 'medium', 'type': 'Triangular', 'params': '8, 13, 20'}, {'name': 'high', 'type': 'Triangular', 'params': '16, 25, 25'}]}]
    fis_rules = [{'if': [('Food Quality', 'bad'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'bad'), ('Service', 'good')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'bad'), ('Service', 'excellent')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'average'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'good'), ('Service', 'poor')], 'then': ('Tip', 'low')}, {'if': [('Food Quality', 'average'), ('Service', 'good')], 'then': ('Tip', 'medium')}, {'if': [('Food Quality', 'average'), ('Service', 'excellent')], 'then': ('Tip', 'high')}, {'if': [('Food Quality', 'good'), ('Service', 'good')], 'then': ('Tip', 'high')}, {'if': [('Food Quality', 'good'), ('Service', 'excellent')], 'then': ('Tip', 'high')}]
    input_vars = [v for v in fis_vars if v["role"] == "Input"]
    print("=" * 60)
    print("TYPE-1 FUZZY INFERENCE SYSTEM")
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
    print("Running Type-1 Fuzzy Inference...")
    print()
    # Run inference
    result = fuzzy_infer(inputs)
    print("Results:")
    print("-" * 40)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("=" * 60)
    print("Inference Complete!")
    print("=" * 60)