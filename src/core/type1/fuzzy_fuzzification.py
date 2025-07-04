from ..fuzzy_utils import safe_div
import numpy as np

def fuzzify(val, sets):
    memberships = {}
    for s in sets:
        try:
            params = [float(p.strip()) for p in s['params'].split(",")]
            if s['type'] == "Triangular" and len(params) == 3:
                a, b, c = params
                if val <= a or val >= c:
                    mu = 0.0
                elif val == b:
                    mu = 1.0
                elif val < b:
                    mu = (val - a) / (b - a) if b > a else 1.0
                else:
                    mu = (c - val) / (c - b) if c > b else 1.0
            elif s['type'] == "Trapezoidal" and len(params) == 4:
                a, b, c, d = params
                if val <= a or val >= d:
                    mu = 0.0
                elif b <= val <= c:
                    mu = 1.0
                elif val < b:
                    mu = (val - a) / (b - a) if b > a else 1.0
                else:
                    mu = (d - val) / (d - c) if d > c else 1.0
            elif s['type'] == "Gaussian" and len(params) == 2:
                mean, sigma = params
                if sigma <= 0:
                    mu = 1.0 if val == mean else 0.0
                else:
                    mu = np.exp(-0.5 * ((val - mean) / sigma) ** 2)
            else:
                mu = 0.0
            mu = max(0.0, min(1.0, mu))
            memberships[s['name']] = mu
        except Exception:
            memberships[s['name']] = 0.0
    return memberships 