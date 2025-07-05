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

def fuzzify_non_singleton(val, std_dev, sets):
    """
    Non-singleton fuzzification that accounts for input uncertainty.
    
    Args:
        val: Input value
        std_dev: Standard deviation representing input uncertainty
        sets: List of fuzzy sets
    
    Returns:
        Dictionary of membership values for each set
    """
    memberships = {}
    
    # If std_dev is 0 or very small, use singleton fuzzification
    if std_dev <= 1e-6:
        return fuzzify(val, sets)
    
    for s in sets:
        try:
            params = [float(p.strip()) for p in s['params'].split(",")]
            
            if s['type'] == "Triangular" and len(params) == 3:
                a, b, c = params
                # For triangular sets, we need to integrate over the uncertainty
                # This is approximated by sampling points around the input value
                num_samples = 100
                x_samples = np.linspace(val - 3*std_dev, val + 3*std_dev, num_samples)
                weights = np.exp(-0.5 * ((x_samples - val) / std_dev) ** 2)
                weights = weights / np.sum(weights)  # Normalise
                
                mu_samples = []
                for x in x_samples:
                    if x <= a or x >= c:
                        mu = 0.0
                    elif x == b:
                        mu = 1.0
                    elif x < b:
                        mu = (x - a) / (b - a) if b > a else 1.0
                    else:
                        mu = (c - x) / (c - b) if c > b else 1.0
                    mu_samples.append(max(0.0, min(1.0, mu)))
                
                mu = np.sum(np.array(mu_samples) * weights)
                
            elif s['type'] == "Trapezoidal" and len(params) == 4:
                a, b, c, d = params
                # Similar integration for trapezoidal sets
                num_samples = 100
                x_samples = np.linspace(val - 3*std_dev, val + 3*std_dev, num_samples)
                weights = np.exp(-0.5 * ((x_samples - val) / std_dev) ** 2)
                weights = weights / np.sum(weights)
                
                mu_samples = []
                for x in x_samples:
                    if x <= a or x >= d:
                        mu = 0.0
                    elif b <= x <= c:
                        mu = 1.0
                    elif x < b:
                        mu = (x - a) / (b - a) if b > a else 1.0
                    else:
                        mu = (d - x) / (d - c) if d > c else 1.0
                    mu_samples.append(max(0.0, min(1.0, mu)))
                
                mu = np.sum(np.array(mu_samples) * weights)
                
            elif s['type'] == "Gaussian" and len(params) == 2:
                mean, sigma = params
                if sigma <= 0:
                    mu = 1.0 if val == mean else 0.0
                else:
                    # For Gaussian sets, the non-singleton membership can be computed analytically
                    # as the convolution of two Gaussians
                    combined_sigma = np.sqrt(sigma**2 + std_dev**2)
                    mu = np.exp(-0.5 * ((val - mean) / combined_sigma) ** 2)
            else:
                mu = 0.0
                
            mu = max(0.0, min(1.0, mu))
            memberships[s['name']] = mu
            
        except Exception:
            memberships[s['name']] = 0.0
    
    return memberships 