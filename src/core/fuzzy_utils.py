import numpy as np

def safe_div(n, d):
    # Handle both scalar and array division safely
    if isinstance(n, np.ndarray) or isinstance(d, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(n, d)
            result[~np.isfinite(result)] = 0  # Replace inf/NaN with 0
        return result
    else:
        return n / d if d != 0 else 0 