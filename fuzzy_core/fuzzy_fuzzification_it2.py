import numpy as np

def fuzzify_it2(val, sets):
    """
    Compute lower and upper membership values for interval type-2 fuzzy sets.
    Each set should define 'lower_params' and 'upper_params' (comma-separated strings).
    Returns: dict of set name -> (lower, upper) membership.
    """
    memberships = {}
    for s in sets:
        try:
            lparams = [float(p.strip()) for p in s['lower_params'].split(",")]
            uparams = [float(p.strip()) for p in s['upper_params'].split(",")]
            if s['type'] == "Triangular" and len(lparams) == 3 and len(uparams) == 3:
                aL, bL, cL = lparams
                aU, bU, cU = uparams
                # Lower MF
                if val <= aL or val >= cL:
                    muL = 0.0
                elif val == bL:
                    muL = 1.0
                elif val < bL:
                    muL = (val - aL) / (bL - aL) if bL > aL else 1.0
                else:
                    muL = (cL - val) / (cL - bL) if cL > bL else 1.0
                # Upper MF
                if val <= aU or val >= cU:
                    muU = 0.0
                elif val == bU:
                    muU = 1.0
                elif val < bU:
                    muU = (val - aU) / (bU - aU) if bU > aU else 1.0
                else:
                    muU = (cU - val) / (cU - bU) if cU > bU else 1.0
            elif s['type'] == "Trapezoidal" and len(lparams) == 4 and len(uparams) == 4:
                aL, bL, cL, dL = lparams
                aU, bU, cU, dU = uparams
                # Lower MF
                if val <= aL or val >= dL:
                    muL = 0.0
                elif bL <= val <= cL:
                    muL = 1.0
                elif val < bL:
                    muL = (val - aL) / (bL - aL) if bL > aL else 1.0
                else:
                    muL = (dL - val) / (dL - cL) if dL > cL else 1.0
                # Upper MF
                if val <= aU or val >= dU:
                    muU = 0.0
                elif bU <= val <= cU:
                    muU = 1.0
                elif val < bU:
                    muU = (val - aU) / (bU - aU) if bU > aU else 1.0
                else:
                    muU = (dU - val) / (dU - cU) if dU > cU else 1.0
            elif s['type'] == "Gaussian" and len(lparams) == 2 and len(uparams) == 2:
                meanL, sigmaL = lparams
                meanU, sigmaU = uparams
                muL = np.exp(-0.5 * ((val - meanL) / sigmaL) ** 2) if sigmaL > 0 else float(val == meanL)
                muU = np.exp(-0.5 * ((val - meanU) / sigmaU) ** 2) if sigmaU > 0 else float(val == meanU)
            else:
                muL, muU = 0.0, 0.0
            muL = max(0.0, min(1.0, muL))
            muU = max(0.0, min(1.0, muU))
            memberships[s['name']] = (muL, muU)
        except Exception:
            memberships[s['name']] = (0.0, 0.0)
    return memberships 