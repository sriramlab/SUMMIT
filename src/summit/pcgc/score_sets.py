"""Complete real confidence sets for a low-degree score inequality."""
import numpy as np


def polynomial_nonpositive_set(coefficients,*,center=0.,scale=1.):
    """Return all intervals where p(t)<=0, mapped by center+scale*t.

    Coefficients are in increasing degree order. Endpoints at infinity use
    None in the JSON representation. Tangencies are retained as singleton
    intervals. Leading terms below roundoff are removed before root finding.
    """
    coefficients = np.asarray(coefficients,dtype=float)
    if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all() or not np.isfinite(center) or not np.isfinite(scale) or scale<=0:
        raise ValueError("invalid score polynomial or coordinate scale")
    magnitude = np.max(np.abs(coefficients))
    if magnitude == 0: return [[None,None]]
    coefficients = coefficients/magnitude
    while len(coefficients)>1 and abs(coefficients[-1])<128*np.finfo(float).eps:
        coefficients = coefficients[:-1]
    if len(coefficients)==1: return [[None,None]] if coefficients[0]<=0 else []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    real = sorted(float(r.real) for r in roots if abs(r.imag)<=1e-7*max(1.,abs(r.real)))
    unique = []
    for root in real:
        if not unique or abs(root-unique[-1])>1e-8*max(1.,abs(root)):
            unique.append(root)
    edges = [-np.inf,*unique,np.inf]
    intervals = []
    degree = len(coefficients)-1
    for lo,hi in zip(edges[:-1],edges[1:]):
        if np.isneginf(lo): sign = coefficients[-1]*(-1)**degree
        elif np.isposinf(hi): sign = coefficients[-1]
        else: sign = np.polynomial.polynomial.polyval((lo+hi)/2,coefficients)
        if sign<=0: intervals.append([lo,hi])
    for root in unique:
        if not any(lo<=root<=hi for lo,hi in intervals): intervals.append([root,root])
    merged = []
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]: merged[-1][1] = max(merged[-1][1],hi)
        else: merged.append([lo,hi])
    return [[None if np.isinf(lo) else float(center+scale*lo),
             None if np.isinf(hi) else float(center+scale*hi)] for lo,hi in merged]
