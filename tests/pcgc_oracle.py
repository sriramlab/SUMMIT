"""Independent tiny PCGC oracle: explicit sample pairs and liability integrals."""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def pair_moment(k1, k2, sampling_ratio, rho):
    t1, t2 = norm.isf([k1, k2])
    joint = quad(lambda x: norm.pdf(x) * norm.sf((t2-rho*x)/np.sqrt(1-rho*rho)),
                 t1, np.inf, epsabs=1e-12)[0]
    probs = np.array([[1-k1-k2+joint, k2-joint], [k1-joint, joint]])
    probs *= np.outer([sampling_ratio, 1], [sampling_ratio, 1])
    probs /= probs.sum()
    p1, p2 = [k/(k+sampling_ratio*(1-k)) for k in (k1, k2)]
    z1 = (np.arange(2)-p1)/np.sqrt(p1*(1-p1))
    z2 = (np.arange(2)-p2)/np.sqrt(p2*(1-p2))
    return np.sum(probs * np.outer(z1, z2))


def risks(y, K, k):
    P = np.mean(y)
    a = K*(1-P)/(P*(1-K))
    p = k/(k+a*(1-k))
    z = (y-p)/np.sqrt(p*(1-p))
    prefix = norm.pdf(norm.isf(k))/np.sqrt(p*(1-p))/(k+a*(1-k))
    return z, prefix*(1-p+a*p)


def pair_equations(X, annotations, z, d, *, inverse=False):
    kernels = np.array([(X * a) @ X.T / a.sum() for a in annotations.T])
    H = np.zeros((len(kernels), len(kernels)))
    b = np.zeros(len(kernels))
    for i in range(len(X)):
        for j in range(i):
            row = kernels[:, i, j]
            response = z[i]*z[j]
            if inverse:
                response /= d[i]*d[j]
            else:
                row = row * d[i]*d[j]
            H += 2*np.outer(row, row)
            b += 2*row*response
    return H, b
