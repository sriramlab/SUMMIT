"""Small-data exact references for qualification; never a production decoder."""
import numpy as np

from summit.context.spec import array_sha256
from summit.sumstats.binary import finite_array
from .moments import BinaryMoments, annotations_array, method_vectors, external_ld_moments


def exact_moments(genotype, annotations, risk, method="pcgc", *, sensitivity=None, reference_cache=None):
    x = finite_array("genotype", genotype, 2)
    a = annotations_array(annotations, x.shape[1])
    if x.shape[0] != risk.n_samples:
        raise ValueError("genotype and risk sample axes differ")
    if method == "pcgc-basis":
        weight = finite_array("sensitivity", sensitivity, 1)
        if weight.shape != (len(x),) or np.any(weight <= 0):
            raise ValueError("basis sensitivity must be positive on the sample axis")
        response = weight * risk.z
    else:
        weight, response = method_vectors(risk, method)
    # Reuse identical reference multipliers across paired research methods.
    # The cache belongs to this one immutable simulation dataset, never a file
    # reader or a cross-study persistent cache.
    factor = float(weight[0]) if np.all(weight == weight[0]) else 1.
    normalized = weight / factor
    key = (array_sha256(normalized), array_sha256(x), array_sha256(a)) if reference_cache is not None else None
    if reference_cache is not None and key in reference_cache:
        ld, sp = reference_cache[key]
    else:
        f = normalized[:, None] * x
        cross = f.T @ f
        squares = f*f
        diagonal_numerator = squares @ a
        ld = ((cross * cross) @ a - squares.T @ diagonal_numerator) / len(x)**2
        diagonal = diagonal_numerator / a.sum(axis=0)
        sp = diagonal.T @ diagonal
        if reference_cache is not None:
            reference_cache[key] = ld, sp
    ld, sp = ld * factor**4, sp * factor**4
    rhs_rows = (x.T @ response)**2 - (x*x).T @ (response*response)
    return BinaryMoments(a, ld, sp, rhs_rows, len(x), method, risk.covariate_variance)


def exact_external_ld(genotype, annotations, *, block_size=256, matmul=np.matmul):
    """Unbiased row-wise estimate of population squared cross moments.

    Requires independently sampled reference people on a fixed population
    genotype scale. The same-person fourth moment is subtracted *per SNP*.
    This does not correct reference/study ascertainment or risk dependence.
    """
    x = finite_array("reference genotype", genotype, 2)
    a = annotations_array(annotations, x.shape[1])
    if len(x) < 2:
        raise ValueError("reference requires at least two people")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("exact reference block size must be a positive integer")
    if len(x) < x.shape[1]:
        # Small-reference oracle: choose the smaller square matrix. Computing
        # X_j.T (X A X.T - diag) X_j is the identical U statistic, with O(R^2)
        # instead of O(M^2) scratch. This is NOT the production probe estimator.
        ld = np.empty_like(a)
        for b, weights in enumerate(a.T):
            if np.all(weights == weights[0]):
                kernel = matmul(x, x.T)*weights[0]
            else:
                kernel = matmul(x*weights, x.T)
            np.fill_diagonal(kernel, 0.)
            for start in range(0, x.shape[1], block_size):
                stop = min(start+block_size, x.shape[1])
                target = x[:, start:stop]
                ld[start:stop, b] = np.einsum('ij,ij->j', target, matmul(kernel, target))
        return ld/(len(x)*(len(x)-1))
    cross = x.T @ x
    squares = x*x
    return ((cross*cross) @ a - squares.T @ (squares @ a)) / (len(x)*(len(x)-1))


def exact_pair(genotype, annotations, left, right, left_rows, right_rows, *, method="pcgc"):
    """Dense small-data rectangular oracle; no assumptions about selection validity."""
    from .cross import PairMoments, response_vectors
    x = finite_array("master genotype", genotype, 2)
    a = annotations_array(annotations, x.shape[1])
    i, j = np.asarray(left_rows), np.asarray(right_rows)
    wl, vl = response_vectors(left, method)
    wr, vr = response_vectors(right, method)
    f1, f2 = x[i]*wl[:, None], x[j]*wr[:, None]
    G1, G2 = f1.T @ f1, f2.T @ f2
    s1, s2 = x[i].T @ vl, x[j].T @ vr
    within = []
    for rows, risk, f, G, v, s in ((i, left, f1, G1, vl, s1), (j, right, f2, G2, vr, s2)):
        squares = f*f
        numerator = squares @ a
        diag = numerator/a.sum(axis=0)
        rhs = s*s-(x[rows]**2).T @ (v*v)
        within.append(BinaryMoments(a, ((G*G) @ a-squares.T @ numerator)/len(rows)**2, diag.T @ diag, rhs,
                                     len(rows), method, risk.covariate_variance))
    overlap, il, jr = np.intersect1d(i, j, assume_unique=True, return_indices=True)
    diag = ((x[overlap]**2) @ a/a.sum(axis=0))*(wl[il]*wr[jr])[:, None]
    rhs = s1*s2-(x[overlap]**2).T @ (vl[il]*vr[jr])
    squares = (x[overlap]**2)*(wl[il]*wr[jr])[:, None]
    cross = BinaryMoments(a, ((G1*G2) @ a-squares.T @ (squares @ a))/len(x)**2, diag.T @ diag, rhs, len(x), method)
    return PairMoments(*within, cross, np.sqrt((1+left.covariate_variance)*(1+right.covariate_variance)), len(overlap))
