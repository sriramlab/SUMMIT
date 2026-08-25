"""Production fixed-effect basis construction without dense projectors."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def thin_rank_revealing_fixed_effect_basis(
    fixed_effects: object, *, rtol: float | None = None
) -> np.ndarray:
    """Return an orthonormal basis for a fixed-effect design's column span.

    The column-pivoted economic QR factorization requires ``O(Np)`` storage
    for an ``N``-sample, ``p``-column design.  In particular, this production
    helper never constructs the dense residual projector ``I - QQ.T``.
    """
    design = np.asarray(fixed_effects, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError(
            f"fixed_effects must be two-dimensional; got {design.shape}"
        )
    n_samples, n_columns = design.shape
    if n_samples < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(design)):
        raise ValueError("fixed_effects contains non-finite values")
    if n_columns == 0:
        return np.empty((n_samples, 0), dtype=np.float64, order="F")
    if rtol is not None:
        rtol = float(rtol)
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")

    q, upper, _ = linalg.qr(
        design,
        mode="economic",
        pivoting=True,
        check_finite=False,
        overwrite_a=False,
    )
    diagonal = np.abs(np.diag(upper))
    scale = float(diagonal[0]) if diagonal.size else 0.0
    tolerance = (
        (max(design.shape) * np.finfo(np.float64).eps if rtol is None else rtol)
        * scale
    )
    rank = int(np.count_nonzero(diagonal > tolerance))
    return np.asfortranarray(q[:, :rank], dtype=np.float64)
