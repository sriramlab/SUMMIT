"""Bounded fixed-design split-ridge diagnostics, not production uncertainty.

Inputs are already in the *native training* genotype scale, with missing calls
imputed using that scale. Known means/variances below are validation inputs.
No outcome-dependent choice of penalty is covered by this calculation.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def ridge_transfer(train_x, test_x, train_c, test_c, penalty):
    """Exact held-out prediction matrix, including unpenalized fixed effects."""
    x, z, c, d = map(np.asarray, (train_x, test_x, train_c, test_c))
    n, m = x.shape
    if n + len(z) > 2048 or m > 4096:
        raise ValueError("dense split reference is bounded at N=2048, M=4096")
    if (
        c.ndim != 2
        or d.shape != (len(z), c.shape[1])
        or len(c) != n
        or z.shape[1] != m
        or penalty <= 0
        or not np.isfinite(penalty)
        or not all(np.isfinite(v).all() for v in (x, z, c, d))
    ):
        raise ValueError("invalid split ridge inputs")
    ci = np.linalg.pinv(c)
    if not np.allclose(d @ ci @ c, d, rtol=1e-9, atol=1e-10):
        raise ValueError("held-out fixed effects not identified in training")
    px = x - c @ (ci @ x)
    dz = z - d @ (ci @ x)
    # Primal/dual equality is tested against an independently formed augmented
    # penalized least-squares problem. Neither path forms a large pair matrix.
    if m <= n:
        gram = px.T @ px + penalty * np.eye(m)
        weights = cho_solve(cho_factor(gram), px.T)
        return d @ ci + dz @ weights
    gram = px @ px.T + penalty * np.eye(n)
    return d @ ci + (dz @ px.T) @ cho_solve(cho_factor(gram), np.eye(n))


def fixed_split_contrast(train_f, test_f, test_c, outcome_transfer, feature_transfer):
    """Fixed-operator coefficient and analytic response, with no noise inputs."""
    ft, f, c, ly, lf = map(
        np.asarray, (train_f, test_f, test_c, outcome_transfer, feature_transfer)
    )
    if (
        ft.ndim != 1
        or f.ndim != 1
        or len(f) + len(ft) > 2048
        or c.ndim != 2
        or len(c) != len(f)
        or ly.shape != (len(f), len(ft))
        or lf.shape != ly.shape
        or not all(np.isfinite(v).all() for v in (ft, f, c, ly, lf))
    ):
        raise ValueError("invalid bounded split contrast inputs")
    residual = f - lf @ ft
    residual -= c @ (np.linalg.pinv(c) @ residual)
    h = residual @ residual
    if h <= 1e-16 * (f @ f):
        raise ValueError("feature absorbed by the split nuisance fit")
    a = residual / h
    b = ly.T @ a
    return dict(
        test_contrast=a,
        training_contrast=b,
        interaction_response=float(a @ f - b @ ft),
        residual_feature_information=float(h),
    )


def split_scalar_reference(
    train_f,
    test_f,
    test_c,
    outcome_transfer,
    feature_transfer,
    *,
    train_mean,
    test_mean,
    train_variance,
    test_variance,
):
    """Return the actual linear contrast, its expectation, and noise variance.

    beta_hat = a' Y_test - b' Y_train, b=L_Y' a.  The contrast a depends
    only on genotype features, fixed effects and the frozen feature penalty.
    ``interaction_response`` need not be one when L_Y differs from L_F.
    """
    ft, f, c, ly, lf, mt, mu, vt, v = map(
        np.asarray,
        (
            train_f,
            test_f,
            test_c,
            outcome_transfer,
            feature_transfer,
            train_mean,
            test_mean,
            train_variance,
            test_variance,
        ),
    )
    if (
        ft.ndim != 1
        or f.ndim != 1
        or len(f) + len(ft) > 2048
        or ly.shape != (len(f), len(ft))
        or lf.shape != ly.shape
        or c.ndim != 2
        or len(c) != len(f)
        or mt.shape != ft.shape
        or mu.shape != f.shape
        or vt.shape != ft.shape
        or v.shape != f.shape
        or np.any(vt <= 0)
        or np.any(v <= 0)
        or not all(np.isfinite(z).all() for z in (ft, f, c, ly, lf, mt, mu, vt, v))
    ):
        raise ValueError("invalid bounded scalar split diagnostic")
    contrast = fixed_split_contrast(ft, f, c, ly, lf)
    a, b = contrast["test_contrast"], contrast["training_contrast"]
    test_noise = float((a * a) @ v)
    train_noise = float((b * b) @ vt)
    return dict(
        **contrast,
        expectation=float(a @ mu - b @ mt),
        test_noise_variance=test_noise,
        training_noise_variance=train_noise,
        total_noise_variance=test_noise + train_noise,
    )


def crossfit_scalar_reference(features, fixed, test_folds, transfers, variance):
    """Combine fixed-tuning folds as ONE linear contrast, retaining overlap.

    Every transfer predicts its held-out fold from the complement. The same
    operator is used for Y and F in this bounded diagnostic. Training sets of
    different folds overlap: their estimating equations are not independent.
    """
    f, c, v = np.asarray(features), np.asarray(fixed), np.asarray(variance)
    n = len(f)
    if n > 2048 or f.shape != (n,) or v.shape != (n,) or np.any(v <= 0):
        raise ValueError("invalid bounded cross-fit reference")
    folds = [np.asarray(i, dtype=int) for i in test_folds]
    if len(folds) != len(transfers) or sorted(np.concatenate(folds).tolist()) != list(
        range(n)
    ):
        raise ValueError("cross-fitting folds must partition the full sample once")
    numerator = np.zeros(n)
    h = 0.0
    naive = 0.0
    for ids, L in zip(folds, transfers):
        train = np.setdiff1d(np.arange(n), ids)
        if np.shape(L) != (len(ids), len(train)):
            raise ValueError("transfer axes do not match the fold complement")
        r = f[ids] - L @ f[train]
        r -= c[ids] @ (np.linalg.pinv(c[ids]) @ r)
        term = np.zeros(n)
        term[ids] = r
        term[train] = -L.T @ r
        numerator += term
        h += r @ r
        naive += (term * term) @ v
    if h <= 0:
        raise ValueError("unidentified cross-fit feature")
    influence = numerator / h
    return dict(
        influence=influence,
        information=float(h),
        variance=float((influence * influence) @ v),
        variance_if_folds_incorrectly_independent=float(naive / h**2),
        interaction_response=float(influence @ f),
    )


def estimated_contrast_noise(
    contrast,
    outcomes,
    mean_design,
    variance_basis=None,
    *,
    nonnegative_components=False,
):
    """Bounded estimated-noise reference with an explicit finite mean model.

    The complete mean design must include the interaction, and describe BOTH
    splits (or every fold). For homoskedastic Gaussian errors, a contrast in
    its column span is independent of RSS and can be studentized with the
    returned residual degrees of freedom. This does not remove ridge mean bias.

    With a supplied low-dimensional variance surface V, fit residual squares
    to diag(P diag(V_j) P), not V_j itself. This is unbiased for its coefficients
    under the stated mean/variance model; normal inference is only asymptotic.
    Negative fitted variances are reported, never clipped. An explicitly
    nonnegative component model instead uses constrained least squares for
    the surface coefficients; it loses finite-sample unbiasedness. No true noise or
    generating-mean input is accepted. This bounded diagnostic is not a p>N
    uncertainty correction and does not cover outcome-selected operators.
    """
    d, y, design = map(
        lambda a: np.asarray(a, float), (contrast, outcomes, mean_design)
    )
    if d.ndim == 1:
        d = d[:, None]
    if y.ndim == 1:
        y = y[:, None]
    n = len(d)
    if (
        n > 2048
        or d.ndim != 2
        or y.ndim != 2
        or design.ndim != 2
        or design.shape[1] == 0
        or len(y) != n
        or len(design) != n
        or not all(np.isfinite(v).all() for v in (d, y, design))
    ):
        raise ValueError("invalid bounded estimated-noise inputs")
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > max(design.shape) * np.finfo(float).eps * singular[0]))
    u = u[:, :rank]
    df = n - rank
    if df < 10:
        raise ValueError(
            "insufficient residual degrees of freedom for noise estimation"
        )
    residual = y - u @ (u.T @ y)
    if variance_basis is None:
        if np.linalg.norm(d - u @ (u.T @ d)) > 1e-8 * np.linalg.norm(d):
            raise ValueError(
                "exact studentization requires contrasts in the fitted mean span "
                f"(relative departure {np.linalg.norm(d-u@(u.T@d))/np.linalg.norm(d):.6g})"
            )
        scale = np.sum(residual**2, axis=0) / df
        return dict(
            covariance=scale[:, None, None] * (d.T @ d)[None, :, :],
            residual_df=df,
            variance_coefficients=scale[None, :],
            positive_variance=np.all(scale > 0),
            inference="Student t for Gaussian common noise and a correct declared finite mean; ridge mean bias remains separate",
        )
    v = np.asarray(variance_basis, float)
    if v.ndim != 2 or len(v) != n or not np.isfinite(v).all() or v.shape[1] > 16:
        raise ValueError("invalid prespecified variance surface")
    leverage = np.sum(u * u, axis=1)
    transformed = np.empty_like(v)
    for j in range(v.shape[1]):
        gram = u.T @ (v[:, j, None] * u)
        transformed[:, j] = v[:, j] * (1 - 2 * leverage) + np.sum(
            (u @ gram) * u, axis=1
        )
    if np.linalg.matrix_rank(transformed) != v.shape[1]:
        raise ValueError("variance surface not identified after fitting the mean")
    if nonnegative_components:
        from scipy.optimize import nnls

        if np.any(v < 0):
            raise ValueError(
                "nonnegative variance components require nonnegative surfaces"
            )
        coefficients = np.column_stack(
            [nnls(transformed, r * r)[0] for r in residual.T]
        )
    else:
        coefficients = np.linalg.lstsq(transformed, residual**2, rcond=None)[0]
    fitted = v @ coefficients
    covariance = np.einsum("ni,nt,nj->tij", d, fitted, d)
    return dict(
        covariance=covariance,
        residual_df=df,
        variance_coefficients=coefficients,
        variance_fit="nonnegative component least squares"
        if nonnegative_components
        else "unconstrained unbiased surface moments",
        positive_variance=np.all(fitted > 0, axis=0),
        inference="asymptotic estimated variance-surface moments under a correct finite mean and fixed operators; ridge mean bias remains separate",
    )
