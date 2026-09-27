from __future__ import annotations

import numpy as np


def resolve_cov_rank(
    *,
    explicit=None,
    explicit_source=None,
    sumstats_value=None,
    sumstats_source=None,
):
    """
    Resolve user-facing cov_rank = number of NON-intercept covariate df.

    Precedence:
        explicit > sumstats > 0
    """
    if explicit is not None:
        v = int(explicit)
        if v < 0:
            raise ValueError(f"cov_rank must be >= 0; got {v}")
        return v, (explicit_source or "explicit")

    if sumstats_value is not None:
        v = int(sumstats_value)
        if v < 0:
            raise ValueError(f"sumstats cov_rank must be >= 0; got {v}")
        return v, (sumstats_source or "file")

    return 0, "default0"


def effective_n_scale(nsamp, cov_rank):
    """
    Reference sample-size scale for the moment equations:
        n_star = N_gwas - cov_rank - 1
    """
    n = float(nsamp) - float(cov_rank) - 1.0
    if not (np.isfinite(n) and n > 0.0):
        raise ValueError(f"Invalid effective scale: nsamp={nsamp}, cov_rank={cov_rank}")
    return n


def gwas_resid_df(n_obs, cov_rank):
    """
    Per-SNP OLS residual df in the GWAS:
        nu_j = n_j - cov_rank - 2
    """
    return np.asarray(n_obs, dtype=np.float64) - float(cov_rank) - 2.0


def derived_wald_z(beta, se, n_obs, n_scale):
    """
    Derived z used only for QC / chi^2 filtering.
    """
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    n_obs = np.asarray(n_obs, dtype=np.float64)
    N = float(n_scale)

    out = np.full(beta.shape, np.nan, dtype=np.float64)
    if not (np.isfinite(N) and N > 0.0):
        return out

    good = (
        np.isfinite(beta) &
        np.isfinite(se) & (se > 0.0) &
        np.isfinite(n_obs) & (n_obs > 0.0)
    )
    out[good] = (beta[good] / se[good]) * np.sqrt(n_obs[good] / N)
    return out


def exact_score_z_from_arrays(beta, se, n_obs, nsamp, cov_rank):
    """
    Score statistic on the reference sample-size scale:
        z*_j = sqrt(n_star) * beta_j / sqrt(beta_j^2 + nu_j * se_j^2)

    where
        n_star = nsamp - cov_rank - 1
        nu_j   = n_obs_j - cov_rank - 2

    Its null variance is n_star / (nu_j + 1), not one when N varies.
    Use h2_moment_from_arrays for the univariate estimating response.
    """
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    n_obs = np.asarray(n_obs, dtype=np.float64)

    q = int(cov_rank)
    n_star = effective_n_scale(nsamp, q)
    nu = gwas_resid_df(n_obs, q)
    den = beta * beta + nu * se * se

    out = np.full(beta.shape, np.nan, dtype=np.float64)
    good = (
        np.isfinite(beta) &
        np.isfinite(se) & (se > 0.0) &
        np.isfinite(n_obs) & (n_obs > 0.0) &
        np.isfinite(nu) & (nu > 0.0) &
        np.isfinite(den) & (den > 0.0)
    )
    out[good] = np.sqrt(n_star) * beta[good] / np.sqrt(den[good])
    return out


def score_noise_scale(n_obs, nsamp, cov_rank):
    """Null variance of a reference-scaled OLS score, N* / n*_j."""
    n_star = effective_n_scale(nsamp, cov_rank)
    local = gwas_resid_df(n_obs, cov_rank) + 1.0
    if np.all(local == n_star):
        return 1.0
    out = np.full(local.shape, np.nan, dtype=np.float64)
    np.divide(n_star, local, out=out, where=np.isfinite(local) & (local > 1.0))
    return out


def summary_noise_scale(matched, info, *, trait=None):
    """Noise scale for beta/SE summaries; supplied moments keep their own scale."""
    if not info or not str(info.get("mode", "")).startswith("beta_se_exact"):
        return 1.0
    key = "cov_rank" if trait is None else f"trait{trait}_cov_rank"
    return score_noise_scale(matched.n, matched.nsamp, int(info.get(key, 0)))


def h2_moment_from_arrays(beta, se, n_obs, nsamp, cov_rank=0):
    """Return 1 + N* [r_j^2 - 1/n*_j], preserving the signal scale.

    The subtraction centers the response at one under the OLS null even
    when SNP sample sizes differ. The result can be negative; it must not
    be clipped or represented as the square of another score statistic.
    """
    z = exact_score_z_from_arrays(beta, se, n_obs, nsamp, cov_rank)
    y = z * z
    y -= score_noise_scale(n_obs, nsamp, cov_rank) - 1.0
    return y


def build_h2_summary_moment(
    matched,
    *,
    cov_rank=None,
    cov_rank_source=None,
):
    """Build the h2 response using the intercept-only GWAS convention."""
    resolved_cov_rank = 0
    source = "forced0_no_covrank_h2"

    y = h2_moment_from_arrays(
        beta=np.asarray(matched.beta, dtype=np.float64),
        se=np.asarray(matched.se, dtype=np.float64),
        n_obs=np.asarray(matched.n, dtype=np.float64),
        nsamp=float(matched.nsamp),
        cov_rank=resolved_cov_rank,
    )

    y[~np.isfinite(y)] = np.nan

    return y, {
        "mode": "beta_se_exact",
        "cov_rank": int(resolved_cov_rank),
        "cov_rank_source": source,
        "n_scale": float(effective_n_scale(matched.nsamp, resolved_cov_rank)),
        "n_nonfinite": int(np.sum(~np.isfinite(y))),
    }


def build_rg_summary_moment(
    matched1,
    matched2,
    *,
    cov_rank1=None,
    cov_rank_source1=None,
    cov_rank2=None,
    cov_rank_source2=None,
):
    resolved_cov_rank1, source1 = resolve_cov_rank(
        explicit=cov_rank1,
        explicit_source=cov_rank_source1,
        sumstats_value=getattr(matched1, "cov_rank", None),
        sumstats_source=getattr(matched1, "cov_rank_source", None),
    )
    resolved_cov_rank2, source2 = resolve_cov_rank(
        explicit=cov_rank2,
        explicit_source=cov_rank_source2,
        sumstats_value=getattr(matched2, "cov_rank", None),
        sumstats_source=getattr(matched2, "cov_rank_source", None),
    )

    z1_star = exact_score_z_from_arrays(
        beta=np.asarray(matched1.beta, dtype=np.float64),
        se=np.asarray(matched1.se, dtype=np.float64),
        n_obs=np.asarray(matched1.n, dtype=np.float64),
        nsamp=float(matched1.nsamp),
        cov_rank=resolved_cov_rank1,
    )
    z2_star = exact_score_z_from_arrays(
        beta=np.asarray(matched2.beta, dtype=np.float64),
        se=np.asarray(matched2.se, dtype=np.float64),
        n_obs=np.asarray(matched2.n, dtype=np.float64),
        nsamp=float(matched2.nsamp),
        cov_rank=resolved_cov_rank2,
    )

    y = z1_star * z2_star
    y[~np.isfinite(y)] = np.nan

    return y, {
        "mode": "beta_se_exact",
        "trait1_cov_rank": int(resolved_cov_rank1),
        "trait1_cov_rank_source": source1,
        "trait1_n_scale": float(effective_n_scale(matched1.nsamp, resolved_cov_rank1)),
        "trait2_cov_rank": int(resolved_cov_rank2),
        "trait2_cov_rank_source": source2,
        "trait2_n_scale": float(effective_n_scale(matched2.nsamp, resolved_cov_rank2)),
        "n_nonfinite": int(np.sum(~np.isfinite(y))),
    }