"""Conditional residual refits on the actual whole-marker public scalar design.

The finite mean is exactly in C; this isolates HC3 inference, not adequacy of
an observed trait's mean model. All nuisance coefficients and HC3 denominators
are re-estimated for every draw, algebraically batched with a fixed QR basis.
"""
import argparse, json, time, resource
from pathlib import Path
import numpy as np
from scipy.stats import norm
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis as basis
from summit.epistasis.features import load_feature_reference
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable
from summit.prediction.genotype import FileGenotypeSource
from scripts.epistasis.continuation_design import decision
from scripts.epistasis.validate import interval


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    with np.load(a.reference, allow_pickle=False) as z:
        identity = json.loads(str(z["manifest"]))["metadata"]["compatibility_id"]
    ref = load_feature_reference(a.reference, compatibility_id=identity)
    f = ref.features
    c = ref.fixed_effects
    n = len(f)
    if f.shape[1] != 1:
        raise ValueError("scalar reference required")
    with FileGenotypeSource(a.genotypes, genome_build="GRCh37") as source:
        if n != len(source.samples):
            raise ValueError("this validator requires the complete source sample axis")
        j = source.variants.ids.index("12:66358347")
        source.prepare(np.arange(n), 1, 1)
        g = source.read(np.array([j]))[:, 0].astype(float)
        g[g == -127] = np.nan
        mu = np.nanmean(g)
        x = np.nan_to_num(g - mu) / np.sqrt(mu * (1 - mu / 2))
    u = basis(c)
    r = f - u @ (u.T @ f)
    r -= u @ (u.T @ r)
    h = float((r.T @ r).item())
    partial = r[:, 0] ** 2 / h
    hc = (u * u).sum(1)
    lev = hc + partial
    tol = 64 * np.finfo(float).eps * max(c.shape + f.shape)
    sat = (abs(1 - hc) <= tol) & (partial <= tol**2)
    r[sat] = 0
    h = float((r.T @ r).item())
    infl = r[:, 0] / h
    if np.any(lev[~sat] >= 1 - 1e-8):
        raise ValueError("essential or unresolved unit leverage")
    den = 1 - lev
    den[sat] = 1
    weight = infl**2 / den**2
    rng = np.random.default_rng(165728)
    rows = []
    first_error = 0.0
    for kind in ("gaussian", "heterogeneous_gaussian", "heterogeneous_t5"):
        b = []
        se = []
        for batch in range(20):
            noise = (
                rng.standard_t(5, size=(n, 100)) * np.sqrt(3 / 5)
                if kind.endswith("t5")
                else rng.normal(size=(n, 100))
            )
            if kind != "gaussian":
                noise *= np.sqrt(0.4 + 0.6 * x[:, None] ** 2)
            noise[sat] = 0
            py = noise - u @ (u.T @ noise)
            coef = infl @ py
            e = py - r * coef[None, :]
            e[sat] = 0
            error = np.sqrt(weight @ (e * e))
            if batch == 0:
                saved = prepare_robust_scores(
                    f,
                    noise[:, :2],
                    c,
                    feature_names=("whole_direction",),
                    trait_names=("a", "b"),
                    metadata={},
                )
                for t in range(2):
                    fit = robust_score_tests(saved, trait=t)
                    first_error = max(
                        first_error,
                        abs(fit["beta"][0] - coef[t]),
                        abs(fit["standard_errors"][0] - error[t]),
                    )
            b.extend(coef)
            se.extend(error)
        b, se = np.array(b), np.array(se)
        for signal in (0.0, 0.005, 0.015):
            z = (b + signal) / se
            for alpha in (0.05, 0.005):
                hits = int((abs(z) > norm.isf(alpha / 2)).sum())
                lo, hi = interval(hits, len(z))
                rows.append(
                    dict(
                        noise=kind,
                        signal=signal,
                        alpha=alpha,
                        draws=len(z),
                        hits=hits,
                        rejection=hits / len(z),
                        lower=lo,
                        upper=hi,
                        mean_estimate=float(b.mean() + signal),
                        bias=float(b.mean()),
                        mean_se=float(se.mean()),
                        empirical_sd=float(b.std(ddof=1)),
                        coverage=float(np.mean(abs(b) <= norm.isf(0.025) * se)),
                        qualification=decision(hits, len(z))
                        if alpha == 0.05 and signal == 0
                        else None,
                    )
                )
    with a.out.open("x") as out:
        json.dump(
            _jsonable(
                dict(
                    n=n,
                    m=ref.metadata.get("complete_variant_count", len(ref.metadata["variants"]["ids"])),
                    seed=165728,
                    records=rows,
                    nuisance_rank=u.shape[1],
                    nuisance_saturated_rows=int(sat.sum()),
                    max_active_leverage=float(lev[~sat].max()),
                    feature_effective_support=float(
                        1 / np.sum((r[:, 0] / np.sqrt(h)) ** 4)
                    ),
                    batched_vs_public_max_abs_error=first_error,
                    seconds=time.perf_counter() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    fixed="actual full-marker genotype features and complete local main-effect design; conditional mean in C; signal coefficient; no reference probes or sketches",
                    regenerated="2000 independent residual draws per noise model; OLS nuisance and HC3 refit each draw; shared residual draws across signal strengths",
                    scope="Design-specific conditional finite-mean validation. Does not establish observed-height mean correctness or remove polygenic confounding. No extrapolation to other high-leverage designs.",
                )
            ),
            out,
            indent=2,
        )


if __name__ == "__main__":
    main()
