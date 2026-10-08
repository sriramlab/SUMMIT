"""Bounded conditional-mean validation; retain misspecification and failed fits."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from summit.prediction.genotype import FileGenotypeSource
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.score import prepare_linear_scores, linear_score_tests
from summit.epistasis.cli import _jsonable
from scripts.epistasis.validate import interval


def load_panel(prefix, seed, n, m, *, return_raw=False):
    rng = np.random.default_rng(seed)
    if prefix:
        with FileGenotypeSource(prefix) as source:
            rows = np.sort(rng.choice(len(source.samples), n, replace=False))
            start = int(rng.integers(1000, len(source.variants.ids) - 2048))
            local = np.arange(start, start + 256)
            candidates = np.unique(
                rng.choice(
                    len(source.variants.ids),
                    min(8 * m, len(source.variants.ids)),
                    replace=False,
                )
            )
            source.prepare(rows, 256, 1)
            blocks = []
            axes = []
            retained = 0
            for indices in [local] + [
                v
                for v in np.array_split(
                    np.setdiff1d(candidates, local),
                    max(1, int(np.ceil(len(candidates) / 256))),
                )
                if len(v)
            ]:
                block = source.read(indices).astype(float)
                block[block == -127] = np.nan
                af = np.nanmean(block, axis=0) / 2
                keep = (
                    (af > 0.15)
                    & (af < 0.85)
                    & (np.mean(np.isnan(block), axis=0) < 0.02)
                )
                blocks.append(block[:, keep])
                axes.append(indices[keep])
                retained += int(keep.sum())
                if retained >= m + 256:
                    break
            raw = np.column_stack(blocks)
            candidates = np.concatenate(axes)
            del blocks
            mean = np.nanmean(raw, axis=0)
            valid = np.ones(len(candidates), dtype=bool)
            localpos = np.flatnonzero(valid & np.isin(candidates, local))
            chosen = []
            # Phenotype-blind numerical fixture admission, not target discovery.
            for i in localpos:
                v = np.nan_to_num(raw[:, i] - mean[i])
                v /= np.linalg.norm(v)
                if all(abs(v @ u) < 0.8 for _, u in chosen):
                    chosen.append((i, v))
                if len(chosen) == 24:
                    break
            if len(chosen) < 24:
                raise ValueError(
                    "regional fixture lacks 24 distinct supported variants"
                )
            lead = [i for i, _ in chosen]
            remaining = np.setdiff1d(np.flatnonzero(valid), lead)
            selected = np.r_[lead, rng.choice(remaining, m - len(lead), replace=False)]
            meta = dict(
                kind="real_genotypes",
                source_identity=source.identity,
                source_n=len(source.samples),
                source_m=len(source.variants.ids),
                seed=seed,
                window_start=start,
                selected_variant_indices=candidates[selected].tolist(),
                sample_selection="sorted seeded random subset",
                admission="MAF .15-.85, missing <.02; first 24 regional fixture loci absolute r<.8",
            )
            raw = raw[:, selected]
    else:
        raw = np.zeros((n, m))
        for _ in range(2):
            hap = rng.binomial(1, 0.3, (n, m))
            for j in range(1, 24):
                hap[:, j] = np.where(rng.random(n) < 0.4, hap[:, j - 1], hap[:, j])
            raw += hap
        raw[rng.random((n, m)) < 0.001] = np.nan
        meta = dict(kind="synthetic_discrete", seed=seed)
    means = np.nanmean(raw, axis=0)
    x = np.nan_to_num((raw - means) / np.sqrt(means * (1 - means / 2)))
    dominance = (raw[:, :24] == 1).astype(float)
    for j in range(24):
        dominance[np.isnan(raw[:, j]), j] = np.nanmean(
            np.where(np.isnan(raw[:, j]), np.nan, dominance[:, j])
        )
    meta.update(
        n=n,
        m=m,
        missing_fraction=float(np.isnan(raw).mean()),
        minimum_genotype_count=int(
            min(np.sum(raw == i, axis=0).min() for i in (0, 1, 2))
        ),
        local_max_abs_ld=float(np.max(abs(np.corrcoef(x[:, :24].T) - np.eye(24)))),
    )
    return (x, dominance, meta, raw) if return_raw else (x, dominance, meta)


def feature_cases(x):
    n, m = x.shape
    a = np.zeros(m)
    a[:3] = 1
    b = np.zeros(m)
    b[3:6] = 1
    over = b.copy()
    over[2] = 1
    f = x[:, 0, None] * x[:, 3:6] / np.sqrt(3)
    cases = {"supplied_pairs": f, "target_region": f}
    for name, left, right, within in [
        ("set_cross", a, b, False),
        ("set_overlap", a, over, False),
        ("within", a, a, True),
    ]:
        # Only six fixture loci participate. Construct their handful of columns
        # directly; the dense sample-kernel oracle intentionally caps N at 2048.
        pairs, weights = [], []
        for i in range(6):
            for j in range(i + 1, 6):
                weight = left[i] * right[j] + (0 if within else left[j] * right[i])
                if weight > 0:
                    pairs.append((i, j))
                    weights.append(weight)
        cases[name] = np.column_stack([x[:, i] * x[:, j] for i, j in pairs]) * np.sqrt(
            np.asarray(weights) / sum(weights)
        )
    # Fixed phenotype-independent directional sketches, not estimated moments.
    rng = np.random.default_rng(80031)
    source = x[:, 6:] @ rng.normal(size=(m - 6, 8)) / np.sqrt(m - 6)
    cases["target_genome_sketch8"] = x[:, 0, None] * source / np.sqrt(8)
    left = x[:, :3] @ rng.normal(size=(3, 8)) / np.sqrt(3)
    remainder = x[:, 3:] @ rng.normal(size=(m - 3, 8)) / np.sqrt(m - 3)
    cases["set_remainder_sketch8"] = left * remainder / np.sqrt(8)
    cases["weighted_score"] = (
        x[:, 0, None] * (x[:, 6:] @ np.ones(m - 6) / np.sqrt(m - 6))[:, None]
    )
    return cases


def run_panel(x, dom, seed, reps, label, records, design, *, full=True):
    rng = np.random.default_rng(seed)
    n, m = x.shape
    # All 24 local additive and heterozygote terms are supplied, independent of
    # simulated causal identity. Unused genome-wide loci outnumber samples.
    c = np.column_stack([np.ones(n), x[:, :24], dom[:, :24]])
    local_effect = rng.normal(size=24) * np.sqrt(0.3 / 24)
    mean = x[:, :24] @ local_effect + 0.6 * dom[:, 0]
    cases = feature_cases(x)
    for family, f in cases.items():
        u = thin_rank_revealing_fixed_effect_basis(c)
        pf = f - u @ (u.T @ f)
        direction = rng.normal(size=f.shape[1])
        direction *= np.sqrt(0.01 / np.mean((pf @ direction) ** 2))
        # Directions and magnitudes fixed once before residual replication.
        settings = ["null_gaussian", "null_hetero_dominance", "signal_mixed01"]
        if family == "supplied_pairs":
            settings += [
                "null_heavy_t5",
                "signal_sparse01",
                "signal_aligned01",
                "omitted_dominance",
                "omitted_additive_dense",
                "omitted_local_causal",
                "nonlinear_mean",
                "nonlinear_mean_adjusted",
                "unmeasured_environment",
            ]
        if not full:
            settings = settings[:3]
        for setting in settings:
            actual = c
            mu = mean.copy()
            coefficient = np.zeros(f.shape[1])
            variance = np.ones(n)
            if setting == "null_hetero_dominance":
                variance = 0.25 + 0.75 * x[:, 0] ** 2
            if setting == "null_heavy_t5":
                noise = rng.standard_t(5, (n, reps)) * np.sqrt(3 / 5)
            else:
                noise = rng.normal(size=(n, reps))
            if setting.startswith("signal"):
                coefficient = direction.copy()
                if setting == "signal_sparse01":
                    coefficient[:] = 0
                    coefficient[0] = np.sqrt(0.01 / np.mean(pf[:, 0] ** 2))
                if setting == "signal_aligned01":
                    coefficient = np.ones(f.shape[1]) * np.sqrt(
                        0.01 / np.mean(pf.sum(axis=1) ** 2)
                    )
            if setting == "omitted_dominance":
                actual = np.column_stack([np.ones(n), x[:, :24]])
            if setting == "omitted_additive_dense":
                mu += x @ (rng.normal(size=m) * np.sqrt(0.5 / m))
            if setting == "omitted_local_causal":
                mu += 1.5 * x[:, 1]
                actual = np.delete(c, [2, 26], axis=1)
            if setting.startswith("nonlinear_mean"):
                g = x[:, :24] @ local_effect
                mu += 0.5 * g * g
                if setting.endswith("adjusted"):
                    actual = np.column_stack([c, g * g])
            if setting == "unmeasured_environment":
                variance = 0.4 + 0.6 * x[:, 0] ** 2
            y = (
                mu[:, None]
                + (f @ coefficient)[:, None]
                + np.sqrt(variance)[:, None] * noise
            )
            name = label + "_" + family + "_" + setting
            design.append(
                dict(
                    setting=name,
                    n=n,
                    m=m,
                    family=family,
                    noise=setting,
                    coefficient=coefficient.tolist(),
                    realized_interaction_variance=float(np.var(pf @ coefficient)),
                    genotype_and_causal_identity="fixed",
                    effects="fixed realized effects across residual replicates",
                    residuals="regenerated",
                    sketches="fixed independent of phenotypes; eight directions",
                    nuisance="mean and HC3 refitted for each phenotype; no generating covariance supplied",
                    interval_target="conditional finite-design coefficient when mean is correctly specified; otherwise no biological-null coverage claim",
                    nonlinear_adjusted="oracle known index comparator; separate independent training investigation required",
                )
            )
            try:
                summary = prepare_robust_scores(
                    f,
                    y,
                    actual,
                    feature_names=tuple(f"f{i}" for i in range(f.shape[1])),
                    trait_names=tuple(map(str, range(reps))),
                    metadata={},
                )
                design[-1]["diagnostics"] = {
                    key: summary.metadata[key]
                    for key in (
                        "fixed_rank",
                        "feature_rank",
                        "max_leverage",
                        "information_condition",
                        "minimum_feature_effective_support",
                        "outside_confirmation_design",
                    )
                }
                for i in range(reps):
                    try:
                        fit = robust_score_tests(
                            summary, trait=i, burden=np.ones(f.shape[1])
                        )
                        for method, key in [
                            ("HC3_kernel", "kernel_p"),
                            ("HC3_burden", "burden_p"),
                            ("HC3_sparse", "sparse_bonferroni_p"),
                            ("HC3_adaptive", "adaptive_bonferroni_p"),
                        ]:
                            records.append(
                                dict(
                                    setting=name,
                                    method=method,
                                    replicate=i,
                                    p=fit[key],
                                    failed=False,
                                    beta0=fit["beta"][0],
                                    truth0=coefficient[0],
                                    se0=fit["standard_errors"][0],
                                    coverage=float(
                                        np.mean(
                                            abs(fit["beta"] - coefficient)
                                            <= 1.95996398454 * fit["standard_errors"]
                                        )
                                    ),
                                    leverage=summary.metadata["max_leverage"],
                                    effective_support=summary.metadata[
                                        "minimum_feature_effective_support"
                                    ],
                                )
                            )
                    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
                        records.append(
                            dict(
                                setting=name,
                                method="HC3_failed",
                                replicate=i,
                                p=np.nan,
                                failed=True,
                                error=str(e),
                            )
                        )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
                for i in range(reps):
                    records.append(
                        dict(
                            setting=name,
                            method="HC3_failed",
                            replicate=i,
                            p=np.nan,
                            failed=True,
                            error=str(e),
                        )
                    )
    print(label, "complete", flush=True)


def reduction(records, out):
    frame = pd.DataFrame(records)
    frame.to_csv(out / "replicates.csv", index=False)
    rows = []
    for (setting, method), part in frame.groupby(["setting", "method"], sort=False):
        hits = int((part.p <= 0.05).sum())
        low, high = interval(hits, len(part))
        rows.append(
            dict(
                setting=setting,
                method=method,
                fits=len(part),
                failures=int(part.failed.sum()),
                rejection=hits / len(part),
                lower=low,
                upper=high,
                valid_rejection=float((part.loc[~part.failed, "p"] <= 0.05).mean()),
                rejection_005=float((part.p <= 0.005).mean()),
                bias=float((part.beta0 - part.truth0).mean())
                if "beta0" in part
                else np.nan,
                empirical_sd=float(part.beta0.std()) if "beta0" in part else np.nan,
                mean_se=float(part.se0.mean()) if "se0" in part else np.nan,
                coverage=float(part.coverage.mean()) if "coverage" in part else np.nan,
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-genotypes")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--variants", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--genotype-seed", type=int, default=82031)
    parser.add_argument("--phenotype-seed", type=int, default=91272)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("at most 100 full fits per setting")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cpu = time.process_time()
    x, d, meta = load_panel(
        args.real_genotypes, args.genotype_seed, args.samples, args.variants
    )
    records = []
    design = []
    run_panel(
        x,
        d,
        args.phenotype_seed,
        args.replicates,
        "real" if args.real_genotypes else "discrete",
        records,
        design,
    )
    table = reduction(records, args.out)
    (args.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    panel=meta,
                    settings=design,
                    arguments=vars(args) | {"out": str(args.out)},
                    seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(table[["setting", "method", "rejection", "failures"]].to_string(index=False))


if __name__ == "__main__":
    main()
