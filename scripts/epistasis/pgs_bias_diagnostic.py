"""Reconstruct the fixed-real-panel PGS failure without changing its RNG stream."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import norm
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis as basis
from summit.epistasis.cli import _jsonable
from scripts.epistasis import direction_continuation as driver


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    original_load = driver.load_panel
    original_prepare = driver.prepare_robust_scores
    state = {}
    results = []

    def capture_panel(*args, **kwargs):
        values = original_load(*args, **kwargs)
        raw = values[3]
        mu = np.nanmean(raw, axis=0)
        mu0 = np.nanmean(raw[:2048], axis=0)
        x = (np.where(np.isnan(raw), mu0, raw) - mu) / np.sqrt(mu * (1 - mu / 2))
        rng = np.random.default_rng(682194)
        effects = rng.normal(size=8192) * np.sqrt(0.3 / 8192)
        state.update(
            additive=(x @ effects)[2048:],
            dominance=0.5 * values[1][2048:, 0],
            variance=0.4 + 0.6 * x[2048:, 0] ** 2,
        )
        return values

    def diagnostic(features, y, c, **kwargs):
        summary = original_prepare(features, y, c, **kwargs)
        if features.shape[1] == 1:
            u = basis(c)
            r = features[:, 0] - u @ (u.T @ features[:, 0])
            contrast = r / (r @ r)
            mean = state["additive"] + state["dominance"]
            oracle = original_prepare(features, y - mean[:, None], c, **kwargs)
            z = summary.scores[0] / np.sqrt(summary.score_covariance[:, 0, 0])
            zo = oracle.scores[0] / np.sqrt(oracle.score_covariance[:, 0, 0])
            sd = np.sqrt((contrast**2) @ state["variance"])
            results.append(
                dict(
                    method=["learned_direction", "additive_PGS", "burden"][
                        len(results)
                    ],
                    additive_shift=float(contrast @ state["additive"]),
                    dominance_shift=float(contrast @ state["dominance"]),
                    true_noise_sd=float(sd),
                    mean_shift_in_noise_sd=float(contrast @ mean / sd),
                    observed_hits=int(np.sum(abs(z) > norm.isf(0.025))),
                    oracle_mean_removed_hits=int(np.sum(abs(zo) > norm.isf(0.025))),
                    draws=y.shape[1],
                    conditioning="genotypes, true causal effects, direction training and fixed penalty; confirmation residuals regenerated",
                )
            )
        return summary

    def aggregates(records, out):
        frame = pd.DataFrame(records)
        return (
            frame.groupby(["setting", "method"], dropna=False)
            .agg(
                rejection=("p", lambda x: np.mean(x <= 0.05)),
                failures=("failed", "sum"),
            )
            .reset_index()
        )

    argv = [
        "direction_continuation",
        "--out",
        str(a.out / "reconstruction"),
        "--scratch",
        str(a.scratch),
        "--real-genotypes",
        a.genotypes,
        "--replicates",
        "1",
        "--nested-draws",
        "2000",
        "--training-samples",
        "2048",
        "--test-samples",
        "4096",
        "--variants",
        "8192",
        "--genotype-seed",
        "261849",
        "--phenotype-seed",
        "682194",
    ]
    with patch.object(driver, "load_panel", capture_panel), patch.object(
        driver, "prepare_robust_scores", diagnostic
    ), patch.object(driver, "reduction", aggregates), patch.object(sys, "argv", argv):
        driver.main()
    (a.out / "diagnostic.json").write_text(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
