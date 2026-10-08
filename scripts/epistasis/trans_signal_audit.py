"""Replay frozen genotype draws, without refitting, to audit signal units.

The RNG replay is checked against every saved burden coefficient's population
truth. No participant arrays or reconstructed phenotypes are persisted.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.epistasis.trans_validation import population_blocks


def audit(path):
    design = json.loads((path / "design.json").read_text())
    args = design["arguments"]
    pools, axis, meta = population_blocks(
        args["genotypes"], args["background_chromosome"], args["genotype_seed"]
    )
    if meta != design["panel"]:
        raise ValueError("genotype donor definition changed")
    ga, gb = pools
    ma = ga.shape[1]
    mb = gb.shape[1]
    mean = np.r_[ga.mean(0), gb.mean(0)]
    inv = 1 / np.sqrt(mean * (1 - mean / 2))
    xa = (ga - mean[:ma]) * inv[:ma]
    xb = (gb - mean[ma:]) * inv[ma:]
    rng = np.random.default_rng(args["seed"])
    ba = rng.normal(size=ma) * np.sqrt(0.35 / ma)
    bb = rng.normal(size=mb) * np.sqrt(0.35 / mb)
    mean_a = xa @ ba + 0.8 * xa[:, 12] + 0.6 * (ga[:, 0] == 1)
    mean_b = xb @ bb
    hs = {
        "aligned_weak": (0.035, xb.sum(1) / np.sqrt(mb)),
        "aligned_moderate": (0.075, xb.sum(1) / np.sqrt(mb)),
        "sparse_moderate": (0.075, xb[:, 0]),
        "distributed_moderate": (0.075, xb @ (rng.normal(size=mb) / np.sqrt(mb))),
    }
    old = pd.read_csv(path / "replicates.csv")
    if old.failed.any():
        raise ValueError("replay currently requires the recorded no-failure path")
    n0 = args["training_samples"]
    n1 = args["test_samples"]
    n = n0 + n1
    records = []
    max_error = 0.0
    for rep in range(args["replicates"]):
        ia = rng.integers(len(ga), size=n)
        ib = rng.integers(len(gb), size=n)
        raw = np.column_stack([ga[ia[n0:]], gb[ib[n0:]]])
        study_mean = raw.mean(0)
        study_inv = 1 / np.sqrt(study_mean * (1 - study_mean / 2))
        e = gb @ (study_inv[ma:] / np.sqrt(mb))
        ec = e - e.mean()
        for setting, (strength, h) in hs.items():
            truth = (
                inv[0]
                / study_inv[0]
                * strength
                * np.mean(ec * (h - h.mean()))
                / np.mean(ec**2)
            )
            saved = old[
                (old.method == "burden")
                & (old.setting == setting)
                & (old.replicate == rep)
            ].truth0.iloc[0]
            max_error = max(max_error, abs(truth - saved))
            records.append(
                dict(
                    panel=path.name,
                    replicate=rep,
                    setting=setting,
                    generating_coefficient=strength,
                    expected_signal_variance=float(
                        strength**2 * np.var(xa[:, 0]) * np.var(h)
                    ),
                    realized_signal_variance=float(
                        np.var(strength * xa[ia[n0:], 0] * h[ib[n0:]])
                    ),
                    expected_additive_mean_variance=float(
                        np.var(mean_a) + np.var(mean_b)
                    ),
                    realized_additive_mean_variance=float(
                        np.var(mean_a[ia[n0:]] + mean_b[ib[n0:]])
                    ),
                )
            )
        rng.normal(size=n)
        rng.normal(size=n)
        rng.standard_t(5, size=n)
        if rep < args["nested_models"]:
            for _ in range(args["nested_draws"]):
                rng.integers(len(ga), size=n1)
                rng.integers(len(gb), size=n1)
                rng.normal(size=n1)
                rng.standard_t(5, size=n1)
    if max_error > 1e-10:
        raise ArithmeticError(
            f"RNG/scale replay disagrees with saved truths: {max_error}"
        )
    return records, dict(
        panel=path.name,
        maximum_truth_replay_error=max_error,
        scope="generating expected variance fixed per panel, realized test-sample variance; no per-replicate renormalization",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    records = []
    checks = []
    for path in args.inputs:
        data, check = audit(path)
        records.extend(data)
        checks.append(check)
    pd.DataFrame(records).to_csv(args.out / "signal_variance.csv", index=False)
    (args.out / "verification.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
