"""Compact aggregate figures; never copy participant-level cohort references."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_studies(paths, phase):
    tables, frozen, designs = [], [], []
    for root in paths:
        design = json.loads((root / "design.json").read_text())
        name = root.name.removeprefix("epistasis_intact_").removesuffix("_20261003")
        summary = root / "summary_corrected.csv"
        if not summary.exists():
            summary = root / "summary.csv"
        table = pd.read_csv(summary)
        table["study"] = name
        table["phase"] = phase
        table["target"] = design["panel"]["actual_target"]
        table["n_train"] = design["arguments"]["training_samples"]
        table["n_test"] = design["arguments"]["test_samples"]
        table["biological_null"] = table.setting.map(
            {k: v["biological_null"] for k, v in design["generating_strengths"].items()}
        )
        tables.append(table)
        if (root / "frozen_summary.csv").exists():
            f = pd.read_csv(root / "frozen_summary.csv")
            f["study"] = name
            frozen.append(f)
        designs.append(
            dict(
                study=name,
                phase=phase,
                design=design,
                resources=json.loads((root / "resources.json").read_text()),
            )
        )
    return tables, frozen, designs


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def intervals(ax, frame, column, *, label=None, offset=0, color=None):
    if frame.empty:
        return
    values = frame[column].to_numpy(float)
    bounds = np.array([json.loads(v) for v in frame[column + "_interval"]])
    yy = np.arange(len(frame)) + offset
    ax.errorbar(
        values,
        yy,
        xerr=np.vstack([values - bounds[:, 0], bounds[:, 1] - values]),
        fmt="o",
        markersize=4,
        capsize=2,
        color=color,
        label=label,
    )


def diagnostic_precision(designs, out, *, draws=1000, seed=52979):
    """Monte Carlo error of diagnostic summaries, never coefficient inference.

    Resample independent full replicates within each setting, or outcome draws
    within each separately identified frozen model. Do not pool the two levels.
    """
    rng = np.random.default_rng(seed)
    records = []
    for study in designs:
        root = Path(study["design"]["arguments"]["out"])
        for filename, level in (
            ("replicates.csv", "full_pipeline"),
            ("frozen_replicates.csv", "conditional_model"),
        ):
            if not (root / filename).exists():
                continue
            frame = pd.read_csv(root / filename)
            keys = ["sampling", "setting", "adjustment", "method"]
            for key, group in frame.groupby(keys):
                valid = group[~group.failed].dropna(subset=["estimate", "truth", "se"])
                n = len(valid)
                if n < 10:
                    continue
                error = (valid.estimate - valid.truth).to_numpy()
                se = valid.se.to_numpy()
                indices = rng.integers(n, size=(draws, n))
                errors = error[indices]
                boot_sd = errors.std(axis=1, ddof=1)
                boot_bias = errors.mean(axis=1)
                boot_ratio = np.sqrt(np.mean(se[indices] ** 2, axis=1)) / boot_sd
                row = dict(
                    zip(keys, key),
                    study=study["study"],
                    phase=study["phase"],
                    level=level,
                    scheduled=len(group),
                    valid=n,
                    bootstrap_draws=draws,
                    bootstrap_seed=seed,
                    bias=float(error.mean()),
                    error_sd=float(error.std(ddof=1)),
                    se_sd_ratio=float(np.sqrt(np.mean(se**2)) / error.std(ddof=1)),
                )
                for name, values in (
                    ("bias", boot_bias),
                    ("error_sd", boot_sd),
                    ("se_sd_ratio", boot_ratio),
                    ("normalized_bias", boot_bias / boot_sd),
                ):
                    row[name + "_mc_interval"] = json.dumps(
                        np.quantile(values, [0.025, 0.975]).tolist()
                    )
                records.append(row)
    with (out / "diagnostic_precision.csv").open("x") as handle:
        pd.DataFrame(records).to_csv(handle, index=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--development", type=Path, action="append", default=[])
    p.add_argument("--confirmation", type=Path, action="append", default=[])
    p.add_argument("--workload", type=Path)
    p.add_argument("--reuse", type=Path)
    p.add_argument("--whole", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    first, frozen, designs = read_studies(a.development, "development")
    second, more, other = read_studies(a.confirmation, "confirmation")
    frozen += more
    designs += other
    table = pd.concat(first + second, ignore_index=True)
    # Preserve valid-only summaries and explicitly expose scheduled denominators.
    for alpha in (0.05, 0.005, 0.0005):
        column = f"rejection_{alpha}"
        table[column + "_scheduled"] = table[column] * table.valid / table.scheduled
    # These labels evaluate MC precision, not model assumptions.
    for alpha, tolerance in ((0.05, 0.075), (0.005, 0.01)):
        upper = table[f"rejection_{alpha}_upper95"]
        table[f"excludes_material_inflation_{alpha}"] = upper < tolerance
    table.to_csv(a.out / "full_pipeline.csv", index=False)
    (a.out / "designs.json").write_text(json.dumps(designs, indent=2))
    diagnostic_precision(designs, a.out)
    if frozen:
        ff = pd.concat(frozen, ignore_index=True)
        ff.to_csv(a.out / "conditional_models.csv", index=False)
        fig, axs = plt.subplots(1, 2, figsize=(12, max(5, len(ff) * 0.27)), sharey=True)
        for ax, alpha in zip(axs, (0.05, 0.005)):
            intervals(ax, ff, f"rejection_{alpha}")
            ax.axvline(alpha, color="black", linestyle="--", linewidth=1)
            ax.axvline(
                0.075 if alpha == 0.05 else 0.01,
                color="red",
                linestyle=":",
                linewidth=1,
            )
            ax.set_xlabel(f"Zero-null rejection at {alpha}; 95% MC interval")
        axs[0].set_yticks(
            range(len(ff)),
            [f"{r.study}: {r.setting}, {r.sampling}" for r in ff.itertuples()],
        )
        axs[0].invert_yaxis()
        fig.suptitle("Conditional checks for each frozen training model")
        save(fig, a.out, "conditional_calibration")
    selected = table.query(
        "phase=='confirmation' and scheduled>=100 and method=='learned' and biological_null"
    ).copy()
    selected["label"] = [
        f"{r.study}: {r.setting}, {r.adjustment}" for r in selected.itertuples()
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(12, max(5, len(selected) * 0.24)), sharey=True
    )
    for ax, alpha in zip(axs, (0.05, 0.005)):
        intervals(ax, selected, f"rejection_{alpha}")
        ax.axvline(alpha, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Biological-null rejection at {alpha}; 95% MC interval")
    axs[0].set_yticks(range(len(selected)), selected.label)
    axs[0].invert_yaxis()
    fig.suptitle("Fresh full training and confirmation replicates")
    save(fig, a.out, "full_pipeline_calibration")
    power = table.query(
        "phase=='confirmation' and scheduled>=100 and setting in ['aligned','aligned_weak','sparse','mixed']"
    ).copy()
    groups = power[["study", "sampling", "setting", "adjustment"]].drop_duplicates()
    labels = [f"{r.study}: {r.setting}" for r in groups.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(4, len(groups) * 0.55)))
    for j, method in enumerate(("learned", "burden", "joint", "oracle")):
        part = groups.merge(
            power.query("method==@method"), on=list(groups.columns), how="left"
        )
        intervals(ax, part, "rejection_0.005", label=method, offset=(j - 1.5) * 0.15)
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Power at .005; matched samples and fixed signal variance")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, a.out, "matched_power")
    estimation = table.query(
        "phase=='confirmation' and scheduled>=100 and method=='learned'"
    ).copy()
    fig, axs = plt.subplots(
        1, 2, figsize=(12, max(5, len(estimation) * 0.23)), sharey=True
    )
    intervals(axs[0], estimation, "coverage")
    axs[0].axvline(0.95, color="black", ls="--")
    axs[0].set_xlabel("Coverage of training-conditional projection truth")
    ratio = estimation.get("rms_se", estimation.mean_se) / estimation.error_sd
    axs[1].plot(ratio, np.arange(len(ratio)), "o", ms=4)
    axs[1].axvline(1, color="black", ls="--")
    axs[1].set_xlabel("RMS SE / SD(estimate − conditional truth)")
    axs[0].set_yticks(
        range(len(estimation)),
        [f"{r.study}: {r.setting}, {r.adjustment}" for r in estimation.itertuples()],
    )
    axs[0].invert_yaxis()
    save(fig, a.out, "estimation")
    whole = []
    for root in a.whole:
        frame = pd.read_csv(root / "replicates.csv")
        design = json.loads((root / "design.json").read_text())
        from scripts.epistasis.intact_validation import binomial_interval

        for key, g in frame.groupby(["noise", "signal_expected_variance"]):
            v = g[~g.failed]
            row = dict(
                sampling=design["sampling"],
                noise=key[0],
                signal_variance=key[1],
                scheduled=len(g),
                failures=int(g.failed.sum()),
                unsupported=int(v.outside_scope.fillna("").ne("").sum()),
                truth=float(v.truth.iloc[0]),
                bias=float((v.beta - v.truth).mean()),
                error_sd=float((v.beta - v.truth).std(ddof=1)),
                mean_se=float(v.se.mean()),
                coverage=float(v.coverage.mean()),
            )
            for alpha in (0.05, 0.005):
                k = int((v.p < alpha).sum())
                row[f"rejection_{alpha}"] = k / len(v)
                row[f"rejection_{alpha}_scheduled"] = k / len(g)
                row[f"rejection_{alpha}_interval"] = json.dumps(
                    binomial_interval(k, len(v))
                )
            whole.append(row)
    if whole:
        pd.DataFrame(whole).to_csv(a.out / "whole_marker.csv", index=False)
    if a.workload:
        workload = json.loads(a.workload.read_text())
        records = workload["records"]
        pd.DataFrame(records).to_csv(a.out / "resource_stages.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            [r["stage"].replace("_", " ") for r in records],
            [r["seconds"] / 60 for r in records],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Measured minutes; shared host, observed cache state")
        ax.set_title(f"{workload['n']:,} participants × {workload['m']:,} markers")
        save(fig, a.out, "resources")
        compact = {
            k: workload[k] for k in workload if k not in ("primary", "confirmation")
        }
        model_manifest = Path(workload["root"]) / "trained/models/manifest.json"
        if model_manifest.exists():
            models = json.loads(model_manifest.read_text())
            compact["native_training_report"] = models["run_report"]
            compact["convergence"] = [
                model["convergence"]
                for trait in models["traits"]
                for model in trait["models"]
            ]
        compact["whole_marker_designs"] = [
            json.loads((root / "design.json").read_text()) for root in a.whole
        ]
        if a.reuse:
            compact["reuse"] = json.loads(a.reuse.read_text())
        (a.out / "workload.json").write_text(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
