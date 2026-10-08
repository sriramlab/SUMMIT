"""Reconstruct every old leverage departure with identical outcomes and RNG draws.

Only selected training fits are rerun. The old causal mean is never replaced by
its fitted PGS. Compare old study-imputed and frozen-imputed nuisance bases.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from bed_reader import to_bed
from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.directions import score_frozen
from summit.epistasis.models import target_design
from summit.epistasis.prepare import fit_scale
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis as basis
from summit.prediction.genotype import FileGenotypeSource
from scripts.epistasis.robust_validation import load_panel


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--real-genotypes", required=True)
    p.add_argument("--null-only", action="store_true")
    a = p.parse_args()
    old = pd.read_csv(
        "benchmarks/epistasis/direction_confirm_scope_20261002/replicates.csv"
    )
    old["setting"] = old["setting"].fillna("null")
    departures = old[old.outside_scope.notna()]
    selected = set(zip(departures.setting, departures.replicate)) | {
        ("null", 67),
        ("null", 74),
    }
    if a.null_only:
        selected = {(s, r) for s, r in selected if s == "null"}
    n0, n1, m = 2048, 8000, 16384
    n = n0 + n1
    x, dom, panel, raw = load_panel(a.real_genotypes, 490123, n, m, return_raw=True)
    missing = np.isnan(raw[n0:, 24:88]).any(axis=1)
    rng = np.random.default_rng(284961)
    records = []
    with tempfile.TemporaryDirectory(
        prefix="imputation-reconstruction-", dir=a.scratch
    ) as tmp:
        root = Path(tmp)
        ids = list(map(str, range(n)))
        region = np.arange(24, 88)
        to_bed(
            root / "input.bed",
            raw,
            properties=dict(
                fid=ids,
                iid=ids,
                sid=[f"v{i}" for i in range(m)],
                chromosome=["1"] * m,
                bp_position=np.arange(1, m + 1),
                allele_1=["A"] * m,
                allele_2=["G"] * m,
            ),
        )
        del raw
        (root / "train.tsv").write_text(
            "FID IID\n" + "".join(f"{i} {i}\n" for i in range(n0))
        )
        (root / "variants.txt").write_text(
            "\n".join(f"v{i}" for i in range(1, m)) + "\n"
        )
        (root / "region.txt").write_text("\n".join(f"v{i}" for i in region) + "\n")
        features = x[:, 0, None] * x[:, region] / 8
        local = np.column_stack([np.ones(n), x[:, :24], dom[:, :24]])
        for setting, strength, arch in [
            ("null", 0.0, "mixed"),
            ("mixed_weak", 0.002, "mixed"),
            ("mixed_moderate", 0.01, "mixed"),
            ("aligned_weak", 0.002, "aligned"),
        ]:
            for rep in range(100):
                additive = rng.normal(size=m) * np.sqrt(0.3 / m)
                effects = rng.normal(size=64) * np.sqrt(strength)
                if arch == "aligned":
                    effects = np.full(64, np.sqrt(strength))
                noise = rng.normal(size=n)
                if (setting, rep) not in selected:
                    continue
                y = (
                    x @ additive
                    + 0.5 * dom[:, 0]
                    + features @ effects
                    + noise * np.sqrt(0.4 + 0.6 * x[:, 0] ** 2)
                )
                stem = f"{setting}-{rep}"
                (root / "y.tsv").write_text(
                    "FID IID y\n"
                    + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
                )
                spec = dict(
                    kind="summit.epistasis.train_direction",
                    schema_version=1,
                    genotypes=dict(geno="input.bed"),
                    samples="train.tsv",
                    phenotype=dict(file="y.tsv", column="y", unit="simulated"),
                    target="v0",
                    variants="variants.txt",
                    interaction_variants="region.txt",
                    local_variants=[f"v{i}" for i in range(24)],
                    dominance_variants=[f"v{i}" for i in range(24)],
                    prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                    storage="packed",
                    solver=dict(rtol=1e-6, max_iterations=150),
                )
                path = root / "train.json"
                path.write_text(json.dumps(spec))
                cli(["train-direction", str(path), "--out", str(root / stem)])
                definitions = [
                    dict(
                        name="pgs",
                        direction=stem + "/direction.json",
                        component=0,
                        adjust=True,
                    ),
                    dict(
                        name="interaction",
                        direction=stem + "/direction.json",
                        component=1,
                    ),
                ]
                with FileGenotypeSource(root / "input.bed") as source:
                    rows = np.arange(n0, n)
                    scores, adjust, _ = score_frozen(
                        definitions,
                        root,
                        source,
                        rows,
                        threads=1,
                        block_size=256,
                        memory_bytes=2**30,
                        main_variants=[f"v{i}" for i in range(88)],
                    )
                    scale = fit_scale(source, rows)
                    design = target_design(
                        source,
                        rows,
                        scale,
                        components=[],
                        annotations={"all": np.ones(m)},
                        additive_annotations=["all"],
                        allow_additive_only=True,
                        local_variants=[f"v{i}" for i in range(88)],
                        dominance_variants=[f"v{i}" for i in range(24)],
                        main_imputation=scores["interaction"]["main_imputation"],
                    )
                learned = scores["interaction"]["values"]
                oldc = np.column_stack([local[n0:], x[n0:, region], adjust, learned])
                newc = np.column_stack([design["fixed_effects"], adjust, learned])
                for version, c in [("original", oldc), ("consistent", newc)]:
                    u = basis(c[:, :-1])
                    uf = basis(c)
                    e = learned - u @ (u.T @ learned)
                    diagnostics = dict(
                        base_rank=u.shape[1],
                        full_rank=uf.shape[1],
                        base_leverage=float((u * u).sum(1).max()),
                        full_leverage=float((uf * uf).sum(1).max()),
                        score_remainder_fraction=float(e @ e / (learned @ learned)),
                        remainder_on_missing=float(e[missing] @ e[missing] / (e @ e)),
                    )
                    for name, f in [
                        ("learned_direction", x[n0:, 0, None] * learned[:, None]),
                        ("additive_PGS_direction", x[n0:, 0, None] * adjust),
                        ("regional_kernel", features[n0:]),
                        ("supplied_burden", features[n0:].sum(1, keepdims=True)),
                    ]:
                        s = prepare_robust_scores(
                            f,
                            y[n0:],
                            c,
                            feature_names=tuple(f"f{i}" for i in range(f.shape[1])),
                            trait_names=("y",),
                            metadata={},
                        )
                        fit = robust_score_tests(s)
                        previous = old[
                            (old.setting == setting)
                            & (old.replicate == rep)
                            & (old.method == name)
                        ].iloc[0]
                        records.append(
                            dict(
                                setting=setting,
                                replicate=rep,
                                version=version,
                                method=name,
                                p=fit["kernel_p"],
                                old_p=previous.p,
                                max_leverage=s.metadata["max_leverage"],
                                outside=list(s.metadata["outside_confirmation_design"]),
                                **diagnostics,
                            )
                        )
                print(stem, "done", flush=True)
    with a.out.open("x") as out:
        json.dump(
            _jsonable(
                dict(
                    panel=panel,
                    old_genotype_seed=490123,
                    old_phenotype_seed=284961,
                    records=records,
                    interpretation="Same old outcomes and raw pair features; nuisance basis alone changed. The prospectively corrected workflow also uses the frozen missing basis in constituent product features. This reconstruction is diagnostic, not independent calibration.",
                )
            ),
            out,
            indent=2,
        )


if __name__ == "__main__":
    main()
