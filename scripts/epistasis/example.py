"""Write a reproducible synthetic training/confirmation example at a new path."""
import argparse, json
from pathlib import Path
import numpy as np
from bed_reader import to_bed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--variants", type=int, default=8192)
    a = p.parse_args()
    if not 200 <= a.samples <= 8192 or not 64 <= a.variants <= 16384:
        raise ValueError("bounded example requires N200..8192, M64..16384")
    a.out.mkdir(parents=True, exist_ok=False)
    n, m = a.samples, a.variants
    rng = np.random.default_rng(875391)
    raw = rng.binomial(2, 0.35, (n, m)).astype(np.int8)
    ids = list(map(str, range(n)))
    to_bed(
        a.out / "cohort.bed",
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
    x = (raw - 0.7) / np.sqrt(0.455)
    direction = x[:, 24:40].sum(1) / 4
    y = (
        0.3 * x[:, 1]
        + 0.5 * (raw[:, 0] == 1)
        + 0.12 * x[:, 0] * direction
        + rng.normal(size=n) * np.sqrt(0.4 + 0.6 * x[:, 0] ** 2)
    )
    (a.out / "phenotypes.tsv").write_text(
        "FID IID y\n" + "".join(f"{i} {i} {v:.17g}\n" for i, v in enumerate(y))
    )
    for name, rows in [("training", range(n // 3)), ("confirmation", range(n // 3, n))]:
        (a.out / (name + ".tsv")).write_text(
            "FID IID\n" + "".join(f"{i} {i}\n" for i in rows)
        )
    (a.out / "background.txt").write_text(
        "\n".join(f"v{i}" for i in range(1, m)) + "\n"
    )
    (a.out / "region.txt").write_text("\n".join(f"v{i}" for i in range(24, 40)) + "\n")
    local = [f"v{i}" for i in range(8)]
    train = dict(
        kind="summit.epistasis.train_direction",
        schema_version=1,
        genotypes=dict(geno="cohort.bed"),
        samples="training.tsv",
        phenotype=dict(file="phenotypes.tsv", column="y", unit="synthetic_trait_units"),
        target="v0",
        variants="background.txt",
        interaction_variants="region.txt",
        local_variants=local,
        dominance_variants=local,
        prior=dict(additive=0.5, interaction=0.05, residual=1.0),
        storage="packed",
        solver=dict(rtol=1e-6, max_iterations=150),
    )
    job = dict(
        id="confirmation",
        additive_annotations=["all"],
        local_variants=local,
        dominance_variants=local,
        components=[
            dict(
                name="learned_direction", frozen_score="direction", background="target"
            )
        ],
        inference=dict(
            method="robust_mean", main_effects="declared", save_reference=True
        ),
    )
    confirm = dict(
        kind="summit.epistasis.prepare",
        schema_version=1,
        genotypes=dict(geno="cohort.bed"),
        samples="confirmation.tsv",
        phenotypes=dict(
            file="phenotypes.tsv", columns=["y"], unit="synthetic_trait_units"
        ),
        annotations=dict(target=dict(v0=1)),
        frozen_scores=[
            dict(
                name="PGS", direction="trained/direction.json", component=0, adjust=True
            ),
            dict(name="direction", direction="trained/direction.json", component=1),
        ],
        jobs=[job],
    )
    pairs = dict(confirm)
    prespecified = {k: v for k, v in confirm.items() if k != "frozen_scores"}
    # The scientific weights are equal on raw allele counts. Convert once to
    # the declared confirmation-cohort HWE scale, using no phenotype values.
    mean_confirmation = raw[n // 3 :, 24:40].mean(axis=0)
    score_weights = np.sqrt(mean_confirmation * (1 - mean_confirmation / 2)) / (
        4 * np.sqrt(0.455)
    )
    prespecified["jobs"] = [
        dict(
            id="prespecified",
            additive_annotations=["all"],
            local_variants=local,
            dominance_variants=local,
            components=[
                dict(
                    name="prespecified_direction",
                    background="target",
                    score={
                        f"v{i}": float(w) for i, w in zip(range(24, 40), score_weights)
                    },
                )
            ],
            inference=dict(
                method="robust_mean", main_effects="declared", save_reference=True
            ),
        )
    ]
    pairs["jobs"] = [
        dict(
            id="pairs",
            additive_annotations=["all"],
            local_variants=local,
            dominance_variants=local,
            pairs=[["v0", "v24"], ["v0", "v25"]],
            burden_weights=[1.0, 1.0],
            inference=dict(method="robust_mean", save_reference=True),
        )
    ]
    follow = dict(
        kind="summit.epistasis.followup",
        schema_version=1,
        summary="pairs/pairs.robust-score.npz",
        groups=dict(supplied_region=[0, 1]),
    )
    for name, value in [
        ("train", train),
        ("confirm", confirm),
        ("prespecified", prespecified),
        ("pairs", pairs),
        ("followup", follow),
    ]:
        (a.out / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    print(a.out.resolve())


if __name__ == "__main__":
    main()
