"""Population trans validation with authentic real-genotype block distributions.

Each synthetic individual is an independent draw from the product of two
empirical chromosome-block distributions. Within-block LD and authentic axes
are retained. This is NOT validation of independence in the original cohort.
Effects stay fixed. Native direction learning is repeated, and its conditional
population coefficient is calculated from the complete donor distributions.
Training uses a strong aligned interaction architecture across confirmation
settings: this studies transfer of an external direction, not matched same-trait
learning. See intact_validation.py for matched training and confirmation.
"""
import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from bed_reader import to_bed
from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.artifacts import load_prediction_models
from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.robust import (
    prepare_robust_scores,
    robust_score_tests,
    load_robust_scores,
)
from scripts.epistasis.robust_validation import reduction


def population_blocks(path, background_chromosome, seed, sizes=(32, 96)):
    rng = np.random.default_rng(seed)
    pools, axes, meta = [], [], []
    with FileGenotypeSource(path, genome_build="GRCh37") as source:
        axis = source.variants
        for chrom, position, size in [
            ("12", 66358347, sizes[0]),
            (background_chromosome, 50000000, sizes[1]),
        ]:
            candidates = np.flatnonzero(
                (np.asarray(axis.chromosome) == chrom)
                & (abs(np.asarray(axis.position) - position) < 4000000)
            )
            if chrom == "12":
                lead = axis.ids.index("12:66358347")
                candidates = np.r_[
                    lead, rng.permutation(candidates[candidates != lead])
                ]
            else:
                candidates = rng.permutation(candidates)
            kept, blocks = [], []
            source.prepare(np.arange(len(source.samples)), 128, 1)
            for begin in range(0, len(candidates), 128):
                take = candidates[begin : begin + 128]
                order = np.argsort(take)
                raw = source.read(take[order]).astype(float)[:, np.argsort(order)]
                observed = raw != -127
                af = np.sum(np.where(observed, raw, 0), axis=0) / observed.sum(0) / 2
                good = (af > 0.15) & (af < 0.85) & (observed.mean(0) > 0.995)
                for j in np.flatnonzero(good):
                    kept.append(take[j])
                    blocks.append(raw[:, j])
                    if len(kept) == size:
                        break
                if len(kept) == size:
                    break
            if len(kept) != size or (chrom == "12" and kept[0] != lead):
                raise ValueError("prespecified block lacks common supported variants")
            raw = np.column_stack(blocks)
            complete = np.all(raw != -127, axis=1)
            if complete.sum() < 1000:
                raise ValueError("donor block has insufficient complete support")
            pools.append(raw[complete])
            axes.append(axis.subset(kept))
            meta.append(
                dict(
                    chromosome=chrom,
                    positions=list(axes[-1].position),
                    variants=list(axes[-1].ids),
                    donor_count=int(complete.sum()),
                    complete_block_selection="genotype-only donor distribution definition, not a real-cohort missingness claim",
                )
            )
        from summit.prediction.spec import VariantAxis

        combined = VariantAxis(
            **{
                k: getattr(axes[0], k) + getattr(axes[1], k)
                for k in ("ids", "chromosome", "position", "counted", "other")
            },
            genome_build="GRCh37",
        )
        return pools, combined, dict(blocks=meta, source_identity=source.identity)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    pools, axis, meta = population_blocks(
        args.genotypes, args.background_chromosome, args.genotype_seed
    )
    ga, gb = pools
    ma, mb = ga.shape[1], gb.shape[1]
    m = ma + mb
    means = np.r_[ga.mean(0), gb.mean(0)]
    inv = 1 / np.sqrt(means * (1 - means / 2))
    xa = (ga - means[:ma]) * inv[:ma]
    xb = (gb - means[ma:]) * inv[ma:]
    # Fixed architectures. Hidden strong local variants are deliberately absent
    # from the declared four-locus adjustment (though observed by the ridge fit).
    ba = rng.normal(size=ma) * np.sqrt(0.35 / ma)
    bb = rng.normal(size=mb) * np.sqrt(0.35 / mb)
    additive_a = xa @ ba + 0.8 * xa[:, 12] + 0.6 * (ga[:, 0] == 1)
    additive_b = xb @ bb
    aligned = xb.sum(1) / np.sqrt(mb)
    sparse = xb[:, 0]
    distributed = xb @ (rng.normal(size=mb) / np.sqrt(mb))
    # Directions trained on one external architecture. The same frozen training
    # experiment is used for all confirmation alternatives and nulls.
    train_strength = 0.15
    records = []
    nested = []
    settings = [
        "finite_null",
        "dense_hidden_null",
        "heavy_null",
        "aligned_weak",
        "aligned_moderate",
        "sparse_moderate",
        "distributed_moderate",
        "nonlinear_scale",
    ]
    for rep in range(args.replicates):
        with tempfile.TemporaryDirectory(prefix="trans-", dir=args.scratch) as tmp:
            root = Path(tmp)
            n0 = args.training_samples
            n1 = args.test_samples
            n = n0 + n1
            ia = rng.integers(len(ga), size=n)
            ib = rng.integers(len(gb), size=n)
            raw = np.column_stack([ga[ia], gb[ib]])
            ids = list(map(str, range(n)))
            to_bed(
                root / "g.bed",
                raw,
                properties=dict(
                    fid=ids,
                    iid=ids,
                    sid=axis.ids,
                    chromosome=axis.chromosome,
                    bp_position=axis.position,
                    allele_1=axis.counted,
                    allele_2=axis.other,
                ),
            )
            for name, rows in [("train", range(n0)), ("test", range(n0, n))]:
                (root / name).write_text(
                    "FID IID\n" + "".join(f"{i} {i}\n" for i in rows)
                )
            (root / "variants").write_text(
                "\n".join(
                    v
                    for j, v in enumerate(axis.ids)
                    if j != 0 and (not args.omit_causal_main or j != 12)
                )
                + "\n"
            )
            (root / "interaction").write_text("\n".join(axis.ids[ma:]) + "\n")
            x = xa[ia, 0]
            base = additive_a[ia] + additive_b[ib]
            sd = np.sqrt(0.4 + 0.6 * x * x)
            ytrain = base + train_strength * x * aligned[ib] + sd * rng.normal(size=n)
            finite = 0.7 * x + 0.4 * (raw[:, 0] == 1) + 0.3 * xb[ib, 0]
            noise = sd * rng.normal(size=n)
            outcomes = np.column_stack(
                [
                    finite + noise,
                    base + noise,
                    base + sd * rng.standard_t(5, size=n) / np.sqrt(5 / 3),
                    base + 0.035 * x * aligned[ib] + noise,
                    base + 0.075 * x * aligned[ib] + noise,
                    base + 0.075 * x * sparse[ib] + noise,
                    base + 0.075 * x * distributed[ib] + noise,
                    base + 0.3 * base**2 + noise,
                ]
            )
            frame = pd.DataFrame(outcomes, columns=settings)
            frame.insert(0, "IID", ids)
            frame.insert(0, "FID", ids)
            frame["train_y"] = ytrain
            frame.to_csv(root / "y", sep="\t", index=False)
            train = dict(
                kind="summit.epistasis.train_direction",
                schema_version=1,
                genotypes=dict(geno="g.bed", genome_build="GRCh37"),
                samples="train",
                phenotype=dict(
                    file="y", column="train_y", unit="simulated original scale"
                ),
                target=axis.ids[0],
                variants="variants",
                interaction_variants="interaction",
                trans_only=True,
                local_variants=list(axis.ids[:4]),
                dominance_variants=list(axis.ids[:4]),
                prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                storage="packed",
                solver=dict(rtol=1e-8, max_iterations=200),
            )
            (root / "train.json").write_text(json.dumps(train))
            try:
                cli(
                    [
                        "train-direction",
                        str(root / "train.json"),
                        "--out",
                        str(root / "trained"),
                    ]
                )
                common = dict(
                    additive_annotations=["all"],
                    trans_target=axis.ids[0],
                    local_variants=list(axis.ids[:4]) + [axis.ids[ma]],
                    dominance_variants=list(axis.ids[:4]),
                    inference=dict(
                        method="robust_mean",
                        sampling_model="iid_population_projection",
                        main_effects="declared",
                        save_reference=True,
                    ),
                )
                spec = dict(
                    kind="summit.epistasis.prepare",
                    schema_version=1,
                    genotypes=dict(geno="g.bed", genome_build="GRCh37"),
                    samples="test",
                    phenotypes=dict(
                        file="y", columns=settings, unit="simulated original scale"
                    ),
                    annotations=dict(target={axis.ids[0]: 1.0}),
                    frozen_scores=[
                        dict(
                            name="pgs",
                            direction="trained/direction.json",
                            component=0,
                            adjust=True,
                        ),
                        dict(
                            name="direction",
                            direction="trained/direction.json",
                            component=1,
                        ),
                    ],
                    jobs=[
                        dict(
                            id="learned",
                            **common,
                            components=[
                                dict(
                                    name="direction",
                                    frozen_score="direction",
                                    background="target",
                                )
                            ],
                        ),
                        dict(
                            id="burden",
                            **common,
                            components=[
                                dict(
                                    name="burden",
                                    score={v: 1 / np.sqrt(mb) for v in axis.ids[ma:]},
                                    background="target",
                                )
                            ],
                        ),
                    ],
                )
                (root / "confirm.json").write_text(json.dumps(spec))
                cli(
                    [
                        "prepare",
                        str(root / "confirm.json"),
                        "--out",
                        str(root / "prepared"),
                    ]
                )
                models = {m.identity:m for m in load_prediction_models(root / "trained/models")}
                frozen = json.loads((root / "trained/direction.json").read_text())
                model = models[frozen["model_identity"]]
                additive = models[frozen.get("additive_model_identity", frozen["model_identity"])]
                # Convert actual learned native weights to raw dosage units.
                coef = np.zeros((m, 2))
                lookup = {v: j for j, v in enumerate(axis.ids)}
                for component, fitted in enumerate((additive,model)):
                    for j, v in enumerate(fitted.variants.ids):
                        coef[lookup[v],component] = fitted.weights[j,component] * fitted.scale.inverse_scale[j]
                if np.any(coef[:ma, 1] != 0):
                    raise ArithmeticError("cis weight leaked into trans direction")
                for method in ("learned", "burden"):
                    summary = load_robust_scores(
                        root / f"prepared/{method}.robust-score.npz"
                    )
                    # Public burden uses study HWE units; learned score retains
                    # native training units. Either truth uses the frozen axis.
                    study_mean = raw[n0:].mean(0)
                    study_inv = 1 / np.sqrt(study_mean * (1 - study_mean / 2))
                    weights = (
                        coef[ma:, 1]
                        if method == "learned"
                        else study_inv[ma:] / np.sqrt(mb)
                    )
                    e = gb @ weights
                    ec = e - e.mean()
                    variance = np.mean(ec * ec)
                    conversion = inv[0] / study_inv[0]
                    truth = {s: 0.0 for s in settings}
                    for setting, strength, h in [
                        ("aligned_weak", 0.035, aligned),
                        ("aligned_moderate", 0.075, aligned),
                        ("sparse_moderate", 0.075, sparse),
                        ("distributed_moderate", 0.075, distributed),
                    ]:
                        truth[setting] = (
                            conversion
                            * strength
                            * np.mean(ec * (h - h.mean()))
                            / variance
                        )
                    truth["nonlinear_scale"] = (
                        conversion
                        * 0.6
                        * np.mean(xa[:, 0] * (additive_a - additive_a.mean()))
                        / np.mean(xa[:, 0] ** 2)
                        * np.mean(ec * (additive_b - additive_b.mean()))
                        / variance
                    )
                    for j, setting in enumerate(settings):
                        fit = robust_score_tests(summary, trait=j)
                        b = fit["beta"][0]
                        se = fit["standard_errors"][0]
                        records.append(
                            dict(
                                setting=setting,
                                method=method,
                                replicate=rep,
                                p=fit["kernel_p"],
                                failed=False,
                                beta0=b,
                                se0=se,
                                truth0=truth[setting],
                                coverage=float(
                                    abs(b - truth[setting]) <= 1.95996398454 * se
                                ),
                                outside_scope=";".join(
                                    summary.metadata["outside_confirmation_design"]
                                ),
                            )
                        )
                # Saved-summary fitting and completed restart exercise actual API.
                if rep == 0:
                    cli(
                        [
                            "fit",
                            str(root / "prepared/learned.robust-score.npz"),
                            "--out",
                            str(root / "fit.json"),
                        ]
                    )
                    cli(
                        [
                            "prepare",
                            str(root / "confirm.json"),
                            "--out",
                            str(root / "prepared"),
                            "--resume",
                        ]
                    )
                if rep < args.nested_models:
                    nested.extend(
                        population_resampling(
                            rng,
                            ga,
                            gb,
                            xa,
                            xb,
                            additive_a,
                            additive_b,
                            coef,
                            args.test_samples,
                            args.nested_draws,
                            rep,
                        )
                    )
            except (ValueError, ArithmeticError, RuntimeError) as error:
                for method in ("learned", "burden"):
                    for setting in settings:
                        if not any(
                            r["replicate"] == rep
                            and r["method"] == method
                            and r["setting"] == setting
                            for r in records
                        ):
                            records.append(
                                dict(
                                    setting=setting,
                                    method=method,
                                    replicate=rep,
                                    failed=True,
                                    p=np.nan,
                                    error=str(error),
                                )
                            )
            finally:
                # The copied dosage coefficients outlive these model maps.
                # Release every alias before NFS temporary-directory cleanup.
                models = model = additive = fitted = None
            if (rep + 1) % 10 == 0:
                print(
                    "completed",
                    rep + 1,
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    table = reduction(records, args.out)
    pd.DataFrame(nested).to_csv(args.out / "population_resampling.csv", index=False)
    (args.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    panel=meta,
                    arguments={
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    fixed="donor genotype block distributions, causal effects, externally supplied hypotheses, ridge penalties",
                    regenerated="independent genotype block draws and residuals in training/confirmation; native direction refitted each outer replicate",
                    nested="fixed native training and effects; BOTH confirmation genotype blocks and phenotype errors redrawn, not conditional fixed-G residual draws",
                    truth="exact conditional learned-score population coefficient from full empirical block covariances; converted to actual public target and score units",
                    assumptions="independent chromosome blocks by construction; no claim that original cohort chromosomes are independent",
                    seconds=time.perf_counter() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                )
            ),
            indent=2,
        )
    )
    print(
        table[
            [
                "setting",
                "method",
                "rejection",
                "bias",
                "mean_se",
                "coverage",
                "failures",
            ]
        ].to_string(index=False)
    )


def population_resampling(rng, ga, gb, xa, xb, mean_a, mean_b, coef, n, draws, outer):
    """Cheap random-design null fits conditional on one learned model."""
    ea = ga @ coef[: ga.shape[1], 0]
    eb = gb @ coef[ga.shape[1] :, 0]
    direction = gb @ coef[ga.shape[1] :, 1]
    records = []
    for rep in range(draws):
        ia = rng.integers(len(ga), size=n)
        ib = rng.integers(len(gb), size=n)
        x = xa[ia, 0]
        e = direction[ib]
        c = np.column_stack(
            [np.ones(n), xa[ia, :4], ga[ia, :4] == 1, xb[ib, 0], ea[ia] + eb[ib], e]
        )
        mean = mean_a[ia] + mean_b[ib]
        sd = np.sqrt(0.4 + 0.6 * x * x)
        y = np.column_stack(
            [
                mean + sd * rng.normal(size=n),
                mean + sd * rng.standard_t(5, size=n) / np.sqrt(5 / 3),
            ]
        )
        try:
            s = prepare_robust_scores(
                (x * e)[:, None],
                y,
                c,
                feature_names=("direction",),
                trait_names=("gaussian", "t5"),
                metadata={},
                sampling_model="iid_population_projection",
            )
            for j, label in enumerate(s.trait_names):
                f = robust_score_tests(s, trait=j)
                records.append(
                    dict(
                        outer=outer,
                        replicate=rep,
                        noise=label,
                        p=f["kernel_p"],
                        beta=f["beta"][0],
                        se=f["standard_errors"][0],
                        failed=False,
                        outside_scope=";".join(
                            s.metadata["outside_confirmation_design"]
                        ),
                    )
                )
        except (ValueError, ArithmeticError) as error:
            for label in ("gaussian", "t5"):
                records.append(
                    dict(
                        outer=outer,
                        replicate=rep,
                        noise=label,
                        p=np.nan,
                        failed=True,
                        error=str(error),
                    )
                )
    return records


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--background-chromosome", default="2")
    p.add_argument("--genotype-seed", type=int, default=395741)
    p.add_argument("--seed", type=int, default=168953)
    p.add_argument("--training-samples", type=int, default=1024)
    p.add_argument("--test-samples", type=int, default=4096)
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--nested-models", type=int, default=3)
    p.add_argument("--nested-draws", type=int, default=2000)
    p.add_argument("--omit-causal-main", action="store_true")
    a = p.parse_args()
    if not 1 <= a.replicates <= 100:
        raise ValueError("at most 100 native training fits per setting")
    run(a)


if __name__ == "__main__":
    main()
