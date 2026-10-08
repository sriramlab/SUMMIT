"""Matched native learning on intact real rows; bounded validation, not discovery.

All generating functions and strengths are frozen on the donor distribution.
Known means enter only diagnostic truth, never the fitted response or covariance.
Population draws preserve complete genotype/covariate/missingness rows. Fixed
panels instead condition on disjoint original participants and regenerate errors.
"""
import argparse
import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed
from scipy.stats import beta as beta_dist, norm

from summit.epistasis.cli import main as cli, _jsonable
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.prediction.artifacts import load_prediction_models
from summit.prediction.genotype import (
    FileGenotypeSource,
    StandardizedBlock,
    native_module,
)


def intact_panel(path, covariates, target, background_chromosome, seed, sizes=(16, 64)):
    """Phenotype-blind bounded panel; never independently filter chromosome rows."""
    rng = np.random.default_rng(seed)
    cv = pd.read_csv(covariates, sep=r"\s+", dtype={"FID": str, "IID": str})
    cv = cv.set_index(["FID", "IID"])
    if not cv.index.is_unique:
        raise ValueError("duplicate covariate participants")
    with FileGenotypeSource(path, genome_build="GRCh37") as source:
        present = cv.reindex(pd.MultiIndex.from_tuples(source.samples)).to_numpy(float)
        rows = np.flatnonzero(np.isfinite(present).all(1))
        cov = present[rows]
        axis = source.variants
        chrom, position = target.split(":")
        pos = np.asarray(axis.position)
        chr_ = np.asarray(axis.chromosome)
        if background_chromosome == chrom:
            raise ValueError("background must be trans")
        candidates = np.flatnonzero(
            (chr_ == chrom) & (abs(pos - int(position)) < 2000000)
        )
        candidates = candidates[np.argsort(abs(pos[candidates] - int(position)))]
        distal = np.flatnonzero(chr_ == background_chromosome)
        distal = rng.permutation(distal)
        source.prepare(rows, 128, 1)
        blocks, indices = [], []
        for pool, size in ((candidates, sizes[0]), (distal, sizes[1])):
            kept, values = [], []
            for begin in range(0, len(pool), 128):
                take = pool[begin : begin + 128]
                order = np.argsort(take)
                raw = source.read(take[order]).astype(float)[:, np.argsort(order)]
                observed = raw != -127
                means = np.where(observed, raw, 0).sum(0) / observed.sum(0)
                good = (means > 0.2) & (means < 1.8) & (observed.mean(0) > 0.99)
                for j in np.flatnonzero(good):
                    kept.append(take[j])
                    values.append(raw[:, j])
                    if len(kept) == size:
                        break
                if len(kept) == size:
                    break
            if len(kept) != size:
                raise ValueError(
                    "insufficient supported markers at prespecified anchor"
                )
            blocks.append(np.column_stack(values))
            indices.extend(kept)
        raw = np.column_stack(blocks)
        complete_target = raw[:, 0] != -127
        raw, cov = raw[complete_target], cov[complete_target]
        selected_axis = axis.subset(indices)
        cov_mean, cov_sd = cov.mean(0), cov.std(0)
        if np.any(cov_sd == 0):
            raise ValueError("constant supplied covariate")
        cov = (cov - cov_mean) / cov_sd
        metadata = dict(
            source_identity=source.identity,
            anchor=target,
            actual_target=selected_axis.ids[0],
            variants=list(selected_axis.ids),
            local_size=sizes[0],
            donor_count=len(raw),
            covariates=list(cv.columns),
            complete_target_exclusions=int((~complete_target).sum()),
            background_missing_fraction=float(np.mean(raw[:, 1:] == -127)),
            row_selection="complete aligned covariates and target only; every selected row stays intact",
            marker_selection="nearest supported target/local markers; seeded whole-chromosome background; MAF>.1, missing<.01",
        )
        return raw, cov, selected_axis, metadata


def coordinates(raw):
    observed = raw != -127
    mean = np.where(observed, raw, 0).sum(0) / observed.sum(0)
    inv = 1 / np.sqrt(mean * (1 - mean / 2))
    standard = StandardizedBlock(native_module(), 1)
    x = standard.prepare(
        np.asfortranarray(raw, dtype=np.int8),
        np.arange(len(raw)),
        np.arange(raw.shape[1]),
        mean,
        inv,
    ).copy()
    d = (raw == 1).astype(float)
    d = np.where(observed, d, (d.sum(0) / observed.sum(0))[None, :])
    return x, d, mean, inv


def architecture(x, d, cov, ma, seed):
    rng = np.random.default_rng(seed)
    n, m = x.shape
    pc = cov[:, 5]  # PC1 in the authenticated input table
    burden = x[:, ma:].sum(1) / np.sqrt(m - ma)
    directions = dict(
        aligned=burden, sparse=x[:, ma], mixed=x[:, ma:] @ rng.normal(size=m - ma)
    )
    directions = {k: (v - v.mean()) / v.std() for k, v in directions.items()}
    dense = x @ rng.normal(size=m)
    dense *= np.sqrt(0.8 / dense.var())
    sparse = x[:, [5, 12, ma + 3, ma + 8]] @ np.array([0.4, 0.8, -0.5, 0.4])
    finite = 0.7 * x[:, 0] + 0.4 * d[:, 0] + 0.25 * pc
    local = dense + 1.3 * x[:, 12]
    dominance = dense + 0.8 * d[:, 12] + 0.8 * d[:, ma + 3]
    structure = local + 0.6 * pc * x[:, 12] + 0.4 * pc * burden
    settings = {
        k: dict(mean=v, signal=np.zeros(n), biological_null=True)
        for k, v in dict(
            finite=finite,
            dense=dense,
            sparse_null=sparse,
            local_withheld=local,
            dominance=dominance,
            structure=structure,
            heavy=local,
        ).items()
    }
    for name, label, variance in (
        ("aligned_weak", "aligned", 0.005),
        ("aligned", "aligned", 0.02),
        ("sparse", "sparse", 0.02),
        ("mixed", "mixed", 0.02),
    ):
        signal = x[:, 0] * directions[label]
        strength = np.sqrt(variance / signal.var())
        signal = strength * signal
        settings[name] = dict(
            mean=local + signal,
            signal=signal,
            biological_null=False,
            oracle=directions[label],
            strength=strength,
            reference_signal_variance=variance,
        )
    settings["scale"] = dict(
        mean=local + 0.3 * local**2,
        signal=0.3 * local**2,
        biological_null=False,
        oracle=directions["aligned"],
        truth="nonlinear transformation of conditional mean; observed-scale projection, not biological-null calibration",
    )
    for value in settings.values():
        value.setdefault("oracle", directions["aligned"])
        value.setdefault("strength", 0.0)
        value.setdefault("reference_signal_variance", float(value["signal"].var()))
    return settings, burden


def frozen_scores(model, raw, axis):
    """Use the native shared scaler, including each fitted missing-value mean."""
    lookup = {v: i for i, v in enumerate(axis.ids)}
    indices = np.array([lookup[v] for v in model.variants.ids])
    standard = StandardizedBlock(native_module(), 1)
    z = standard.prepare(
        np.asfortranarray(raw[:, indices], dtype=np.int8),
        np.arange(len(raw)),
        np.arange(len(indices)),
        model.scale.mean,
        model.scale.inverse_scale,
    ).copy()
    scores = np.empty((len(raw), 2), order="F")
    native_module().prediction_product(
        np.asfortranarray(z), np.asfortranarray(model.weights), scores, False, 1
    )
    return scores


def frozen_main_coordinates(model, raw, axis, x, means, inv):
    """Match production's main_imputation in the common donor affine units.

    Otherwise an almost redundant score can isolate a handful of missing calls,
    creating high leverage in a complete single-locus nuisance design.
    """
    lookup = {v: i for i, v in enumerate(axis.ids)}
    result = x.copy()
    for j, v in enumerate(model.variants.ids):
        k = lookup[v]
        result[raw[:, k] == -127, k] = (model.scale.mean[j] - means[k]) * inv[k]
    return result


def nuisance(x, d, cov, pgs, e, burden, ma, adjustment):
    local = 4 if adjustment == "pgs" else ma
    columns = [np.ones(len(x)), cov, x[:, :local], d[:, :local], x[:, ma], pgs, e]
    if adjustment == "supplied":
        # An explicit finite single-locus mean, not an unrestricted function of G.
        columns.extend(
            [x[:, ma:], d[:, ma:], cov[:, 5, None] * x[:, :ma], cov[:, 5] * burden]
        )
    return np.column_stack(columns)


def projection_truth(f, c, mean):
    """Independent diagnostic FWL reference on the complete declared law."""
    r = f - c @ np.linalg.lstsq(c, f, rcond=1e-11)[0]
    return np.linalg.lstsq(r, mean, rcond=1e-11)[0]


def frozen_population(rng, f, c, mean, x, n, draws, truth, *, heavy=False):
    """New intact donor rows and outcomes, conditional on ONE trained score."""
    records = []
    for draw in range(draws):
        ids = rng.integers(len(x), size=n)
        noise = rng.standard_t(5, n) / np.sqrt(5 / 3) if heavy else rng.normal(size=n)
        y = mean[ids] + np.sqrt(0.4 + 0.6 * x[ids, 0] ** 2) * noise
        row = dict(
            replicate=draw,
            failed=True,
            p=np.nan,
            estimate=np.nan,
            truth=float(truth[0]),
            se=np.nan,
        )
        try:
            summary = prepare_robust_scores(
                f[ids],
                y,
                c[ids],
                feature_names=("learned",),
                trait_names=("y",),
                metadata={},
                sampling_model="iid_population_projection",
            )
            b = float(summary.scores[0, 0] / summary.information[0, 0])
            se = float(
                np.sqrt(summary.score_covariance[0, 0, 0]) / summary.information[0, 0]
            )
            row.update(
                failed=False,
                p=float(2 * norm.sf(abs(b / se))),
                estimate=b,
                se=se,
                coverage=float(abs(b - truth[0]) <= norm.isf(0.025) * se),
                alignment_squared=np.nan,
                outside_scope=";".join(summary.metadata["outside_confirmation_design"]),
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            row["error"] = str(error)
        records.append(row)
    return records


def binomial_interval(k, n):
    if not n:
        return [None, None]
    return [
        float(beta_dist.ppf(0.025, k, n - k + 1)) if k else 0.0,
        float(beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
    ]


def frozen_fixed(rng, f, c, mean, x, draws, truth, *, heavy=False):
    """Batched production HC3 on one fixed panel, with each nuisance refitted."""
    records = []
    for start in range(0, draws, 128):
        batch = min(128, draws - start)
        noise = (
            rng.standard_t(5, (len(x), batch)) / np.sqrt(5 / 3)
            if heavy
            else rng.normal(size=(len(x), batch))
        )
        y = mean[:, None] + np.sqrt(0.4 + 0.6 * x[:, 0, None] ** 2) * noise
        try:
            summary = prepare_robust_scores(
                f,
                y,
                c,
                feature_names=("learned",),
                trait_names=tuple(str(i) for i in range(batch)),
                metadata={},
            )
            b = summary.scores[0] / summary.information[0, 0]
            se = np.sqrt(summary.score_covariance[:, 0, 0]) / summary.information[0, 0]
            records.extend(
                dict(
                    replicate=start + i,
                    failed=False,
                    p=float(2 * norm.sf(abs(b[i] / se[i]))),
                    estimate=float(b[i]),
                    truth=float(truth[0]),
                    se=float(se[i]),
                    coverage=float(abs(b[i] - truth[0]) <= norm.isf(0.025) * se[i]),
                    alignment_squared=np.nan,
                    outside_scope=";".join(
                        summary.metadata["outside_confirmation_design"]
                    ),
                )
                for i in range(batch)
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            records.extend(
                dict(
                    replicate=start + i,
                    failed=True,
                    p=np.nan,
                    estimate=np.nan,
                    truth=float(truth[0]),
                    se=np.nan,
                    error=str(error),
                )
                for i in range(batch)
            )
    return records


def reduce_records(records, out, *, summary_name="summary.csv", write_replicates=True):
    frame = pd.DataFrame(records)
    if write_replicates:
        frame.to_csv(out / "replicates.csv", index=False)
    summaries = []
    keys = ["sampling", "setting", "adjustment", "method"]
    for key, group in frame.groupby(keys, dropna=False):
        valid = group[~group.failed]
        n = len(valid)
        row = dict(
            zip(keys, key),
            scheduled=len(group),
            valid=n,
            failures=int(group.failed.sum()),
            unsupported=int(valid.outside_scope.fillna("").ne("").sum()),
        )
        if n:
            for alpha in (0.05, 0.005, 0.0005):
                k = int((valid.p < alpha).sum())
                row[f"rejection_{alpha}"] = k / n
                row[f"rejection_{alpha}_interval"] = json.dumps(binomial_interval(k, n))
                row[f"rejection_{alpha}_upper95"] = (
                    float(beta_dist.ppf(0.95, k + 1, n - k)) if k < n else 1.0
                )
            scalar = valid.dropna(subset=["estimate", "truth", "se"])
            if len(scalar):
                error = scalar.estimate - scalar.truth
                coverage = np.asarray(scalar.coverage, dtype=float)
                row.update(
                    bias=float(error.mean()),
                    error_sd=float(error.std(ddof=1)),
                    mean_se=float(scalar.se.mean()),
                    rms_se=float(np.sqrt(np.mean(scalar.se**2))),
                    zero_truth_rms=float(np.sqrt(np.mean(scalar.truth**2))),
                    coverage=float(coverage.mean()),
                    coverage_interval=json.dumps(
                        binomial_interval(int(coverage.sum()), len(scalar))
                    ),
                    alignment_squared=float(scalar.alignment_squared.mean()),
                )
        summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(out / summary_name, index=False)
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    args.out.chmod(0o700)
    start, cpu = time.perf_counter(), time.process_time()
    raw, cov, axis, meta = intact_panel(
        args.genotypes,
        args.covariates,
        args.target,
        args.background_chromosome,
        args.panel_seed,
    )
    x, d, means, inv = coordinates(raw)
    ma = meta["local_size"]
    settings, burden = architecture(x, d, cov, ma, args.architecture_seed)
    selected = args.settings.split(",")
    if set(selected) - settings.keys():
        raise ValueError("unknown setting")
    adjustments = args.adjustments.split(",")
    if set(adjustments) - {"pgs", "local", "supplied"}:
        raise ValueError("unknown adjustment")
    n0, n1 = args.training_samples, args.test_samples
    if "fixed" in args.sampling.split(",") and n0 + n1 > len(raw):
        raise ValueError("fixed panel requires disjoint original donors")
    fixed = (
        np.random.default_rng(args.panel_seed + 1).choice(
            len(raw), n0 + n1, replace=False
        )
        if "fixed" in args.sampling.split(",")
        else None
    )
    plan = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        panel=meta,
        generating_strengths={
            k: {
                field: settings[k][field]
                for field in (
                    "strength",
                    "reference_signal_variance",
                    "biological_null",
                )
            }
            for k in selected
        },
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target_scale="fixed empirical donor HWE; frozen native learned-score units",
        training="same mean/effect sizes/noise in train and test, fresh full native training per setting/replicate",
        missingness="target complete; observed background patterns retained; generating mean uses fixed donor imputation; prediction and fitted constituent main effects share frozen training imputation",
        withheld="local causal column 12 excluded from native predictor, present only in richer confirmation nuisance",
        truth="diagnostic projection of known mean on each learned design; fixed confirmation rows or complete empirical donor law",
        tolerances="see benchmarks/epistasis/intact_decisions_20261003.md; no threshold changes",
    )
    (args.out / "design.json").write_text(json.dumps(_jsonable(plan), indent=2))
    rng = np.random.default_rng(args.seed)
    records, nested = [], []
    for sampling in args.sampling.split(","):
        if sampling not in ("fixed", "population"):
            raise ValueError("unknown sampling design")
        for rep in range(args.replicates):
            donors = (
                fixed if sampling == "fixed" else rng.integers(len(raw), size=n0 + n1)
            )
            train, test = donors[:n0], donors[n0:]
            with tempfile.TemporaryDirectory(prefix="intact-", dir=args.scratch) as tmp:
                root = Path(tmp)
                ids = list(map(str, range(n0 + n1)))
                to_bed(
                    root / "g.bed",
                    np.where(raw[donors] == -127, np.nan, raw[donors]),
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
                (root / "train").write_text(
                    "FID IID\n" + "".join(f"{i} {i}\n" for i in range(n0))
                )
                (root / "variants").write_text(
                    "\n".join(v for j, v in enumerate(axis.ids) if j not in (0, 12))
                    + "\n"
                )
                (root / "interaction").write_text("\n".join(axis.ids[ma:]) + "\n")
                cv = pd.DataFrame(
                    cov[donors], columns=[f"c{i}" for i in range(cov.shape[1])]
                )
                cv.insert(0, "IID", ids)
                cv.insert(0, "FID", ids)
                cv.to_csv(root / "cov", sep="\t", index=False)
                for setting in selected:
                    definition = settings[setting]
                    sd = np.sqrt(0.4 + 0.6 * x[donors, 0] ** 2)
                    noise = (
                        rng.standard_t(5, len(donors)) / np.sqrt(5 / 3)
                        if setting == "heavy"
                        else rng.normal(size=len(donors))
                    )
                    y = definition["mean"][donors] + sd * noise
                    ph = pd.DataFrame(dict(FID=ids, IID=ids, y=y))
                    ph.to_csv(root / f"y_{setting}", sep="\t", index=False)
                    manifest = dict(
                        kind="summit.epistasis.train_direction",
                        schema_version=1,
                        genotypes=dict(geno="g.bed", genome_build="GRCh37"),
                        samples="train",
                        phenotype=dict(
                            file=f"y_{setting}",
                            column="y",
                            unit="fixed simulation units",
                        ),
                        covariates=dict(file="cov", columns=list(cv.columns[2:])),
                        target=axis.ids[0],
                        variants="variants",
                        interaction_variants="interaction",
                        trans_only=True,
                        local_variants=list(axis.ids[:4]),
                        dominance_variants=list(axis.ids[:4]),
                        prior=dict(additive=0.5, interaction=0.05, residual=1.0),
                        storage="packed",
                        solver=dict(rtol=1e-8, max_iterations=250),
                    )
                    (root / "train.json").write_text(json.dumps(manifest))
                    trained = root / setting
                    training_error = None
                    try:
                        cli(
                            [
                                "train-direction",
                                str(root / "train.json"),
                                "--out",
                                str(trained),
                            ]
                        )
                        models = {m.identity:m for m in load_prediction_models(trained / "models")}
                        frozen = json.loads((trained / "direction.json").read_text())
                        model = models[frozen["model_identity"]]
                        model_identity = model.identity
                        additive = models[frozen.get("additive_model_identity", frozen["model_identity"])]
                        scores = frozen_scores(model, raw, axis)
                        if additive.identity != model.identity:
                            scores[:, 0] = frozen_scores(additive, raw, axis)[:, 0]
                        main_x = frozen_main_coordinates(
                            model, raw, axis, x, means, inv
                        )
                    except (ValueError, ArithmeticError, RuntimeError) as error:
                        training_error = str(error)
                    finally:
                        # Scores/coordinates own their arrays. Release the
                        # read-only model maps before TemporaryDirectory exits
                        # (open mapped files cannot be unlinked cleanly on NFS).
                        models = model = additive = None
                    methods = getattr(
                        args, "methods", "learned,burden,oracle,joint"
                    ).split(",")
                    if set(methods) - {"learned", "burden", "oracle", "joint"}:
                        raise ValueError("unknown direction comparison")
                    for adjustment in adjustments:
                        for method in methods:
                            row = dict(
                                sampling=sampling,
                                setting=setting,
                                replicate=rep,
                                adjustment=adjustment,
                                method=method,
                                biological_null=definition["biological_null"],
                                reference_signal_variance=definition[
                                    "reference_signal_variance"
                                ],
                                realized_train_signal_variance=float(
                                    definition["signal"][train].var()
                                ),
                                realized_test_signal_variance=float(
                                    definition["signal"][test].var()
                                ),
                                estimate=np.nan,
                                truth=np.nan,
                                se=np.nan,
                                failed=True,
                                p=np.nan,
                            )
                            try:
                                if training_error:
                                    raise RuntimeError(training_error)
                                e = (
                                    scores[:, 1]
                                    if method == "learned"
                                    else burden
                                    if method == "burden"
                                    else definition["oracle"]
                                )
                                if method == "joint":
                                    e = np.column_stack([scores[:, 1], burden])
                                if e.ndim == 1:
                                    e = e[:, None]
                                f = x[:, 0, None] * e
                                c = nuisance(
                                    main_x,
                                    d,
                                    cov,
                                    scores[:, 0],
                                    e,
                                    burden,
                                    ma,
                                    adjustment,
                                )
                                law = (
                                    test if sampling == "fixed" else np.arange(len(raw))
                                )
                                truth = projection_truth(
                                    f[law], c[law], definition["mean"][law]
                                )
                                summary = prepare_robust_scores(
                                    f[test],
                                    y[n0:],
                                    c[test],
                                    feature_names=tuple(
                                        f"f{j}" for j in range(f.shape[1])
                                    ),
                                    trait_names=("y",),
                                    metadata={},
                                    sampling_model="iid_population_projection"
                                    if sampling == "population"
                                    else "fixed_design_correct_mean",
                                )
                                result = robust_score_tests(summary)
                                row.update(
                                    failed=False,
                                    p=result["joint_p"]
                                    if method == "joint"
                                    else result["kernel_p"],
                                    max_leverage=summary.metadata["max_leverage"],
                                    fixed_rank=summary.metadata["fixed_rank"],
                                    feature_support=summary.metadata[
                                        "minimum_feature_effective_support"
                                    ],
                                    outside_scope=";".join(
                                        summary.metadata["outside_confirmation_design"]
                                    ),
                                )
                                if method != "joint":
                                    b, se = float(result["beta"][0]), float(
                                        result["standard_errors"][0]
                                    )
                                    # A conditional-panel projection under misspecification is only a diagnostic target.
                                    row.update(
                                        estimate=b,
                                        truth=float(truth[0]),
                                        se=se,
                                        coverage=float(
                                            abs(b - truth[0]) <= norm.isf(0.025) * se
                                        ),
                                        alignment_squared=float(
                                            np.corrcoef(e[:, 0], definition["oracle"])[
                                                0, 1
                                            ]
                                            ** 2
                                        ),
                                    )
                                if (
                                    sampling == "population"
                                    and method == "learned"
                                    and rep < getattr(args, "nested_models", 0)
                                    and setting
                                    in getattr(args, "nested_settings", "").split(",")
                                ):
                                    inner_rng = np.random.default_rng(
                                        args.seed
                                        + 10000003
                                        + rep * 13249
                                        + selected.index(setting) * 739
                                    )
                                    inner_sampling = getattr(
                                        args, "nested_sampling", "population"
                                    )
                                    if inner_sampling == "fixed":
                                        inner_truth = projection_truth(
                                            f[test], c[test], definition["mean"][test]
                                        )
                                        inner = frozen_fixed(
                                            inner_rng,
                                            f[test],
                                            c[test],
                                            definition["mean"][test],
                                            x[test],
                                            args.nested_draws,
                                            inner_truth,
                                            heavy=setting == "heavy",
                                        )
                                    else:
                                        inner = frozen_population(
                                            inner_rng,
                                            f,
                                            c,
                                            definition["mean"],
                                            x,
                                            n1,
                                            args.nested_draws,
                                            truth,
                                            heavy=setting == "heavy",
                                        )
                                    nested.extend(
                                        dict(
                                            r,
                                            sampling=f"{inner_sampling}_frozen_model_{rep}",
                                            setting=setting,
                                            adjustment=adjustment,
                                            method=method,
                                            model_identity=model_identity,
                                        )
                                        for r in inner
                                    )
                            except (
                                ValueError,
                                ArithmeticError,
                                RuntimeError,
                                np.linalg.LinAlgError,
                            ) as error:
                                row["error"] = str(error)
                            records.append(row)
            if (rep + 1) % 10 == 0 or rep + 1 == args.replicates:
                print(
                    sampling, rep + 1, round(time.perf_counter() - start, 2), flush=True
                )
    table = reduce_records(records, args.out)
    if nested:
        pd.DataFrame(nested).to_csv(args.out / "frozen_replicates.csv", index=False)
        reduce_records(
            nested, args.out, summary_name="frozen_summary.csv", write_replicates=False
        )
    (args.out / "resources.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                native_fits=args.replicates
                * len(selected)
                * len(args.sampling.split(",")),
                source_genotype_traversals="one selected-marker read per source block; per-replicate native scale/pack/cache passes",
                rows_preserved=True,
            ),
            indent=2,
        )
    )
    print(
        table[
            [
                "sampling",
                "setting",
                "adjustment",
                "method",
                "rejection_0.05",
                "failures",
            ]
        ].to_string(index=False)
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--covariates", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--target", default="12:66358347")
    p.add_argument("--background-chromosome", default="5")
    p.add_argument("--panel-seed", type=int, default=73451)
    p.add_argument("--architecture-seed", type=int, default=379417)
    p.add_argument("--seed", type=int, default=153973)
    p.add_argument("--training-samples", type=int, default=2048)
    p.add_argument("--test-samples", type=int, default=4096)
    p.add_argument("--replicates", type=int, default=12)
    p.add_argument(
        "--settings",
        default="finite,dense,sparse_null,local_withheld,dominance,structure,heavy,aligned_weak,aligned,sparse,mixed,scale",
    )
    p.add_argument("--sampling", default="fixed,population")
    p.add_argument("--adjustments", default="pgs,local,supplied")
    p.add_argument("--methods", default="learned,burden,oracle,joint")
    p.add_argument("--nested-models", type=int, default=0)
    p.add_argument("--nested-draws", type=int, default=2000)
    p.add_argument("--nested-settings", default="finite,local_withheld,heavy")
    p.add_argument(
        "--nested-sampling", choices=("population", "fixed"), default="population"
    )
    args = p.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("use 1..100 independent full learning replicates")
    if not 0 <= args.nested_models <= args.replicates or args.nested_draws < 1:
        raise ValueError(
            "nested models must be within full replicates and draws positive"
        )
    run(args)


if __name__ == "__main__":
    main()
