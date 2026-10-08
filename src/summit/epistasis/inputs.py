"""Genotype-supported supplied-target input construction without causal labels.

Only input preparation is performed. Statistical validity still requires the
assumptions of the selected finite mean or population projection analysis.
"""
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from summit.prediction.genotype import FileGenotypeSource, native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.cli import _table
from summit.prediction._validation import closed
from .cli import _jsonable


def build_trans_inputs(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    if not 0 < a.minimum_cell_fraction < 1 / 3 or a.local_window_bp < 1:
        raise ValueError(
            "positive local window and genotype-cell fraction in (0,1/3) required"
        )
    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    covariates = _table(a.covariates)
    outcomes = _table(a.phenotypes)
    names = (
        a.covariate_columns.split(",")
        if a.covariate_columns
        else list(covariates.columns)
    )
    structure = a.structure_covariates.split(",") if a.structure_covariates else []
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - set(covariates.columns)
        or len(set(structure)) != len(structure)
        or set(structure) - set(names)
        or a.phenotype_column not in outcomes
    ):
        raise ValueError("declare distinct available phenotype/covariate columns")
    with FileGenotypeSource(a.genotypes, genome_build=a.genome_build) as source:
        if getattr(a,'content_identity',False):
            source.authenticate_content()
        if not source.hard_calls:
            raise ValueError(
                "local dominance input preparation currently requires hard-call genotypes"
            )
        target = source.variants.ids.index(a.target)
        chromosome = np.asarray(source.variants.chromosome)
        from .trans import select_trans_background
        background, background_record = select_trans_background(
            source, a.target, chromosome=getattr(a, "background_chromosome", None),
            variants=getattr(a, "interaction_variants", None),
        )
        lookup = {pair: i for i, pair in enumerate(source.samples)}
        supplied = {
            label: list(_table(path).index)
            for label, path in [
                ("training", a.training_samples),
                ("confirmation", a.confirmation_samples),
            ]
        }
        if set(supplied["training"]) & set(supplied["confirmation"]):
            raise ValueError("training and confirmation sample lists overlap")
        rows, counts = {}, {}
        for label, samples in supplied.items():
            if set(samples) - lookup.keys():
                raise ValueError(f"{label} contains unknown genotype sample IDs")
            ordered = sorted(samples, key=lookup.__getitem__)
            selected = np.array([lookup[s] for s in ordered])
            aligned_cov = covariates.reindex(pd.MultiIndex.from_tuples(ordered))[
                names
            ].to_numpy(float)
            aligned_y = outcomes.reindex(pd.MultiIndex.from_tuples(ordered))[
                a.phenotype_column
            ].to_numpy(float)
            source.prepare(selected, 1, a.num_threads)
            target_complete = source.read(np.array([target]))[:, 0] != -127
            finite_cov = np.isfinite(aligned_cov).all(1)
            finite_y = np.isfinite(aligned_y)
            keep = target_complete & finite_cov & finite_y
            rows[label] = selected[keep]
            if not keep.any():
                raise ValueError(f"{label} has no complete analysis participants")
            counts[label] = dict(
                supplied=len(selected),
                retained=int(keep.sum()),
                missing_target=int((~target_complete).sum()),
                missing_covariates=int((~finite_cov).sum()),
                missing_phenotype=int((~finite_y).sum()),
            )
            pd.DataFrame(
                [source.samples[i] for i in rows[label]], columns=["FID", "IID"]
            ).to_csv(out / f"{label}.tsv", sep="\t", index=False)
        pos = np.asarray(source.variants.position)
        local = np.flatnonzero(
            (chromosome == chromosome[target])
            & (abs(pos - pos[target]) <= a.local_window_bp)
        )
        source.prepare(rows["training"], 128, a.num_threads)
        selected = []
        for lo in range(0, len(local), 128):
            indices = local[lo : lo + 128]
            raw = source.read(indices)
            cells = np.stack([(raw == i).sum(0) for i in (0, 1, 2)])
            selected.extend(
                indices[cells.min(0) >= a.minimum_cell_fraction * len(rows["training"])]
            )
        if target not in selected:
            raise ValueError("target lacks the prespecified training genotype support")
        # Freeze numerical units in training. Affine changes preserve the
        # finite mean span. Random genetic slopes additionally have a declared
        # origin: centering z changes diag(z) K diag(z) by additive/slope cross
        # terms absent from a diagonal component model. Rescale those columns
        # without moving the supplied origin.
        train_index = pd.MultiIndex.from_tuples(
            [source.samples[i] for i in rows["training"]]
        )
        training_cov = covariates.reindex(train_index)[names].to_numpy(float)
        center, spread = training_cov.mean(0), training_cov.std(0)
        if np.any(spread <= 0) or not np.all(np.isfinite(spread)):
            raise ValueError("remove covariates constant in the training sample")
        preserved_origins=(structure if getattr(a,'inference_method','robust_mean')
            =='conditional_polygenic_mean' else [])
        for name in preserved_origins:
            center[names.index(name)]=0.
        combined = np.sort(np.concatenate(list(rows.values())))
        cov_index = pd.MultiIndex.from_tuples(
            [source.samples[i] for i in combined], names=["FID", "IID"]
        )
        standardized = (
            covariates.reindex(cov_index)[names].to_numpy(float) - center
        ) / spread
        pd.DataFrame(standardized, index=cov_index, columns=names).reset_index().to_csv(
            out / "covariates.tsv", sep="\t", index=False
        )
        local_ids = [source.variants.ids[i] for i in selected]
        (out / "variants.txt").write_text(
            "\n".join(v for i, v in enumerate(source.variants.ids) if i != target)
            + "\n"
        )
        (out / "background.txt").write_text(
            "\n".join(source.variants.ids[i] for i in background) + "\n"
        )
        model_id = "target_" + re.sub("[^a-zA-Z0-9_]", "_", a.target)
        geno = dict(geno=str(Path(a.genotypes).resolve()), genome_build=a.genome_build)
        if getattr(a,'content_identity',False):
            geno['content_identity']=True
        cv = dict(file="covariates.tsv", columns=names)
        if structure:
            cv["varying_effects"] = structure
        pheno = dict(
            file=str(a.phenotypes.resolve()), column=a.phenotype_column, unit=a.unit
        )
        train = dict(
            kind="summit.epistasis.train_direction",
            schema_version=1,
            id=model_id,
            genotypes=geno,
            samples="training.tsv",
            phenotype=pheno,
            covariates=cv,
            target=a.target,
            variants="variants.txt",
            interaction_variants="background.txt",
            trans_only=True,
            local_variants=local_ids,
            dominance_variants=local_ids,
            prior=dict(additive=0.5, interaction=0.05, residual=1.0),
            storage="packed",
            solver=dict(rtol=1e-6, max_iterations=200),
        )
        prepare = dict(
            kind="summit.epistasis.prepare",
            schema_version=1,
            genotypes=geno,
            samples="confirmation.tsv",
            covariates=cv,
            phenotypes=dict(
                file=pheno["file"], columns=[a.phenotype_column], unit=a.unit
            ),
            annotations=dict(target={a.target: 1}),
            frozen_scores=[
                dict(name="additive", direction="trained/direction.json", component=0),
                dict(
                    name="interaction", direction="trained/direction.json", component=1
                ),
            ],
            jobs=[
                dict(
                    id=model_id,
                    trans_target=a.target,
                    additive_annotations=["all"],
                    adjust_scores=["additive"],
                    local_variants=local_ids,
                    dominance_variants=local_ids,
                    components=[
                        dict(
                            name="learned",
                            frozen_score="interaction",
                            background="target",
                        )
                    ],
                    inference=dict(
                        method="robust_mean",
                        main_effects="declared",
                        save_reference=True,
                        sampling_model="iid_population_projection",
                    ),
                )
            ],
        )
        if getattr(a,'inference_method','robust_mean')=='conditional_polygenic_mean':
            prepare=dict(kind='summit.epistasis.prepare',schema_version=1,
                training='train.json',samples='confirmation.tsv',
                phenotypes=dict(file=pheno['file'],columns=[a.phenotype_column],unit=a.unit),
                directions=[dict(phenotype=a.phenotype_column,direction='trained/direction.json')],
                inference=dict(method='conditional_polygenic_mean'))
            if getattr(a,'moment_weighting','none')!='none':
                prepare['inference']['moment_weighting']=a.moment_weighting
        record = dict(
            source_identity=source.identity,
            variant_axis_identity=source.variants.identity,
            target=a.target,
            background_chromosome=getattr(a, "background_chromosome", None),
            interaction_background=background_record,
            counts=counts,
            local_window_bp=a.local_window_bp,
            minimum_training_cell_fraction=a.minimum_cell_fraction,
            local_variants=local_ids,
            available_markers=len(source.variants.ids),
            training_markers=len(source.variants.ids) - 1,
            covariate_scaling=dict(
                source=str(a.covariates.resolve()),
                columns=names,
                center=center.tolist(),
                scale=spread.tolist(),
                estimated_in="retained training participants",
                preserved_origins=preserved_origins,
            ),
            selection="supplied targets and split; complete cases; genotype-only training support; no outcome values used for local selection",
            scientific_assumptions="independent frozen training; adequate declared finite mean or justified separable population null; independent participants and sufficient support",
        )
        if getattr(a,'inference_method','robust_mean')=='conditional_polygenic_mean':
            record['scientific_assumptions']='declared finite mean and Gaussian additive, dominance and covariate-dependent genetic covariance; independent individual residuals; no cross-locus interaction under the null'
            record['status']='research conditional mean procedure; full-marker calibration and power not yet qualified'
        for name, value in [
            ("train.json", train),
            ("prepare.json", prepare),
            ("inputs.json", record),
        ]:
            with (out / name).open("x") as handle:
                json.dump(_jsonable(value), handle, indent=2, allow_nan=False)
    print(json.dumps(record["counts"], sort_keys=True))
    return train, prepare


def prepare_inputs(manifest, output, *, threads=1):
    """Read a portable recipe and write cohort-side train/prepare manifests."""
    path = Path(manifest).resolve()
    root = path.parent
    spec = json.loads(path.read_text())
    closed(
        spec,
        (
            "kind",
            "schema_version",
            "genotypes",
            "target",
            "training_samples",
            "confirmation_samples",
            "phenotype",
            "covariates",
        ),
        ("local_window_bp", "minimum_cell_fraction", "background_chromosome", "interaction_variants", "inference"),
        name="trans input preparation",
    )
    if spec["kind"] != "summit.epistasis.trans_inputs" or spec["schema_version"] != 1:
        raise ValueError("unsupported trans input preparation manifest")
    if ("background_chromosome" in spec) == ("interaction_variants" in spec):
        raise ValueError("supply exactly one background chromosome or interaction variant file")
    genotype, phenotype, covariates = (
        spec[k] for k in ("genotypes", "phenotype", "covariates")
    )
    closed(genotype, ("geno", "genome_build"), ('content_identity',), name="input genotypes")
    if type(genotype.get('content_identity',False)) is not bool:
        raise ValueError('content_identity must be boolean')
    inference=spec.get('inference',dict(method='robust_mean'))
    closed(inference,('method',),('moment_weighting',),name='input inference')
    if inference['method'] not in ('robust_mean','conditional_polygenic_mean'):
        raise ValueError('unsupported supplied-trans input procedure')
    weighting=inference.get('moment_weighting','none')
    if (weighting not in ('none','genotype_diagonal')
            or ('moment_weighting' in inference and inference['method']!='conditional_polygenic_mean')):
        raise ValueError('covariance moment weighting requires the conditional polygenic procedure')
    closed(phenotype, ("file", "column", "unit"), name="input phenotype")
    closed(
        covariates, ("file", "columns"), ("varying_effects",), name="input covariates"
    )
    for names in (covariates["columns"], covariates.get("varying_effects", [])):
        if not isinstance(names, list) or any(
            not isinstance(n, str) or not n or "," in n for n in names
        ):
            raise ValueError("input covariate columns must be named lists")
    if (
        not covariates["columns"]
        or not isinstance(phenotype["unit"], str)
        or not phenotype["unit"]
    ):
        raise ValueError("input covariates and phenotype units must be declared")
    return build_trans_inputs(
        SimpleNamespace(
            out=Path(output),
            genotypes=str(root / genotype["geno"]),
            genome_build=genotype["genome_build"],
            content_identity=genotype.get('content_identity',False),
            inference_method=inference['method'],
            moment_weighting=weighting,
            target=spec["target"],
            background_chromosome=str(spec["background_chromosome"]) if "background_chromosome" in spec else None,
            interaction_variants=root / spec["interaction_variants"] if "interaction_variants" in spec else None,
            training_samples=root / spec["training_samples"],
            confirmation_samples=root / spec["confirmation_samples"],
            phenotypes=root / phenotype["file"],
            phenotype_column=phenotype["column"],
            unit=phenotype["unit"],
            covariates=root / covariates["file"],
            covariate_columns=",".join(covariates["columns"]),
            structure_covariates=",".join(covariates.get("varying_effects", [])),
            local_window_bp=spec.get("local_window_bp", 100000),
            minimum_cell_fraction=spec.get("minimum_cell_fraction", 0.02),
            num_threads=threads,
        )
    )
