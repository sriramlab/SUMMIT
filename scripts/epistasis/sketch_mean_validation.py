"""Native fixed pair sketches: fixed phenotypes/varying draws and vice versa."""
import argparse, json, time
from pathlib import Path
import numpy as np
from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis
from summit.epistasis.prepare import SelectedStudy, fit_scale
from summit.epistasis.features import prepare_feature_reference
from summit.epistasis.models import annotation_weights, target_design
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable
from scripts.epistasis.robust_validation import load_panel, reduction


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--real-genotypes", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(771296)
    x, d, meta, raw = load_panel(a.real_genotypes, 581307, 2048, 4096, return_raw=True)
    n = len(x)
    raw = raw[:, :64]
    ids = tuple(f"v{i}" for i in range(64))
    source = ArrayGenotypeSource(
        raw,
        [(str(i), str(i)) for i in range(n)],
        VariantAxis(ids, ("1",) * 64, tuple(range(1, 65)), ("A",) * 64, ("G",) * 64),
        hard_calls=True,
    )
    scale = fit_scale(source, np.arange(n))
    annotations = annotation_weights(
        ids,
        dict(A={f"v{i}": 1 for i in range(4)}, B={f"v{i}": 1 for i in range(4, 12)}),
    )
    design = target_design(
        source,
        np.arange(n),
        scale,
        components=[],
        annotations=annotations,
        additive_annotations=["all"],
        local_variants=ids[:24],
        dominance_variants=ids[:24],
        allow_additive_only=True,
    )
    study = SelectedStudy(
        source,
        np.arange(n),
        scale,
        **design,
        backend="native",
        block_size=64,
        memory_bytes=2**30,
    )
    job = dict(
        id="cross",
        groups=[dict(name="cross", mode="cross", left="A", right="B")],
        additive_annotations=["all"],
    )
    exact = prepare_feature_reference(study, job, annotations, main_effects="declared")
    c = exact.fixed_effects
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis

    u = thin_rank_revealing_fixed_effect_basis(c)
    f = exact.features
    pf = f - u @ (u.T @ f)
    effect = rng.normal(size=f.shape[1])
    effect *= np.sqrt(0.005 / np.mean((pf @ effect) ** 2))
    mean = x[:, :24] @ rng.normal(size=24) / np.sqrt(24) + 0.5 * d[:, 0]
    noise = np.sqrt(0.3 + 0.7 * x[:, 0] ** 2)[:, None] * rng.normal(size=(n, 100))
    records = []
    errors = []
    for dimensions in (None, 8, 16):
        for bank in range(1 if dimensions is None else 16):
            ref = (
                exact
                if dimensions is None
                else prepare_feature_reference(
                    study,
                    job,
                    annotations,
                    main_effects="declared",
                    sketch_dimensions=dimensions,
                    seed=84317,
                    bank=bank,
                )
            )
            ff = ref.features
            rf = ff - u @ (u.T @ ff)
            # Kernel Frobenius error without allocating an N-square matrix.
            cross = np.sum((rf.T @ pf) ** 2)
            size = np.sum((pf.T @ pf) ** 2)
            error = np.sqrt(
                max(0.0, np.sum((rf.T @ rf) ** 2) + size - 2 * cross) / size
            )
            errors.append(
                dict(dimensions=dimensions, bank=bank, relative_kernel_error=error)
            )
            for signal in (False, True):
                Y = mean[:, None] + noise + (f @ effect)[:, None] * signal
                setting = f'{dimensions or "exact"}_bank{bank}_' + (
                    "signal005" if signal else "null_heterogeneous"
                )
                try:
                    s = prepare_robust_scores(
                        ff,
                        Y,
                        c,
                        feature_names=ref.metadata["feature_names"],
                        trait_names=tuple(map(str, range(100))),
                        metadata={},
                    )
                    for rep in range(100):
                        fit = robust_score_tests(s, trait=rep)
                        records.append(
                            dict(
                                setting=setting,
                                method="HC3_kernel",
                                replicate=rep,
                                p=fit["kernel_p"],
                                failed=False,
                            )
                        )
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    for rep in range(100):
                        records.append(
                            dict(
                                setting=setting,
                                method="HC3_kernel",
                                replicate=rep,
                                p=np.nan,
                                failed=True,
                                error=str(error),
                            )
                        )
    reduction(records, a.out)
    (a.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    panel=meta,
                    errors=errors,
                    seconds=time.perf_counter() - start,
                    seed=771296,
                    sketch_seed=84317,
                    banks=16,
                    replicates=100,
                    signal_variance=0.005,
                    fixed="same genotypes, realized effects and 100 residual outcomes for every bank/dimension; mean and HC3 refitted; no reference trace probes",
                    interpretation="conditional fixed-feature tests; varying banks changes alternatives and power, not just an estimated normal matrix. Banks share phenotypes and cannot be counted as independent replication. No selection of the smallest P value.",
                )
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print("native sketch comparison completed", time.perf_counter() - start)


if __name__ == "__main__":
    main()
