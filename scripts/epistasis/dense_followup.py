"""Independent dense-imputation follow-up of prespecified bounded local pair fixtures.

Select a hidden causal variant using discovery GENOTYPES only to create an
imperfect-tagging stress experiment. No phenotype target-selection method is
implemented. Every eligible dense local main effect is restored, not just the
known simulated causal variant. Participant-level arrays are never persisted.
"""
import argparse, json, time, resource
from pathlib import Path
import numpy as np
from summit.prediction.genotype import FileGenotypeSource
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.score import prepare_linear_scores, linear_score_tests
from summit.epistasis.cli import _jsonable
from scripts.epistasis.robust_validation import reduction


def read(source, rows, variants):
    source.prepare(rows, 128, 1)
    raw = np.column_stack(
        [
            source.read(variants[j : j + 128]).copy()
            for j in range(0, len(variants), 128)
        ]
    ).astype(float)
    raw[raw == -127] = np.nan
    return raw


def mean_terms(raw):
    mean = np.nanmean(raw, axis=0)
    g = np.where(np.isnan(raw), mean, raw)
    h = (raw == 1).astype(float)
    for j in range(raw.shape[1]):
        h[np.isnan(raw[:, j]), j] = np.mean(h[~np.isnan(raw[:, j]), j])
    return np.column_stack([g, h])


def supported_tile(ga, gd, apos, dpos, phase, center, width):
    """Phenotype-blind fixture admission; retain all eligible loci in a tile."""
    failures = []
    for left in range(center - 1000000, center + 1000000, width):
        a = np.flatnonzero((apos >= left) & (apos < left + width))
        d = np.flatnonzero((dpos >= left) & (dpos < left + width))
        if len(a) < 2 or len(d) < 3:
            continue
        ca = np.column_stack([np.ones(len(ga)), mean_terms(ga[:, a])])
        full = np.column_stack([ca, mean_terms(gd[:, d])])
        valid = True
        for mask, c in ((phase, ca), (~phase, ca), (~phase, full)):
            u = thin_rank_revealing_fixed_effect_basis(c[mask])
            if u.shape[1] / mask.sum() > 0.05 or np.max(np.sum(u * u, axis=1)) > 0.08:
                valid = False
                break
        if valid:
            return a, d, left, failures
        failures.append(left)
    return None, None, None, failures


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--array", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument(
        "--tile-bp",
        type=int,
        default=0,
        help="optional phenotype-blind bounded tile admission within each fixed 2-Mb region",
    )
    p.add_argument(
        "--inference",
        choices=["robust_mean", "linear_exact", "both"],
        default="robust_mean",
    )
    args = p.parse_args()
    if not 1 <= args.replicates <= 100:
        raise ValueError("at most 100 replicates")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(508624)
    records = []
    panels = []
    n0, n1 = 2048, 4096
    with FileGenotypeSource(
        args.array, genome_build="GRCh37"
    ) as array, FileGenotypeSource(args.dense, genome_build="GRCh37") as dense:
        dense_rows = {v: i for i, v in enumerate(dense.samples)}
        common = [i for i, v in enumerate(array.samples) if v in dense_rows]
        if len(common) < n0 + n1:
            raise ValueError("insufficient aligned unrelated samples")
        chosen = rng.choice(common, n0 + n1, replace=False)
        # Native readers permit arbitrary sample order, but monotonically
        # ordered row reads are mapped back to the frozen discovery/test order.
        ar = np.sort(chosen)
        dr = np.array([dense_rows[array.samples[i]] for i in ar])
        do = np.argsort(dr)
        dr_sorted = dr[do]
        undo = np.argsort(do)
        phase = np.isin(ar, chosen[:n0])
        apos = np.asarray(array.variants.position)
        dpos = np.asarray(dense.variants.position)
        achr = np.asarray(array.variants.chromosome)
        for center in (10000000, 50000000, 150000000):
            ai = np.flatnonzero((achr == "1") & (abs(apos - center) <= 1000000))
            di = np.flatnonzero(abs(dpos - center) <= 1000000)
            if len(ai) < 2 or not len(di):
                raise ValueError("empty prespecified local region")
            ga = read(array, ar, ai)
            gd = read(dense, dr_sorted, di)[undo]
            af = np.nanmean(ga[phase], axis=0) / 2
            df = np.nanmean(gd[phase], axis=0) / 2
            ak = (af > 0.15) & (af < 0.85) & (np.isnan(ga).mean(axis=0) < 0.02)
            dk = (df > 0.15) & (df < 0.85) & (np.isnan(gd).mean(axis=0) < 0.02)
            print(
                "regional QC",
                center,
                "array",
                len(ai),
                int(ak.sum()),
                "dense",
                len(di),
                int(dk.sum()),
                "dense AF range",
                float(np.nanmin(df)),
                float(np.nanmax(df)),
                flush=True,
            )
            ga = ga[:, ak]
            gd = gd[:, dk]
            ai = ai[ak]
            di = di[dk]
            fixture_info = {}
            if args.tile_bp:
                aa, dd, left, rejected = supported_tile(
                    ga, gd, apos[ai], dpos[di], phase, center, args.tile_bp
                )
                fixture_info = dict(
                    tile_bp=args.tile_bp,
                    tile_start=left,
                    genotype_only_rejected_tiles=rejected,
                    admission="first genomic tile with >=2 typed and >=3 dense common loci, nuisance rank/N<=.05 and max leverage<=.08 in both sample phases; no outcomes used",
                )
                if aa is None:
                    for rep in range(args.replicates):
                        records.append(
                            dict(
                                setting=str(center),
                                method="unidentifiable_region",
                                replicate=rep,
                                p=np.nan,
                                failed=True,
                                error="no tile passes frozen genotype-only support rule",
                            )
                        )
                    panels.append(dict(center=center, **fixture_info))
                    continue
                ga = ga[:, aa]
                gd = gd[:, dd]
                ai = ai[aa]
                di = di[dd]
            if len(ai) < 2 or len(di) < 3:
                for rep in range(args.replicates):
                    records.append(
                        dict(
                            setting=str(center),
                            method="unidentifiable_region",
                            replicate=rep,
                            p=np.nan,
                            failed=True,
                            error="insufficient common variants after genotype-only QC",
                        )
                    )
                panels.append(
                    dict(
                        center=center,
                        window_bp=2000000,
                        array_variants=len(ai),
                        dense_variants=len(di),
                        failure="insufficient genotype support",
                    )
                )
                continue
            # First supported nonredundant pair by genomic order, independent
            # of phenotypes. If the fixed rule fails, retain the failed region.
            ca = np.column_stack([np.ones(len(ar)), mean_terms(ga)])
            ud = thin_rank_revealing_fixed_effect_basis(ca[phase])
            pair = None
            for j in range(1, min(12, len(ai))):
                raw_pair = np.prod(mean_terms(ga)[:, [0, j]], axis=1)
                residual = raw_pair[phase] - ud @ (ud.T @ raw_pair[phase])
                if (
                    np.sum(residual**2) > 1e-4
                    and 1 / np.sum((residual / np.linalg.norm(residual)) ** 4) > 100
                ):
                    pair = (0, j)
                    f = raw_pair[:, None]
                    r = residual
                    break
            if pair is None:
                for rep in range(args.replicates):
                    records.append(
                        dict(
                            setting=str(center),
                            method="unidentifiable_region",
                            replicate=rep,
                            p=np.nan,
                            failed=True,
                            error="no pair meets fixed genotype support rule",
                        )
                    )
                continue
            hidden = ~np.isin(dpos[di], apos[ai])
            if not hidden.any():
                raise ValueError(
                    "dense region supplies no hidden common causal variants"
                )
            imputed = mean_terms(gd)[:, : len(di)]
            residual_hidden = imputed[phase] - ud @ (ud.T @ imputed[phase])
            correlation = (r @ residual_hidden) / (
                np.linalg.norm(r)
                * np.maximum(np.linalg.norm(residual_hidden, axis=0), 1e-30)
            )
            candidate = int(np.argmax(np.where(hidden, abs(correlation), -1)))
            generating = imputed[:, candidate] - imputed[:, candidate].mean()
            # One fixed causal effect, no interactions; all full-dense main
            # effects enter follow-up without revealing candidate to the fit.
            y = generating[:, None] + rng.normal(size=(len(ar), args.replicates))
            full = np.column_stack([ca, mean_terms(gd)])
            diagnostics = {}
            procedures = (
                [("HC3", prepare_robust_scores, robust_score_tests)]
                if args.inference == "robust_mean"
                else [("linear_exact", prepare_linear_scores, linear_score_tests)]
            )
            if args.inference == "both":
                procedures = [
                    ("HC3", prepare_robust_scores, robust_score_tests),
                    ("linear_exact", prepare_linear_scores, linear_score_tests),
                ]
            definitions = [
                (name + "_" + method, mask, c, prepare, fit_fn)
                for name, mask, c in [
                    ("discovery_array", phase, ca),
                    ("confirmation_array", ~phase, ca),
                    ("confirmation_dense", ~phase, full),
                ]
                for method, prepare, fit_fn in procedures
            ]
            for name, mask, c, prepare, fit_fn in definitions:
                ucheck = thin_rank_revealing_fixed_effect_basis(c[mask])
                rr = f[mask] - ucheck @ (ucheck.T @ f[mask])
                rr -= ucheck @ (ucheck.T @ rr)
                hc = np.sum(ucheck*ucheck,axis=1)
                partial = (rr[:,0]**2) / np.sum(rr**2)
                near = hc+partial >= 1-1e-8
                diagnostics[name] = dict(fixed_rank=ucheck.shape[1], interaction_energy=float(np.sum(rr**2)),
                    near_unit_rows=int(near.sum()), maximum_partial_leverage=float(partial.max()),
                    maximum_partial_on_near_unit=float(partial[near].max()) if near.any() else 0.,
                    effective_support=float(1/np.sum(partial**2)), max_nuisance_leverage=float(hc.max()))
                try:
                    s = prepare(
                        f[mask],
                        y[mask],
                        c[mask],
                        feature_names=("pair",),
                        trait_names=tuple(map(str, range(args.replicates))),
                        metadata={},
                    )
                    diagnostics[name].update({
                        k: s.metadata.get(k)
                        for k in (
                            "fixed_rank",
                            "max_leverage",
                            "minimum_feature_effective_support",
                            "outside_confirmation_design",
                            "nuisance_saturated_rows",
                            "max_active_leverage",
                        )
                    })
                    for rep in range(args.replicates):
                        fit = fit_fn(s, trait=rep)
                        if name.endswith("linear_exact"):
                            fit = dict(
                                fit,
                                beta=fit["joint_beta"],
                                standard_errors=fit["joint_beta_se"],
                                beta_interval_95=fit["joint_beta_interval_95"],
                            )
                        records.append(
                            dict(
                                setting=str(center),
                                method=name,
                                replicate=rep,
                                p=fit["kernel_p"],
                                failed=False,
                                beta0=fit["beta"][0],
                                truth0=0.0,
                                se0=fit["standard_errors"][0],
                                coverage=float(
                                    fit["beta_interval_95"][0][0]
                                    <= 0
                                    <= fit["beta_interval_95"][0][1]
                                ),
                            )
                        )
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    for rep in range(args.replicates):
                        records.append(
                            dict(
                                setting=str(center),
                                method=name,
                                replicate=rep,
                                p=np.nan,
                                failed=True,
                                error=str(error),
                            )
                        )
            panels.append(
                dict(
                    center=center,
                    window_bp=2000000,
                    array_variants=len(ai),
                    dense_variants=len(di),
                    discovery_n=n0,
                    confirmation_n=n1,
                    causal_position=int(dpos[di[candidate]]),
                    supplied_pair_positions=[int(apos[ai[i]]) for i in pair],
                    discovery_conditional_causal_product_correlation=float(
                        correlation[candidate]
                    ),
                    diagnostics=diagnostics,
                    **fixture_info,
                )
            )
        source_info = dict(
            array_identity=array.identity,
            dense_identity=dense.identity,
            common_samples=len(common),
            dense_total_variants=len(dense.variants.ids),
            dense_total_samples=len(dense.samples),
        )
    table = reduction(records, args.out)
    (args.out / "design.json").write_text(
        json.dumps(
            _jsonable(
                dict(
                    sources=source_info,
                    panels=panels,
                    seed=508624,
                    replicates=args.replicates,
                    fixed="genotypes, target pairs, causal identities and additive effect; residuals regenerated; main and HC3 refitted",
                    fixture_selection="first numerically supported typed pair; hidden causal variant maximizes genotype-only partial product correlation in discovery; not a target-discovery or real-phenotype selection workflow",
                    inference="biological absence of epistasis; discovery/array confirmation can have nonzero misspecified marker coefficients; full dense model contains causal main effect by restoring all eligible variants",
                    family="three supplied regions; assess rejection at .05/3 for family claims, .05 is a diagnostic",
                    inference_methods=args.inference,
                    noise="independent Gaussian with constant variance; linear_exact is a separate correctly specified comparator, not a data-selected fallback",
                    seconds=time.perf_counter() - start,
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
