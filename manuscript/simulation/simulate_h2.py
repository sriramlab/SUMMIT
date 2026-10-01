#!/usr/bin/env python3
"""Simulate the continuous, binary, enrichment, and MAF–LD h² benchmarks."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed

from _common import (
    genotype_blocks,
    ldak_weights,
    new_directory,
    read_annotations,
    read_maf_ld,
    read_samples,
    standardize,
    write_phenotype,
    write_settings,
)

SCENARIOS = {"weak": (1.5, 0.75), "moderate": (2.0, 0.5), "strong": (3.0, 0.3)}


def read_parameters(args, bins):
    values = dict(
        p_causal=args.p_causal,
        maf_ex=args.maf_ex,
        ld_ex=args.ld_ex,
        min_maf=args.min_maf,
        max_maf=args.max_maf,
    )
    if args.params:
        lines = [
            line.split()
            for line in args.params.read_text().splitlines()
            if line.strip()
        ]
        if len(lines) != 3 or len(lines[0]) != len(lines[1]):
            raise ValueError(
                "Parameter files need a header, values, and per-bin variances."
            )
        raw = dict(zip(lines[0], lines[1]))
        required = {*values, "total_h2"}
        if not required <= raw.keys():
            raise ValueError(f"Missing parameters: {sorted(required - raw.keys())}")
        # The manuscript files use bin-specific targets and Bernoulli causal sampling.
        optional = {
            "alloc_mode": "bin_fixed",
            "weight_mode": "legacy",
            "causal_sampling": "bernoulli",
            "effect_weighting": "weighted",
        }
        for key, expected in optional.items():
            if key in raw and raw[key] != expected:
                raise ValueError(f"Unsupported manuscript setting: {key}={raw[key]}")
        values.update({key: float(raw[key]) for key in required})
        targets = np.array(lines[2], dtype=float)
        replicates = (
            args.num_reps
            if args.num_reps is not None
            else int(raw.get("num_simul", 100))
        )
    else:
        if args.sigma is None and args.model != "enrichment":
            raise ValueError("Supply --sigma or --params.")
        targets = (
            None if args.sigma is None else np.array(args.sigma.split(","), dtype=float)
        )
        values["total_h2"] = (
            args.total_h2 if args.model == "enrichment" else float(targets.sum())
        )
        replicates = args.num_reps if args.num_reps is not None else 100
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Simulation parameters must be finite.")
    if not 0 <= values["total_h2"] <= 1 or not 0 < values["p_causal"] <= 1:
        raise ValueError("h² must lie in [0, 1] and the causal proportion in (0, 1].")
    if not 0 <= values["min_maf"] <= values["max_maf"] <= 0.5:
        raise ValueError("MAF bounds must satisfy 0 <= min_maf <= max_maf <= 0.5.")
    if targets is not None:
        if (
            len(targets) != bins
            or not np.all(np.isfinite(targets))
            or np.any(targets < 0)
        ):
            raise ValueError(
                "Supply one finite, nonnegative variance per annotation bin."
            )
        if not np.isclose(targets.sum(), values["total_h2"], rtol=1e-6, atol=1e-10):
            raise ValueError("Per-bin variances must sum to total_h2.")
    if replicates <= 0:
        raise ValueError("The number of replicates must be positive.")
    return values, targets, replicates


def power_weights(features, maf_ex, ld_ex):
    maf, ld = features[:, 0], features[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        w_maf = np.power(maf, maf_ex) if maf_ex != 0 else np.ones_like(maf)
        w_ld = np.power(ld, ld_ex) if ld_ex != 0 else np.ones_like(ld)
        weights = w_maf * w_ld
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Effect-variance weights must be finite and nonnegative.")
    return weights


def enrichment_targets(a, features, params, scenario, architecture):
    if a.shape[1] != 2:
        raise ValueError("Enrichment simulation needs two columns: target, rest.")
    params["maf_ex"], params["ld_ex"] = (
        (0.0, 0.0) if architecture == "gcta" else (0.75, -1.0)
    )
    weights = power_weights(features, params["maf_ex"], params["ld_ex"])
    eligible = (features[:, 2] >= params["min_maf"]) & (
        features[:, 2] <= params["max_maf"]
    )
    mass = np.array([weights[(a[:, b] == 1) & eligible].sum() for b in range(2)])
    enrichment = np.array(SCENARIOS[scenario], dtype=float)
    if float((enrichment * mass).sum()) <= 0:
        raise ValueError("No weighted variants in the selected MAF interval.")
    return params["total_h2"] * (enrichment * mass) / float((enrichment * mass).sum())


def draw_causals(
    candidates, probability, rng, *, skip_full=False, weighted_fallback=None
):
    if not len(candidates):
        raise ValueError("A positive variance target has no eligible variants.")
    if skip_full and probability >= 1:
        return candidates
    selected = candidates[rng.random(len(candidates)) < probability]
    if not len(selected):
        if weighted_fallback is not None:
            q = weighted_fallback[candidates].astype(float, copy=True)
            p = None if float(q.sum()) <= 0 else q / float(q.sum())
            selected = rng.choice(candidates, size=1, replace=False, p=p)
        elif skip_full:
            selected = rng.choice(candidates, size=1, replace=False)
        else:
            selected = candidates[rng.integers(0, len(candidates), size=1)]
    return selected


def effect_variances(causal, weights, target):
    denominator = weights[causal].sum()
    if denominator <= 0:
        return np.full(len(causal), target / len(causal))
    return target * (weights[causal] / denominator)


def draw_trait_effects(a, features, params, targets, replicates, rng, model):
    """Draw effects before the genotype pass for continuous and binary benchmarks."""
    weights = power_weights(features, params["maf_ex"], params["ld_ex"])
    eligible = (features[:, 2] >= params["min_maf"]) & (
        features[:, 2] <= params["max_maf"]
    )
    candidates = [np.flatnonzero((a[:, b] == 1) & eligible) for b in range(a.shape[1])]
    dtype = np.float64 if model == "enrichment" else np.float32
    beta = np.zeros((a.shape[0], replicates), dtype=dtype)
    oracle = np.zeros((replicates, a.shape[1]))
    counts = np.zeros(replicates, dtype=int)
    for r in range(replicates):
        sigma = np.zeros(a.shape[0])
        for b, target in enumerate(targets):
            if target <= 0:
                continue
            causal = draw_causals(candidates[b], params["p_causal"], rng)
            variance = effect_variances(causal, weights, target)
            sigma[causal] = np.sqrt(variance)
            counts[r] += len(causal)
            oracle[r, b] = variance.sum()
            if model == "binary":
                # Binary simulations draw each bin's effects immediately after its causal set.
                beta[causal, r] = rng.normal(0, sigma[causal], size=len(causal)).astype(
                    np.float32
                )
        if model != "binary":
            active = sigma > 0
            if active.any():
                beta[active, r] = rng.normal(0, sigma[active]).astype(dtype)
    return beta, oracle, counts


def trait_genetic_values(bed_path, beta, rng, model, chunk_size, rep_batch, max_mem):
    with open_bed(str(bed_path)) as bed:
        n = bed.iid_count
        if max_mem:
            x = standardize(
                bed.read(dtype="float64"), rng, missing_only=model == "enrichment"
            )
            if model == "enrichment":
                return np.column_stack(
                    [x.dot(beta[:, r]) for r in range(beta.shape[1])]
                )
            return x.dot(beta)
    values = np.zeros((n, beta.shape[1]))
    for start, end, x in genotype_blocks(bed_path, chunk_size):
        standardize(x, rng)
        for r0 in range(0, beta.shape[1], rep_batch):
            r1 = min(r0 + rep_batch, beta.shape[1])
            values[:, r0:r1] += x.dot(beta[start:end, r0:r1])
    return values


def maf_ld_variances(a, features, params, targets, replicates, seed, architecture):
    """Allocate GCTA variance by bin, or LDAK variance across the selected MAF stratum."""
    bin_id = a.argmax(axis=1)
    eligible = (features[:, 2] >= params["min_maf"]) & (
        features[:, 2] <= params["max_maf"]
    )
    weights = (
        ldak_weights(features)
        if architecture == "ldak"
        else power_weights(features, params["maf_ex"], params["ld_ex"])
    )
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    variance = np.zeros((replicates, len(a)), dtype=np.float32)
    if architecture == "ldak":
        candidates = np.flatnonzero(eligible & (targets[bin_id] > 0) & (weights > 0))
        for r in range(replicates):
            causal = draw_causals(candidates, params["p_causal"], rng, skip_full=True)
            # Keep the arithmetic order used by the LDAK benchmark.
            variance[r, causal] = (
                params["total_h2"] * weights[causal] / float(weights[causal].sum())
            )
    else:
        candidates = [
            np.flatnonzero((bin_id == b) & eligible) for b in range(a.shape[1])
        ]
        for r in range(replicates):
            for b, target in enumerate(targets):
                if target > 0:
                    causal = draw_causals(
                        candidates[b],
                        params["p_causal"],
                        rng,
                        skip_full=True,
                        weighted_fallback=weights,
                    )
                    denominator = float(weights[causal].sum())
                    variance[r, causal] = (
                        float(target) / len(causal)
                        if denominator <= 0
                        else float(target) * weights[causal] / denominator
                    )
    return variance


def maf_ld_genetic_values(
    bed_path, variance, seed, architecture, chunk_size, rep_batch, dtype
):
    sigma = np.sqrt(variance).astype(np.float32, copy=False)
    _, beta_seed, environment_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(beta_seed)
    with open_bed(str(bed_path)) as bed:
        values = np.zeros((bed.iid_count, len(sigma)), dtype=dtype)
    for start, end, block in genotype_blocks(bed_path, chunk_size, "float32"):
        x = standardize(block.astype(dtype, copy=False))
        for r0 in range(0, len(sigma), rep_batch):
            r1 = min(r0 + rep_batch, len(sigma))
            sd = sigma[r0:r1, start:end].T
            if architecture == "ldak":
                # The LDAK runs draw effects only for variants active in this batch.
                active = np.any(sd != 0, axis=1)
                if not active.any():
                    continue
                sd = sd[active].astype(dtype, copy=False)
                effects = rng.standard_normal(sd.shape).astype(dtype, copy=False)
                effects *= sd
                values[:, r0:r1] += x[:, active] @ effects
            else:
                effects = rng.standard_normal(sd.shape).astype(dtype, copy=False)
                effects *= sd.astype(dtype, copy=False)
                values[:, r0:r1] += x @ effects
    return values, np.random.default_rng(environment_seed)


def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (Acklam approximation).
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        raise ValueError("p must be in (0,1)")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return float(num / den)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return float(num / den)
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return float(num / den)


def ascertain_case_control(
    is_case: np.ndarray,
    case_frac: float,
    n_target: int | None,
    rng,
) -> tuple[np.ndarray, int, int]:
    """
    Choose an ascertained subset to match target case fraction.
    Returns (keep_mask, n_case_sel, n_ctrl_sel).
    """
    if not (0.0 < case_frac < 1.0):
        raise ValueError("--case-frac must be in (0,1).")

    cases = np.where(is_case)[0]
    ctrls = np.where(~is_case)[0]
    nc_av, nn_av = cases.size, ctrls.size
    if nc_av == 0 or nn_av == 0:
        raise ValueError(
            f"Degenerate population labels: cases={nc_av}, controls={nn_av}. "
            f"Increase n or adjust prevalence."
        )

    if n_target is None:
        n_max = int(np.floor(min(nc_av / case_frac, nn_av / (1.0 - case_frac))))
        if n_max < 2:
            raise ValueError(
                f"Not enough samples to form ascertained set: n_max={n_max}. "
                f"(cases={nc_av}, controls={nn_av}, case_frac={case_frac})"
            )
        n_case = int(np.round(case_frac * n_max))
        n_case = max(1, min(n_case, nc_av))
        n_ctrl = n_max - n_case
        if n_ctrl > nn_av:
            n_ctrl = nn_av
            n_case = int(np.floor((case_frac / (1.0 - case_frac)) * n_ctrl))
            n_case = max(1, min(n_case, nc_av))
            n_max = n_case + n_ctrl
        if n_case + n_ctrl < 2:
            raise ValueError(
                "Ascertainment produced <2 samples; adjust prevalence/case-frac."
            )
    else:
        if n_target < 2:
            raise ValueError("--n-sample must be >= 2.")
        n_case = int(np.round(case_frac * n_target))
        n_case = max(1, min(n_case, n_target - 1))
        n_ctrl = n_target - n_case
        if n_case > nc_av or n_ctrl > nn_av:
            raise ValueError(
                f"Requested n-sample={n_target} with case-frac={case_frac} "
                f"needs (cases={n_case}, ctrls={n_ctrl}) but available "
                f"(cases={nc_av}, ctrls={nn_av})."
            )

    sel_cases = rng.choice(cases, size=n_case, replace=False)
    sel_ctrls = rng.choice(ctrls, size=n_ctrl, replace=False)
    keep = np.zeros(is_case.size, dtype=bool)
    keep[sel_cases] = True
    keep[sel_ctrls] = True
    return keep, int(n_case), int(n_ctrl)


def binary_phenotype(liability, args, rng):
    cases = liability > norm_ppf(1 - args.prevalence)
    if not cases.any():
        cases[np.argmax(liability)] = True
    elif cases.all():
        cases[np.argmin(liability)] = False
    keep = None
    if args.sampling == "case-control":
        keep, _, _ = ascertain_case_control(cases, args.case_frac, args.n_sample, rng)
    y = cases.astype(float) + (1 if args.coding == "12" else 0)
    if keep is not None:
        y[~keep] = -9 if args.coding == "12" else np.nan
    return y, keep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["continuous", "binary", "enrichment", "maf-ld"],
        default="continuous",
    )
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--annot", type=Path, required=True)
    parser.add_argument("--mafld", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--sigma", help="Comma-separated per-bin genetic variances.")
    targets.add_argument(
        "--params", type=Path, help="Three-line manuscript parameter file."
    )
    parser.add_argument("--p-causal", type=float, default=0.5)
    parser.add_argument("--maf-ex", type=float, default=0)
    parser.add_argument("--ld-ex", type=float, default=0)
    parser.add_argument("--min-maf", type=float, default=0)
    parser.add_argument("--max-maf", type=float, default=0.5)
    parser.add_argument(
        "--num-reps", type=int, help="Overrides the parameter file; otherwise 100."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Variants per block; default 8192 for MAF–LD, 10000 otherwise.",
    )
    parser.add_argument(
        "--rep-batch",
        type=int,
        help="Replicates per multiplication; see README for benchmark defaults.",
    )
    parser.add_argument(
        "--max-mem",
        action="store_true",
        help="Read the whole genotype matrix for binary or enrichment simulations.",
    )
    parser.add_argument(
        "--acc-dtype",
        choices=["float32", "float64"],
        default="float64",
        help="MAF–LD accumulator precision.",
    )
    architecture = parser.add_argument_group("enrichment and MAF–LD")
    architecture.add_argument(
        "--architecture", choices=["gcta", "ldak"], default="gcta"
    )
    architecture.add_argument(
        "--scenario", choices=SCENARIOS, help="Two-bin enrichment scenario."
    )
    architecture.add_argument(
        "--total-h2",
        type=float,
        help="Total genetic variance for an enrichment scenario.",
    )
    binary = parser.add_argument_group("binary traits")
    binary.add_argument("--prevalence", type=float, default=0.1)
    binary.add_argument(
        "--sampling", choices=["cohort", "case-control"], default="cohort"
    )
    binary.add_argument("--case-frac", type=float, default=0.5)
    binary.add_argument("--n-sample", type=int)
    binary.add_argument(
        "--keep-all",
        action="store_true",
        help="Retain unselected samples with missing phenotypes.",
    )
    binary.add_argument("--coding", choices=["12", "01"], default="12")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.max_mem and args.model not in {"binary", "enrichment"}:
        parser.error("--max-mem applies to binary and enrichment simulations.")
    if args.model == "enrichment":
        if args.scenario is None or args.total_h2 is None or args.params or args.sigma:
            parser.error(
                "Enrichment requires --scenario and --total-h2, without --sigma or --params."
            )
    elif args.scenario is not None or args.total_h2 is not None:
        parser.error("--scenario and --total-h2 apply to enrichment simulations.")
    if args.model in {"continuous", "binary"} and args.architecture != "gcta":
        parser.error("Use --maf-ex/--ld-ex for continuous or binary effect weights.")
    if args.model != "maf-ld" and args.acc_dtype != "float64":
        parser.error("--acc-dtype applies to MAF–LD simulations.")
    if args.model == "binary" and (
        not 0 < args.prevalence < 1 or not 0 < args.case_frac < 1
    ):
        parser.error("Prevalence and case fraction must lie in (0, 1).")

    with open_bed(str(args.bed)) as bed:
        n, m = bed.iid_count, bed.sid_count
    if n < 2 or m == 0:
        parser.error("Simulation requires at least two samples and one variant.")
    samples = read_samples(args.bed)
    a, names = read_annotations(args.annot, m)
    features, _ = read_maf_ld(args.mafld, m)
    params, h2, replicates = read_parameters(args, a.shape[1])
    if args.model == "binary" and (params["min_maf"] != 0 or params["max_maf"] != 0.5):
        parser.error(
            "The binary benchmark uses all variants; restrict the input data before simulation."
        )
    if args.model == "enrichment":
        h2 = enrichment_targets(a, features, params, args.scenario, args.architecture)
    args.chunk_size = (
        args.chunk_size
        if args.chunk_size is not None
        else (8192 if args.model == "maf-ld" else 10000)
    )
    defaults = {
        "continuous": min(128, replicates),
        "binary": 64,
        "enrichment": 32,
        "maf-ld": 25
        if args.architecture == "gcta"
        else (100 if params["p_causal"] == 1 else 10),
    }
    args.rep_batch = (
        args.rep_batch if args.rep_batch is not None else defaults[args.model]
    )
    if args.chunk_size <= 0 or args.rep_batch <= 0:
        parser.error("Chunk size and replicate batch size must be positive.")
    out = new_directory(args.out_dir)
    variance_e = max(0.0, 1 - params["total_h2"])
    if args.model == "maf-ld":
        variance = maf_ld_variances(
            a, features, params, h2, replicates, args.seed, args.architecture
        )
        genetic, rng = maf_ld_genetic_values(
            args.bed,
            variance,
            args.seed,
            args.architecture,
            args.chunk_size,
            args.rep_batch,
            args.acc_dtype,
        )
        if variance_e > 0:
            genetic += rng.normal(0, np.sqrt(variance_e), size=genetic.shape).astype(
                args.acc_dtype, copy=False
            )
        oracle = np.column_stack(
            [
                variance[:, a[:, b] == 1].sum(axis=1, dtype=np.float64)
                for b in range(a.shape[1])
            ]
        )
        counts = np.count_nonzero(variance > 0, axis=1)
    else:
        rng = np.random.default_rng(args.seed)
        beta, oracle, counts = draw_trait_effects(
            a, features, params, h2, replicates, rng, args.model
        )
        genetic = trait_genetic_values(
            args.bed,
            beta,
            rng,
            args.model,
            args.chunk_size,
            args.rep_batch,
            args.max_mem,
        )

    for r in range(replicates):
        y = genetic[:, r].astype(np.float64, copy=True)
        if args.model != "maf-ld" and variance_e > 0:
            y += rng.normal(0, np.sqrt(variance_e), size=n)
        keep = None
        if args.model == "binary":
            y, keep = binary_phenotype(y, args, rng)
            if keep is not None:
                samples.loc[keep].to_csv(
                    out / f"sim_{r}.keep", sep=" ", index=False, header=False
                )
        write_phenotype(
            out / f"sim_{r}.phen", samples, y, None if args.keep_all else keep
        )
    table = pd.DataFrame(oracle, columns=[f"h2bin_{b}" for b in range(a.shape[1])])
    table.insert(0, "rep", np.arange(replicates))
    table["causal_count"] = counts
    table["total_h2"] = oracle.sum(axis=1)
    table.to_csv(out / "sim.oracle_h2.tsv", sep="\t", index=False)
    write_settings(
        out / "settings.json",
        {
            **vars(args),
            "parameters": params,
            "annotation_names": names,
            "h2_bin": h2,
            "num_reps": replicates,
        },
    )
    print(f"Wrote {replicates} phenotype files to {out}")


if __name__ == "__main__":
    main()
