#!/usr/bin/env python3
"""Simulate paired traits with partitioned or overlapping genetic covariance."""
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


def parse_vector(value, bins, name):
    values = np.array(str(value).split(","), dtype=float)
    if len(values) == 1:
        values = np.repeat(values, bins)
    if len(values) != bins or not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} needs one finite value or {bins} comma-separated values."
        )
    return values


def covariance_matrix(v1, v2, covariance):
    limit = np.sqrt(max(v1, 0.0) * max(v2, 0.0))
    covariance = (
        0.0 if limit <= 1e-18 else np.clip(covariance, -(limit - 1e-18), limit - 1e-18)
    )
    return np.array([[v1, covariance], [covariance, v2]], dtype=np.float64)


def covariance_factor(covariance):
    try:
        return np.linalg.cholesky(covariance + np.eye(2) * 1e-30)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(covariance)
        return (vectors * np.sqrt(np.clip(values, 0, None))[None, :]) @ vectors.T


def effect_coefficients(a, setting):
    """Return the two-trait effect factor for each annotation."""
    counts = np.asarray(a.sum(axis=0), dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("Each selected annotation must contain at least one variant.")
    q = setting.get("weights")
    mass = counts.astype(float) if q is None else a.T @ q
    if np.any(mass <= 0):
        raise ValueError("Each annotation must have positive total effect weight.")
    factors = np.zeros((a.shape[1], 2, 2))
    for b in range(a.shape[1]):
        v1, v2 = float(setting["sig1"][b]), float(setting["sig2"][b])
        covariance = float(setting["rho_g"][b]) * np.sqrt(v1 * v2)
        if q is None:
            factor = covariance_matrix(
                v1 / float(counts[b]),
                v2 / float(counts[b]),
                covariance / float(counts[b]),
            )
        else:
            factor = covariance_matrix(v1, v2, covariance)
        factors[b] = covariance_factor(factor)
    return factors, mass


def simulate_group(
    bed_path, samples, a, settings, chunk_size, rep_batch, dtype, max_mem
):
    """Read each genotype block once for settings sharing the same annotation."""
    with open_bed(str(bed_path)) as bed:
        n = bed.iid_count
    for setting in settings:
        effect_seed, environment_seed = np.random.SeedSequence(setting["seed"]).spawn(2)
        setting["effect_rng"] = np.random.default_rng(effect_seed)
        setting["environment_rng"] = np.random.default_rng(environment_seed)
        setting["factors"], setting["mass"] = effect_coefficients(a, setting)
        setting["genetic"] = np.zeros((n, setting["num_reps"], 2), dtype=dtype)

    for start, end, block in genotype_blocks(bed_path, chunk_size, "float32", max_mem):
        x = standardize(block.astype(dtype, copy=False))
        indices = [np.flatnonzero(a[start:end, b]) for b in range(a.shape[1])]
        for setting in settings:
            rng = setting["effect_rng"]
            for r0 in range(0, setting["num_reps"], rep_batch):
                r1 = min(r0 + rep_batch, setting["num_reps"])
                k = r1 - r0
                effects = np.zeros((end - start, 2 * k), dtype=dtype)
                for b, loc in enumerate(indices):
                    if not len(loc):
                        continue
                    factor = setting["factors"][b]
                    z0 = rng.standard_normal((len(loc), k), dtype=dtype)
                    z1 = rng.standard_normal((len(loc), k), dtype=dtype)
                    if setting.get("weights") is not None:
                        scale = np.sqrt(
                            setting["weights"][start:end][loc] / setting["mass"][b]
                        ).astype(dtype, copy=False)
                        z0 *= scale[:, None]
                        z1 *= scale[:, None]
                    beta1 = z0 * factor[0, 0] + z1 * factor[0, 1]
                    beta2 = z0 * factor[1, 0] + z1 * factor[1, 1]
                    if setting["annotation_model"] == "overlap_additive":
                        effects[loc, :k] += beta1
                        effects[loc, k:] += beta2
                    else:
                        effects[loc, :k] = beta1
                        effects[loc, k:] = beta2
                values = x @ effects
                setting["genetic"][:, r0:r1, 0] += values[:, :k]
                setting["genetic"][:, r0:r1, 1] += values[:, k:]

    for setting in settings:
        rng = setting["environment_rng"]
        ve1 = max(0.0, 1 - float(setting["sig1"].sum()))
        ve2 = max(0.0, 1 - float(setting["sig2"].sum()))
        gamma = setting["gamma_e"]
        if gamma != 0:
            factor = covariance_factor(covariance_matrix(ve1, ve2, gamma))
        for r in range(setting["num_reps"]):
            if gamma == 0:
                e1 = rng.normal(0, np.sqrt(ve1), size=n)
                e2 = rng.normal(0, np.sqrt(ve2), size=n)
            else:
                z0, z1 = rng.standard_normal(n), rng.standard_normal(n)
                e1 = z0 * factor[0, 0] + z1 * factor[0, 1]
                e2 = z0 * factor[1, 0] + z1 * factor[1, 1]
            write_phenotype(
                setting["output"] / f"sim_{r}_1.phen",
                samples,
                setting["genetic"][:, r, 0] + e1,
            )
            write_phenotype(
                setting["output"] / f"sim_{r}_2.phen",
                samples,
                setting["genetic"][:, r, 1] + e2,
            )
        print(f"Wrote {setting['num_reps']} phenotype pairs to {setting['output']}")
        del setting["genetic"]


def relative_path(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(
            f"Use a relative path within the input/output directory: {value}"
        )
    return path


def optional(row, key, default):
    value = row.get(key)
    return default if value is None or pd.isna(value) else value


def load_settings(args):
    table = pd.read_csv(args.settings)
    required = {
        "pop",
        "annot",
        "out_prefix",
        "sig1",
        "sig2",
        "rho_g",
        "gamma_e",
        "num_reps",
        "seed",
        "max_mem",
    }
    if table.empty or not required <= set(table.columns):
        raise ValueError(
            f"Settings CSV is empty or missing columns: {sorted(required - set(table.columns))}"
        )
    groups, cache, destinations = {}, {}, set()
    for _, row in table.iterrows():
        pop = str(relative_path(row["pop"]))
        bed_path = args.input_dir / "genotypes" / pop / "genotypes.bed"
        annotation_path = (
            args.input_dir
            / "annotations"
            / pop
            / relative_path(str(row["annot"]).format(pop=pop))
        )
        destination = (
            args.out_dir / pop / relative_path(str(row["out_prefix"]).format(pop=pop))
        )
        if destination in destinations:
            raise ValueError(f"Duplicate simulation output: {destination}")
        destinations.add(destination)
        architecture = str(optional(row, "architecture", "gcta")).lower()
        model = str(optional(row, "annotation_model", "partition"))
        if architecture not in {"gcta", "ldak"} or model not in {
            "partition",
            "overlap_additive",
        }:
            raise ValueError(
                "Use gcta/ldak architecture and partition/overlap_additive annotations."
            )
        if model == "overlap_additive" and architecture != "gcta":
            raise ValueError("The overlapping-annotation benchmark uses GCTA effects.")
        columns = optional(row, "annot_cols", None)
        add_base = str(
            optional(row, "add_base", 1 if model == "overlap_additive" else 0)
        ).lower() in {"true", "1", "1.0", "yes"}
        if model == "overlap_additive" and columns is None:
            raise ValueError(
                "Overlapping simulations require an explicit annot_cols list."
            )
        if model == "partition" and add_base:
            raise ValueError("Do not add a base column to a disjoint partition.")
        key = (bed_path, annotation_path, columns, add_base, model)
        if key not in cache:
            with open_bed(str(bed_path)) as bed:
                variants = bed.sid_count
            a, names = read_annotations(
                annotation_path,
                variants,
                columns=columns,
                add_base=add_base,
                partition=model == "partition",
            )
            cache[key] = (read_samples(bed_path), a, names)
        samples, a, names = cache[key]
        setting = {
            name: parse_vector(row[name], a.shape[1], name)
            for name in ("sig1", "sig2", "rho_g")
        }
        if (
            np.any(setting["sig1"] < 0)
            or np.any(setting["sig2"] < 0)
            or max(setting["sig1"].sum(), setting["sig2"].sum()) > 1
        ):
            raise ValueError(
                "Trait variances must be nonnegative and sum to at most one."
            )
        if np.any(abs(setting["rho_g"]) > 1):
            raise ValueError("Genetic correlations must lie in [-1, 1].")
        gamma = float(row["gamma_e"])
        bound = np.sqrt((1 - setting["sig1"].sum()) * (1 - setting["sig2"].sum()))
        if not np.isfinite(gamma) or abs(gamma) > bound:
            raise ValueError("Environmental covariance exceeds the variance bound.")
        integers = [float(row[key]) for key in ("num_reps", "seed", "max_mem")]
        if any(not value.is_integer() for value in integers):
            raise ValueError("num_reps, seed, and max_mem must be integers.")
        num_reps, seed, max_mem = map(int, integers)
        if num_reps <= 0 or seed < 0 or max_mem not in {0, 1}:
            raise ValueError(
                "Use positive num_reps, nonnegative seed, and max_mem 0 or 1."
            )
        batch = (
            args.rep_batch
            if args.rep_batch is not None
            else (100 if model == "overlap_additive" else 10)
        )
        setting.update(
            gamma_e=gamma,
            num_reps=num_reps,
            seed=seed,
            output=destination,
            architecture=architecture,
            annotation_model=model,
        )
        if architecture == "ldak":
            weight_path = (
                args.input_dir
                / "annotations"
                / pop
                / relative_path(optional(row, "mafld", "maf_ld_features.tsv"))
            )
            weight_key = (weight_path, len(a))
            if weight_key not in cache:
                features, explicit_weights = read_maf_ld(weight_path, len(a))
                cache[weight_key] = ldak_weights(features, explicit_weights)
            setting["weights"] = cache[weight_key]
        group_key = (*key, num_reps, max_mem, batch)
        groups.setdefault(
            group_key,
            dict(
                bed_path=bed_path,
                samples=samples,
                a=a,
                settings=[],
                chunk_size=args.chunk_size,
                rep_batch=batch,
                dtype=np.dtype(args.acc_dtype).type,
                max_mem=bool(max_mem),
            ),
        )
        groups[group_key]["settings"].append(setting)
        setting["record"] = {
            **row.dropna().to_dict(),
            "annotation_names": names,
            "architecture": architecture,
            "annotation_model": model,
            "chunk_size": args.chunk_size,
            "rep_batch": batch,
            "acc_dtype": args.acc_dtype,
        }
        effect_coefficients(a, setting)
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Manuscript simulation settings CSV.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--rep-batch",
        type=int,
        help="Default 10 for partitions, 100 for overlapping annotations.",
    )
    parser.add_argument(
        "--acc-dtype", choices=["float32", "float64"], default="float64"
    )
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or (args.rep_batch is not None and args.rep_batch <= 0):
        parser.error("Chunk size and replicate batch size must be positive.")
    groups = load_settings(args)
    new_directory(args.out_dir)
    for group in groups.values():
        for setting in group["settings"]:
            setting["output"].mkdir(parents=True)
            write_settings(setting["output"] / "settings.json", setting.pop("record"))
        simulate_group(**group)


if __name__ == "__main__":
    main()
