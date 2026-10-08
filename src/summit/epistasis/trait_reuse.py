"""Cohort-side new-phenotype preparation from sealed individual feature designs."""
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from summit.context.spec import canonical_sha256, array_sha256
from summit.prediction._validation import closed, digest
from summit.prediction.cli import _table, _aligned_table
from .features import load_feature_reference
from .robust import prepare_robust_scores, write_robust_scores


def prepare_traits(path, output, *, threads=1, memory_bytes=4 * 2**30):
    """No genotype access. Same sample mask and frozen nuisance/feature space.

    This is preparation, not summary-only inference: the reference contains
    individual genotype-derived features and must remain on the cohort side.
    """
    path = Path(path)
    root = path.parent
    spec = json.loads(path.read_text())
    closed(
        spec,
        ("kind", "schema_version", "reference", "samples", "phenotypes"),
        name="cohort trait reuse",
    )
    if spec["kind"] != "summit.epistasis.prepare_traits" or spec["schema_version"] != 1:
        raise ValueError("unsupported phenotype reuse manifest")
    p = spec["phenotypes"]
    closed(p, ("file", "columns", "unit"), name="reused phenotypes")
    if not isinstance(p["unit"], str) or not p["unit"] or not p["columns"]:
        raise ValueError("phenotype columns and scientific unit required")
    reference_path = root / spec["reference"]
    # Admit decompressed designs, QR/projection copies, table and trait meat
    # before loading sample arrays. The NPZ remains a cohort artifact.
    with ZipFile(reference_path) as z:
        stored = sum(v.file_size for v in z.infolist())
    if 8 * stored + 128 * 2**20 > memory_bytes:
        raise MemoryError("saved cohort reference exceeds preparation memory")
    with np.load(reference_path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["manifest"]))["metadata"]
    strata = meta.get("definitions", {}).get("fixed_effect_strata")
    if (
        strata is not None
        and strata.get("preparation")
        != "complete_feature_nuisance_design_by_stratum_v1"
    ):
        raise ValueError("older stratified feature references require new preparation")
    n = meta["n_samples"]
    q = len(meta["feature_names"])
    t = len(p["columns"])
    if (
        8 * stored + 8 * t * (4 * n + 2 * q * q) + 256 * n + 128 * 2**20
        > memory_bytes
    ):
        raise MemoryError("trait batch and covariance exceed preparation memory")
    reference = load_feature_reference(
        reference_path, compatibility_id=meta["compatibility_id"]
    )
    settings = meta["job"].get("inference", {})
    if settings.get("method") != "robust_mean":
        raise ValueError(
            "phenotype reuse currently requires a robust_mean cohort reference"
        )
    samples = list(_table(root / spec["samples"]).index)
    tokens = meta.get("cohort_sample_tokens")
    if tokens is not None:
        supplied = {canonical_sha256(list(s)): s for s in samples}
        if len(supplied) != len(tokens) or set(supplied) != set(tokens):
            raise ValueError(
                "phenotype reuse requires the identical cohort sample mask"
            )
        samples = [supplied[token] for token in tokens]
    if len(samples) != n or digest(samples) != meta["sample_hash"]:
        raise ValueError(
            "reference sample alignment failed; old references require original source order"
        )
    y = _aligned_table(root / p["file"], samples)[p["columns"]].to_numpy(float)
    from summit.prediction.genotype import native_module
    from summit.prediction.runtime import configure_prediction_threads
    from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
    from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator

    native = native_module()
    configure_prediction_threads(native, threads)
    nn = ProtectedNNOperator(threads=threads, native_module=native)
    tn = ProtectedTNOperator(threads=threads, native_module=native)
    nn.begin_execution()
    tn.begin_execution()
    metadata = {k: v for k, v in meta.items() if k != "cohort_sample_tokens"}
    metadata.update(
        trait_unit=p["unit"],
        phenotype_units="raw",
        phenotypes_hash=array_sha256(y),
        coefficient_contrasts=meta["job"].get("coefficient_contrasts"),
        component_index=reference.component_index.tolist(),
        burden_weights=meta["job"].get("burden_weights"),
        preparation_identity=canonical_sha256(
            dict(
                reference=meta["compatibility_id"],
                phenotypes=array_sha256(y),
                unit=p["unit"],
                traits=p["columns"],
                settings=settings,
            )
        ),
        reuse="identical cohort features and fixed effects; no genotype access",
        genotype_passes_this_preparation=0,
    )
    summary = prepare_robust_scores(
        reference.features,
        y,
        reference.fixed_effects,
        feature_names=meta["feature_names"],
        trait_names=p["columns"],
        metadata=metadata,
        nn=lambda a, b: nn.matmul(np.asfortranarray(a), np.asfortranarray(b)),
        tn=lambda a, b: tn.matmul_tn(np.asfortranarray(a), np.asfortranarray(b)),
        sampling_model=settings.get("sampling_model", "fixed_design_correct_mean"),
        wild_draws=settings.get("wild_draws", 0),
        seed=meta["seed"],
    )
    nn.finish_execution()
    return write_robust_scores(summary, output)
