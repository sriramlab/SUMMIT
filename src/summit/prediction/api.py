"""Public workflow: explicit admission, joint fit, streamed export and reload."""
from dataclasses import asdict
import time

from .batch import plan_prediction
from .operator import GenotypeOperator
from .solver import solve
from .spec import SolverSpec
from .artifacts import ModelWriter, load_prediction_models
from ._validation import array_digest, digest


def fit_prediction(traits, source, *, output, plan=None, solver=SolverSpec(), backend="native"):
    traits = tuple(traits)
    if plan is None:
        plan = plan_prediction(traits, source)
    # Revalidate axes and resource arithmetic; a plan is not an authorization to
    # substitute different trait masks/priors after admission.
    current = plan_prediction(traits, source, storage=plan.storage, block_size=plan.block_size,
        rhs_columns=plan.rhs_columns, threads=plan.threads, memory_bytes=plan.memory_bytes)
    if current != plan:
        raise ValueError("resource plan does not match the current fit")
    from pathlib import Path
    if Path(output).exists():
        raise FileExistsError(output)
    import shutil
    expected_disk = sum(8*len(t.variants)*t.phi.shape[1]*len(t.candidates) for t in traits)
    if shutil.disk_usage(Path(output).parent).free < expected_disk + 16*sum(len(t.variants) for t in traits) + 256*2**20:
        raise OSError("insufficient space for posterior model artifacts")
    started = time.monotonic()
    operator = GenotypeOperator(source, traits, plan, backend=backend)
    writer = None
    try:
        operator.setup()
        result = solve(operator, solver)
        from .artifacts import file_digest
        python_identity = digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        provenance = dict(source=source.identity, backend=backend, arithmetic="affine64_fp64",
            prediction_python_sha256=python_identity,
            native_build=operator.native.build_info() if operator.native is not None else None,
            solver=asdict(solver), training={t.id: dict(sample_identity=t.scale.sample_identity,
                variant_identity=t.scale.variant_identity, scale_identity=t.scale.identity,
                phenotype=array_digest(t.y), fixed=array_digest(t.fixed), phi=array_digest(t.phi),
                residuals={c.id: array_digest(c.residual) for c in t.candidates}) for t in traits})
        writer = ModelWriter(output, traits, source, result, provenance)
        operator.extract(result.solutions, writer.write)
        report = dict(plan=plan.to_dict(), ledger=asdict(operator.ledger),
                      solver_seconds=result.elapsed_seconds, elapsed_seconds=time.monotonic()-started)
        writer.finish(report)
    finally:
        if writer is not None:
            writer.close()
        operator.release()
    # Score exactly the exported representation, not unsaved in-memory weights.
    return load_prediction_models(output)
