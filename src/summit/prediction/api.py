"""Public workflow: explicit admission, joint fit, streamed export and reload."""
from dataclasses import asdict
from contextlib import ExitStack
import time

from .batch import plan_prediction
from .operator import GenotypeOperator
from .solver import solve
from .spec import SolverSpec
from .artifacts import ModelWriter, load_prediction_models
from ._validation import array_digest, digest


def fit_prediction(traits, source, *, output, plan=None, solver=SolverSpec(), backend="native",
                   checkpoint=None, resume=False):
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
    if resume and checkpoint is None:
        raise ValueError("resume requires a solver checkpoint path")
    if Path(output).exists():
        raise FileExistsError(output)
    import shutil
    expected_disk = sum(8*len(t.variants)*t.phi.shape[1]*len(t.candidates) for t in traits)
    if shutil.disk_usage(Path(output).parent).free < expected_disk + 16*sum(len(t.variants) for t in traits) + 256*2**20:
        raise OSError("insufficient space for posterior model artifacts")
    started = time.monotonic()
    from .artifacts import file_digest
    from .genotype import native_module
    python_identity = digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    native = native_module() if backend == "native" else None
    checkpoint_identity = digest(dict(fit=plan.fit_identity, solver=asdict(solver), backend=backend,
        python=python_identity, native=None if native is None else file_digest(native.__file__),
        block_size=plan.block_size, rhs_columns=plan.rhs_columns, threads=plan.threads))
    owner = ExitStack()
    operator = None
    writer = None
    try:
        state = None
        if checkpoint is not None:
            from .checkpoint import SolverCheckpoint
            checkpoint_bytes = sum(3*8*len(t.rows)*len(t.candidates) for t in traits)
            if shutil.disk_usage(Path(checkpoint).parent).free < 2*checkpoint_bytes + 64*2**20:
                raise OSError("insufficient space for atomic solver checkpoints")
            state = owner.enter_context(SolverCheckpoint(checkpoint, checkpoint_identity, resume=resume))
        operator = GenotypeOperator(source, traits, plan, backend=backend)
        operator.setup()
        result = solve(operator, solver, checkpoint=state)
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
                      solver_seconds=result.elapsed_seconds, elapsed_seconds=time.monotonic()-started,
                      elapsed_seconds_scope="current_attempt_including_setup_and_export",
                      solver_seconds_scope="cumulative_completed_solver_attempts",
                      resumed_solver=bool(resume), solver_checkpoint_enabled=checkpoint is not None)
        writer.finish(report)
    finally:
        if writer is not None:
            writer.close()
        if operator is not None:
            operator.release()
        owner.close()
    # Score exactly the exported representation, not unsaved in-memory weights.
    return load_prediction_models(output)
