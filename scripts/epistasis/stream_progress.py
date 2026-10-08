"""Durable genotype-pass progress around an unchanged research/public command.

Use inside the qualified checkout/private launcher. Only aggregate block counts
are recorded. This wrapper does not change genotype blocks, cache construction,
arithmetic, checkpoints or runtime placement. Resource limits remain the outer
supervisor's responsibility.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import runpy
import sys
import time
from unittest.mock import patch


@contextmanager
def record_progress(path, *, interval=60.):
    from summit.prediction.genotype import RawBlockStream
    if interval<=0:raise ValueError('a positive progress interval is required')
    original=RawBlockStream.blocks
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    pass_number=0
    active={}
    with Path(path).open('x') as handle:
        os.chmod(path,0o600)
        directory=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        def observed(stream,phase,**kwargs):
            nonlocal pass_number
            if handle.closed:
                yield from original(stream,phase,**kwargs)
                return
            pass_number+=1;current=pass_number
            start=time.monotonic();last=start;completed=0;status='closed_before_completion'
            def emit(event):
                if handle.closed:return
                value=dict(unix_time=time.time(),pid=os.getpid(),boot_id=boot,
                    pass_number=current,phase=phase,event=event,rows=len(stream.rows),
                    variants=len(stream.variants),completed_variants=completed,
                    storage=stream.storage,elapsed_seconds=time.monotonic()-start)
                handle.write(json.dumps(value,separators=(',',':'))+'\n')
                handle.flush();os.fsync(handle.fileno())
            active[current]=emit
            emit('started')
            try:
                for block in original(stream,phase,**kwargs):
                    yield block
                    # The consumer has returned after processing this block.
                    completed=int(block[0])+len(block[1])
                    now=time.monotonic()
                    if now-last>=interval:emit('progress');last=now
                status='finished'
            except Exception:
                status='failed';raise
            finally:
                if active.pop(current,None) is not None:emit(status)
        with patch.object(RawBlockStream,'blocks',observed):
            try:yield
            finally:
                # A caller may retain a partially consumed iterator. Record its
                # state before closing the log; later closure must not write to
                # the closed handle or mask the caller's original exception.
                for emit in active.values():emit('closed_before_completion')
                active.clear()


@contextmanager
def verify_recovered_solves(path):
    """Recompute full-operator residuals after loading a frozen solver result.

    Completed checkpoints previously reused their saved residual reports. This
    recovery check uses the unchanged operator and original tolerances before
    any dependent calculation; it never repairs or alters a checkpoint.
    """
    import numpy as np
    from summit.epistasis import polygenic
    from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
    from summit.prediction.spec import SolverSpec
    original = polygenic.projected_solve
    with Path(path).open('x') as handle:
        os.chmod(path, 0o600)
        def checked(operator, y, fixed, theta, **kwargs):
            solution, result = original(operator, y, fixed, theta, **kwargs)
            if kwargs.get('resume', False):
                spec = kwargs.get('spec', SolverSpec(rtol=1e-8))
                basis = thin_rank_revealing_fixed_effect_basis(fixed, rtol=spec.qr_rtol)
                rhs = np.asarray(y, float)
                if rhs.ndim == 1:
                    rhs = rhs[:, None]
                projected_rhs = rhs-basis@(basis.T@rhs)
                coefficients = np.asarray(theta, float)
                if coefficients.ndim == 1:
                    coefficients = np.repeat(coefficients[:,None], rhs.shape[1], axis=1)
                value = operator.apply(solution, coefficients, phase='recovery_full_residual')
                residual = projected_rhs-(value-basis@(basis.T@value))
                norms = np.linalg.norm(residual, axis=0)
                thresholds = np.maximum(spec.atol, spec.rtol*np.linalg.norm(projected_rhs, axis=0))
                leakage = np.linalg.norm(basis.T@solution, axis=0)/np.maximum(
                    np.linalg.norm(solution, axis=0), np.finfo(float).tiny)
                passed = bool(np.all(np.isfinite(norms)) and np.all(norms <= thresholds)
                    and np.all(leakage <= max(1e-12, 10*spec.qr_rtol)))
                record = dict(unix_time=time.time(), checkpoint=str(kwargs.get('checkpoint')),
                    checked_columns=rhs.shape[1], residual_norms=norms.tolist(),
                    thresholds=thresholds.tolist(), fixed_projections=leakage.tolist(), passed=passed)
                handle.write(json.dumps(record, separators=(',', ':'))+'\n')
                handle.flush(); os.fsync(handle.fileno())
                if not passed:
                    raise RuntimeError('recovered solve failed fresh full-operator residual verification')
            return solution, result
        with patch.object(polygenic, 'projected_solve', checked):
            yield


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify-recovery',type=Path,help='New log for full-operator verification of resumed solves')
    parser.add_argument('module')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    sys.argv=[args.module,*args.arguments]
    with record_progress(args.out), (verify_recovered_solves(args.verify_recovery)
            if args.verify_recovery else nullcontext()):runpy.run_module(args.module,run_name='__main__')


if __name__=='__main__':main()
