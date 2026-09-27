"""Bounded, same-input mixture qualification; no existing training jobs/artifacts.

Run in a fresh process with an explicit thread/placement environment. The baseline
is the pre-optimization source at --baseline-commit, with only its native version
check accepting the backwards-compatible residual API. All artifacts use a new
output directory. This synthetic benchmark is not a full-array runtime forecast.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['fit','residual'], default='fit')
    p.add_argument('--baseline-commit', default='6111a12')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--m', type=int, default=384)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--memory-gib', type=float, default=16.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Process-local only, before any large allocation on tabla.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux') and libc.prctl(41,1,0,0,0):
        raise OSError(ctypes.get_errno(), 'PR_SET_THP_DISABLE failed')
    root = Path(__file__).resolve().parents[2]
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_gwldcore_editable']
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import summit
    spec = importlib.util.spec_from_file_location('summit.gxeldcore', args.native.resolve())
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sys.modules['summit.gxeldcore'] = native
    summit.gxeldcore = native
    from summit.prediction.runtime import configure_prediction_threads
    configure_prediction_threads(native,args.threads)
    from summit.prediction import mixture as optimized
    from summit.prediction.artifacts import file_digest
    record = dict(native=str(args.native.resolve()), native_sha256=file_digest(args.native),
                  native_build=native.build_info(), threads=args.threads, n=args.n, m=args.m,
                  seed=20260924, memory_gib=args.memory_gib,
                  source_hashes={name:file_digest(root/name) for name in (
                      'src/summit/prediction/mixture.py', 'src/native/prediction.inc',
                      'src/native/prediction_residual.inc', 'scripts/prediction/benchmark_mixture_optimization.py')},
                  scope='Synthetic bounded benchmark, not a full-array throughput qualification.')
    rng = np.random.default_rng(20260924)
    if args.mode == 'residual':
        n,d,k,r = args.n,640,57,46
        basis = np.asfortranarray(np.linalg.qr(rng.normal(size=(n,r)))[0])
        yw = rng.normal(size=n)
        w = np.asfortranarray(rng.normal(size=(n,d)))
        projection = np.asfortranarray(basis.T@w)
        delta = np.asfortranarray(rng.normal(scale=.001,size=(d,k)))
        output = np.empty((d,k),order='F')
        snapshots, times = {}, {False:[],True:[]}
        for rep in range(args.repeats+1):
            for deferred in ([False,True] if rep%2==0 else [True,False]):
                workspace = native.PredictionMixtureResidual(yw,basis,k,args.threads,deferred)
                start = time.monotonic()
                for _ in range(8):
                    workspace.score(w,projection,output)
                    workspace.update(w,delta,projection)
                workspace.synchronize()
                residual = np.empty((n,k),order='F')
                workspace.copy_residual(residual)
                elapsed = time.monotonic()-start
                if rep:
                    times[deferred].append(elapsed)
                snapshots[deferred] = residual
        error = np.linalg.norm(snapshots[False]-snapshots[True])/np.linalg.norm(snapshots[False])
        if error > 1e-11:
            raise AssertionError(('deferred residual mismatch',error))
        record.update(mode='residual', cases=times, relative_error=float(error),
                      speedup=float(np.median(times[False])/np.median(times[True])))
    else:
        from summit.prediction.genotype import ArrayGenotypeSource, estimate_scale, standardize
        from summit.prediction.spec import VariantAxis, CandidatePrior, TraitTraining
        baseline = types.ModuleType('summit.prediction._benchmark_baseline')
        baseline.__file__ = str(root/'src/summit/prediction/mixture.py')
        sys.modules[baseline.__name__] = baseline
        code = subprocess.check_output(['git','show',f'{args.baseline_commit}:src/summit/prediction/mixture.py'],cwd=root,text=True)
        needle = "if getattr(op.native, 'prediction_residual_version', 0) != 1:"
        if code.count(needle)!=1:
            raise ValueError('baseline compatibility patch is not applicable')
        exec(compile(code.replace(needle,"if getattr(op.native, 'prediction_residual_version', 0) not in (1,2):"),
                     baseline.__file__,'exec'),baseline.__dict__)
        n,m,q = args.n,args.m,5
        calls = rng.binomial(2,.35,size=(n,m)).astype(float)
        # Correlated blocks make easy and slow independent candidates coexist.
        for j in range(m//2,m):
            copy = rng.random(n)<.85
            calls[copy,j] = calls[copy,j-m//2]
        calls[rng.random((n,m))<.01] = np.nan
        axis = VariantAxis(tuple(f'rs{i}' for i in range(m)),('1',)*m,tuple(range(1,m+1)),('A',)*m,('G',)*m,'GRCh37')
        source = ArrayGenotypeSource(calls,[('f',str(i)) for i in range(n)],axis,hard_calls=True)
        rows,variants = np.arange(n),np.arange(m)
        scale = estimate_scale(source,rows,variants,block_size=128,threads=args.threads)
        g = standardize(source.values,scale.mean,scale.inverse_scale)
        phi = np.column_stack([np.ones(n),rng.normal(size=(n,q-1))])
        fixed = np.column_stack([phi,rng.normal(size=(n,7))])
        y = rng.normal(size=n)+.12*g[:,2]+.1*g[:,5]*phi[:,1]
        omega = np.diag([.3,.06,.04,.03,.02])
        omega[0,1]=omega[1,0]=.025
        timings, results = {'baseline':[],'optimized':[]}, {}
        for rep in range(args.repeats):
            for name in (['baseline','optimized'] if rep%2==0 else ['optimized','baseline']):
                module = baseline if name=='baseline' else optimized
                candidates,mixtures = [],{}
                for family in range(19):
                    for strength in (.5,1.,2.):
                        model_id = f'family{family}_s{strength}'
                        covariance = omega.copy()*strength*(1+.03*family)
                        if family<3:
                            covariance[1:]=0.;covariance[:,1:]=0.
                        elif family<6:
                            covariance *= .02
                        else:
                            covariance *= .25+(family-6)/4
                        candidates.append(CandidatePrior(model_id,covariance,np.ones(n),{'source':'synthetic benchmark'}))
                        mixture = (module.MixtureSpec(.5,.5) if family<6 else
                            module.MixtureSpec(.03,.1) if family<12 else
                            module.SeparateSparsitySpec(module.MixtureSpec(.03,.1),module.MixtureSpec(.15,.1)))
                        mixtures['trait',model_id] = mixture
                trait = TraitTraining('trait',rows,variants,y,phi,fixed,scale,tuple(candidates),
                                      {'names':['intercept','e1','e2','e3','e4']},
                                      {'names':[f'fixed{i}' for i in range(fixed.shape[1])]}, {'units':'synthetic'})
                out = args.output/f'{name}_{rep}'
                start = time.monotonic()
                try:
                    models = module.fit_mixture_prediction([trait],source,output=out,storage='compact',
                        block_size=128,threads=args.threads,mixtures=mixtures,
                        solver=module.MixtureSolverSpec(rtol=1e-7,max_sweeps=250),memory_bytes=int(args.memory_gib*2**30))
                except Exception as error:
                    failure = dict(record,passed=False,case=name,repeat=rep,error=repr(error))
                    (args.output/'FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n')
                    raise
                elapsed = time.monotonic()-start
                timings[name].append(elapsed)
                results[name] = models
                print(json.dumps(dict(case=name,repeat=rep,seconds=elapsed,
                                      max_sweeps=max(x.convergence['iterations'] for x in models))),flush=True)
        errors, prediction_errors = [],[]
        for a,b in zip(results['baseline'],results['optimized']):
            assert a.key==b.key and b.convergence['true_residual_norm']<=b.convergence['threshold']
            errors.append(float(np.max(abs(a.weights-b.weights))))
            pa=(g@a.weights*phi).sum(1)+fixed@a.fixed_coefficients
            pb=(g@b.weights*phi).sum(1)+fixed@b.fixed_coefficients
            prediction_errors.append(float(np.max(abs(pa-pb))))
            np.testing.assert_allclose(pb,pa,rtol=2e-6,atol=2e-6)
        record.update(mode='fit',baseline_commit=args.baseline_commit,seconds=timings,
                      speedup=float(np.median(timings['baseline'])/np.median(timings['optimized'])),
                      max_weight_difference=max(errors),max_prediction_difference=max(prediction_errors))
    (args.output/'BENCHMARK.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='native_build'}),flush=True)


if __name__=='__main__':
    main()
