"""Streamed multivariate mixture variational Bayes with cached block Gram matrices.

Mixture candidates share decoding, projected designs and GEMMs when their
residual surfaces agree. Coordinate updates use the native blocked kernel.
The fixed-effect span is integrated, including in posterior precisions.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import os
import tempfile
import time

import numpy as np
from scipy import linalg

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from ._validation import array_digest, canonical, digest, positive_int
from .artifacts import ModelWriter, file_digest, load_prediction_models
from .batch import plan_prediction
from .checkpoint import SolverCheckpoint
from .operator import GenotypeOperator
from .solver import ConvergenceError, SolveResult


@dataclass(frozen=True)
class MixtureSpec:
    probability: float = .01
    small_variance_fraction: float = .1

    def __post_init__(self):
        if not np.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError('mixture probability must lie in (0,1)')
        if not np.isfinite(self.small_variance_fraction) or not 0 <= self.small_variance_fraction <= 1:
            raise ValueError('small variance fraction must lie in [0,1]')

    @property
    def probabilities(self):
        return np.array([self.probability, 1-self.probability])

    @property
    def scales(self):
        p, f = self.probability, self.small_variance_fraction
        return np.array([(1-f)/p, f/(1-p)])

    def components(self, covariance):
        return self.probabilities, self.scales[:,None,None,None]*covariance[None]


@dataclass(frozen=True)
class SeparateSparsitySpec:
    """Separate mixture indicators for baseline and orthogonal response.

    The first coefficient defines the baseline anchor. Conditional response
    covariance is the Schur complement, recomputed for each annotated SNP.
    A zero baseline variance is supported without division by zero.
    """
    baseline: MixtureSpec = MixtureSpec()
    response: MixtureSpec = MixtureSpec()
    coupling: float = 0.

    def __post_init__(self):
        if not isinstance(self.baseline, MixtureSpec) or not isinstance(self.response, MixtureSpec):
            raise TypeError('baseline and response must be MixtureSpec')
        if not np.isfinite(self.coupling) or not -1 < self.coupling < 1:
            raise ValueError('indicator coupling must lie strictly in (-1,1)')

    def components(self, covariance):
        first = covariance[:,:,0]
        a = covariance[:,0,0]
        amplified = np.zeros_like(covariance)
        positive = a > 0
        amplified[positive] = (first[positive,:,None]*first[positive,None,:])/a[positive,None,None]
        orthogonal = covariance-amplified
        # A pure amplification prior has a zero Schur complement. Subtraction
        # can leave roundoff with a negative eigenvalue; compare that error to
        # the original covariance, not to the nearly zero complement itself.
        orthogonal[:,0,:] = 0
        orthogonal[:,:,0] = 0
        if covariance.shape[-1] > 1:
            values, vectors = np.linalg.eigh(orthogonal[:,1:,1:])
            scale = np.maximum(np.max(abs(covariance),axis=(1,2)),np.finfo(float).tiny)
            if np.any(values < -1e-10*scale[:,None]):
                raise ValueError('separate-sparsity Schur complement is not positive semidefinite')
            orthogonal[:,1:,1:] = (vectors*np.maximum(values,0)[:,None,:])@vectors.transpose(0,2,1)
        probabilities = np.outer(self.baseline.probabilities,self.response.probabilities)
        # Move toward the positive/negative Frechet bound while preserving both
        # marginal mixture probabilities and therefore the mean covariance.
        bound = (min(probabilities[0,1],probabilities[1,0]) if self.coupling >= 0
                 else min(probabilities[0,0],probabilities[1,1]))
        probabilities += self.coupling*bound*np.array([[1.,-1.],[-1.,1.]])
        probabilities = probabilities.ravel()
        components = np.array([sa*amplified+su*orthogonal
            for sa in self.baseline.scales for su in self.response.scales])
        return probabilities, components


def mixture_from_dict(value):
    value = dict(value)
    kind = value.pop('kind', 'radial')
    if kind == 'radial':
        return MixtureSpec(**value)
    if (kind == 'separate_sparsity' and {'baseline','response'} <= set(value)
            and not set(value)-{'baseline','response','coupling'}):
        return SeparateSparsitySpec(MixtureSpec(**value['baseline']), MixtureSpec(**value['response']),
                                   value.get('coupling',0.))
    raise ValueError('invalid mixture kind or separate-sparsity fields')


def shrink_orthogonal_covariance(covariance, factors, *, environment_metric=None):
    """Shrink descending baseline-orthogonal eigen-directions of a prior.

    The baseline variance and amplification coefficients remain fixed. With
    metric S=Cov(E), directions diagonalize S**(1/2) U S**(1/2), making their
    order reflect prediction variance in the training environment distribution.
    Factors are nonnegative and at most one. Unequal factors in a tied
    eigenspace are rejected because its individual directions are not defined.
    Supports homogeneous and per-annotation/per-SNP covariance arrays.
    """
    covariance = np.asarray(covariance,dtype=float)
    q = covariance.shape[-1] if covariance.ndim >= 2 else 0
    if q < 2 or covariance.shape[-2] != q or not np.isfinite(covariance).all():
        raise ValueError('orthogonal shrinkage requires finite square covariances with q >= 2')
    if not np.allclose(covariance,covariance.swapaxes(-1,-2),rtol=1e-12,atol=1e-14):
        raise ValueError('covariance must be symmetric')
    factors = np.asarray(factors,dtype=float)
    if factors.shape != (q-1,) or not np.isfinite(factors).all() or np.any((factors<0)|(factors>1)):
        raise ValueError('supply one shrinkage factor in [0,1] per orthogonal direction')
    metric = np.eye(q-1) if environment_metric is None else np.asarray(environment_metric,dtype=float)
    if metric.shape != (q-1,q-1) or not np.isfinite(metric).all() or not np.allclose(metric,metric.T):
        raise ValueError('environment metric must be finite and symmetric')
    values,vectors = np.linalg.eigh(metric)
    if np.any(values <= max(np.max(abs(values)),np.finfo(float).tiny)*1e-12):
        raise ValueError('environment metric must be positive definite')
    root = (vectors*np.sqrt(values))@vectors.T
    inverse = (vectors/np.sqrt(values))@vectors.T
    flat = covariance.reshape(-1,q,q)
    # Reuse the singular-support and Schur-complement validation in the prior.
    scale = np.maximum(np.max(abs(flat),axis=(1,2)),np.finfo(float).tiny)
    if np.any(np.linalg.eigvalsh(flat)<-1e-10*scale[:,None]):
        raise ValueError('prior covariance must be positive semidefinite')
    a = flat[:,0,0]
    amplified = np.zeros_like(flat)
    positive = a>0
    amplified[positive] = flat[positive,:,0,None]*flat[positive,None,0,:]/a[positive,None,None]
    u = flat[:,1:,1:]-amplified[:,1:,1:]
    values,vectors = np.linalg.eigh(root@u@root)
    values,vectors = values[:,::-1],vectors[:,:,::-1]
    if np.any(values < -1e-10*scale[:,None]*np.linalg.norm(metric,2)):
        raise ValueError('orthogonal covariance must be positive semidefinite')
    tied = abs(np.diff(values,axis=1)) <= 1e-10*np.maximum(np.max(abs(values),axis=1,keepdims=True),np.finfo(float).tiny)
    if np.any(tied & (abs(np.diff(factors))[None,:]>1e-12)):
        raise ValueError('tied eigen-directions require equal shrinkage factors')
    out = amplified.copy()
    out[:,1:,1:] += inverse@((vectors*(np.maximum(values,0)*factors)[:,None,:])@vectors.swapaxes(-1,-2))@inverse
    return out.reshape(covariance.shape)


@dataclass(frozen=True)
class MixtureSolverSpec:
    rtol: float = 1e-7
    atol: float = 1e-10
    max_sweeps: int = 100
    block_sweeps: int = 50
    qr_rtol: float = 1e-12
    residual_refresh: int = 5

    def __post_init__(self):
        for key in ('rtol', 'qr_rtol'):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) < 1:
                raise ValueError(key+' must lie in (0,1)')
        if not np.isfinite(self.atol) or self.atol < 0:
            raise ValueError('atol must be finite and nonnegative')
        positive_int(self.max_sweeps, 'max_sweeps')
        positive_int(self.block_sweeps, 'block_sweeps')
        positive_int(self.residual_refresh, 'residual_refresh')


class _Checkpoint(SolverCheckpoint):
    """Reuse the existing exclusive ownership, with mixture-specific state."""
    def save(self, groups, sweep, history, elapsed):
        arrays, records = {}, {}
        for j, group in enumerate(groups):
            for field in ('weights', 'residual', 'penalty'):
                name = f'{field}_{j}'
                value = group[field]
                arrays[name] = value
                records[name] = dict(shape=list(value.shape), sha256=array_digest(value))
        meta = dict(schema=1, kind='mixture', identity=self.identity, arrays=records,
                    sweep=sweep, history=history, elapsed=elapsed,
                    initialization=getattr(self,'initialization',None))
        arrays['metadata'] = np.frombuffer(canonical(meta).encode(), dtype=np.uint8)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name+'.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            from .artifacts import _sync_directory
            _sync_directory(self.path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, groups):
        with np.load(self.path, allow_pickle=False) as data:
            meta = json.loads(data['metadata'].tobytes())
            if meta.get('kind') != 'mixture' or meta['identity'] != self.identity:
                raise ValueError('mixture checkpoint identity mismatch')
            if type(meta['sweep']) is not int or meta['sweep'] < 0:
                raise ValueError('invalid mixture checkpoint iteration')
            self.initialization = meta.get('initialization')
            for j, group in enumerate(groups):
                for field in ('weights', 'residual', 'penalty'):
                    name = f'{field}_{j}'
                    value = data[name]
                    record = meta['arrays'][name]
                    if (value.shape != group[field].shape or value.dtype != np.float64 or
                            not np.isfinite(value).all() or array_digest(value) != record['sha256']):
                        raise ValueError('mixture checkpoint array mismatch')
                    group[field][...] = value
            return meta['sweep'], meta['history'], meta['elapsed']


def _posterior(gram, covariance, spec, component_count=None):
    """Batch posterior component covariances, retaining singular prior support."""
    q = gram.shape[-1]
    covariances, normalizers = [], []
    probabilities, components = spec.components(covariance)
    if component_count is not None and len(probabilities) != component_count:
        if component_count % len(probabilities):
            raise ValueError('incompatible mixture component counts')
        factor = component_count//len(probabilities)
        probabilities = np.repeat(probabilities/factor,factor)
        components = np.repeat(components,factor,axis=0)
    for probability, component in zip(probabilities, components):
        values, vectors = np.linalg.eigh(component)
        scale = np.maximum(np.max(abs(values),axis=1,keepdims=True),np.finfo(float).tiny)
        if np.any(values < -1e-10*scale):
            raise ValueError('mixture component covariance is not positive semidefinite')
        values = np.where(values > scale*1e-12,values,0)
        factor = vectors*np.sqrt(values[:,None,:])
        information = factor.transpose(0,2,1)@gram@factor
        precision = np.eye(q)+information
        sign, logdet = np.linalg.slogdet(precision)
        if np.any(sign <= 0):
            raise FloatingPointError('mixture posterior precision is not positive definite')
        covariances.append(factor@np.linalg.solve(precision, factor.transpose(0, 2, 1)))
        normalizers.append(np.log(probability)-.5*logdet)
    return np.stack(covariances, axis=1), np.stack(normalizers, axis=1)


def _numpy_update(gram, score, covariances, normalizers, weights, penalty, q,
                  sweeps, tolerance, independent):
    """Small independent backend for kernel and end-to-end qualification."""
    from scipy.special import logsumexp
    k, d = weights.shape
    b = d//q
    components = normalizers.shape[1]//b
    covariance = covariances.reshape(k, b, components, q, q)
    normalizer = normalizers.reshape(k, b, components)
    metrics = np.zeros((k, 2))
    for model in range(k):
        gradient = score[model].copy()
        for sweep in range(1 if independent else sweeps):
            energy = 0.
            for j in range(b):
                sl = slice(j*q, (j+1)*q)
                diagonal = gram[sl, sl]
                old = weights[model, sl].copy()
                t = gradient[sl]+diagonal@old
                means = covariance[model, j]@t
                logweights = normalizer[model, j]+.5*(means@t)
                logz = logsumexp(logweights)
                mean = np.exp(logweights-logz)@means
                delta = mean-old
                weights[model, sl] = mean
                penalty[model, j] = t@mean-.5*mean@diagonal@mean-logz
                energy += max(0., delta@diagonal@delta)
                if not independent:
                    gradient -= gram[:, sl]@delta
            metrics[model] = energy, sweep+1
            if energy <= tolerance*tolerance:
                break
    return metrics


def plan_mixture_prediction(traits, source, *, storage='stream', block_size=128,
                            threads=1, memory_bytes=16*2**30):
    """Metadata-only admission including Gram and component-posterior caches."""
    traits = tuple(traits)
    if not traits or storage not in ('stream', 'compact'):
        raise ValueError('mixture requires traits and stream or compact storage')
    if any(not 1 <= t.phi.shape[1] <= 32 for t in traits):
        raise ValueError('mixture supports one to 32 basis coordinates')
    rhs = max(t.phi.shape[1]*len(t.candidates) for t in traits)
    plan = plan_prediction(traits, source, storage=storage, block_size=block_size,
                          rhs_columns=rhs, threads=threads, memory_bytes=memory_bytes)
    extra = 0
    for t in traits:
        n, q = t.phi.shape
        m, k = len(t.variants), len(t.candidates)
        surfaces = len({array_digest(c.residual) for c in t.candidates})
        extra += 8*(surfaces*m*(min(block_size, m)*q*q+t.fixed.shape[1]*q) + k*m*(4*q*q+3*q+7)
                    + 3*n*k + surfaces*n*(t.fixed.shape[1]+1)
                    + 6*n*min(block_size, m)*q
                    + (min(block_size, m)*q)**2
                    + min(threads,k)*min(block_size, m)*q)
    peak = plan.estimated_peak_bytes+extra
    if peak > memory_bytes:
        raise MemoryError('mixture Gram/posterior cache exceeds the memory budget; reduce block_size')
    return replace(plan, estimated_peak_bytes=peak,
                   allocations={**plan.allocations, 'mixture_gram_posterior_and_design': extra})


def fit_mixture_prediction(traits, source, *, output, mixtures, storage='stream',
                           block_size=128, threads=1, memory_bytes=16*2**30,
                           solver=MixtureSolverSpec(), backend='native',
                           checkpoint=None, resume=False, progress=None, initial_weights=None):
    """Fit covariance-preserving two-normal mixtures, exporting standard models.

    ``mixtures`` maps (trait ID, candidate ID) to MixtureSpec. The prior is
    p*N(0, (1-f)/p * Omega_j) + (1-p)*N(0, f/(1-p) * Omega_j).
    Omega_j is the existing homogeneous/annotation SNP covariance divided by M.
    Equal p=f=.5 is the Gaussian limit. Candidate fitting does not select a
    model; use discovery validation and the existing score/calibration API.

    A success certificate is a simultaneous, frozen-weight VB fixed-point
    check using a freshly reconstructed residual, not a Gaussian PCG residual.
    An optional checkpoint atomically saves every completed sweep. The same
    source/input/build/solver identity is required for resume.
    """
    traits = tuple(traits)
    keys = {(t.id, c.id) for t in traits for c in t.candidates}
    if set(mixtures) != keys or any(not isinstance(x, (MixtureSpec,SeparateSparsitySpec)) for x in mixtures.values()):
        raise ValueError('supply exactly one MixtureSpec for each trait/candidate')
    if storage not in ('stream', 'compact'):
        raise ValueError('mixture storage must be stream or compact')
    if not isinstance(solver, MixtureSolverSpec):
        raise TypeError('mixture fitting requires MixtureSolverSpec')
    if resume and checkpoint is None:
        raise ValueError('resume requires a mixture checkpoint')
    if resume and initial_weights is not None:
        raise ValueError('warm start and checkpoint resume are distinct operations')
    if initial_weights is not None:
        if set(initial_weights) != keys:
            raise ValueError('warm start requires all candidate keys')
        for t in traits:
            for c in t.candidates:
                w = np.asarray(initial_weights[t.id,c.id])
                if w.shape != (len(t.variants),t.phi.shape[1]) or not np.isfinite(w).all():
                    raise ValueError('invalid warm-start weights')
    if Path(output).exists():
        raise FileExistsError(output)
    import shutil
    required_disk = sum(8*len(t.variants)*t.phi.shape[1]*len(t.candidates) for t in traits)+256*2**20
    if shutil.disk_usage(Path(output).parent).free < required_disk:
        raise OSError('insufficient disk space for mixture model artifacts')
    if checkpoint is not None:
        state_bytes = sum(8*len(t.candidates)*(len(t.variants)*(t.phi.shape[1]+1)+len(t.rows)) for t in traits)
        if shutil.disk_usage(Path(checkpoint).parent).free < 2*state_bytes+64*2**20:
            raise OSError('insufficient disk space for atomic mixture checkpoints')
    enriched = []
    for t in traits:
        candidates = []
        for c in t.candidates:
            specification = dict(c.specification)
            if 'mixture' in specification:
                raise ValueError('mixture specification must be supplied only once')
            specification['mixture'] = dict(kind=('separate_sparsity' if isinstance(mixtures[t.id,c.id],SeparateSparsitySpec)
                else 'covariance_preserving_two_normal'),
                **asdict(mixtures[t.id, c.id]), fixed_effects='integrated')
            candidates.append(replace(c, specification=specification))
        enriched.append(replace(t, candidates=tuple(candidates)))
    traits = tuple(enriched)
    plan = plan_mixture_prediction(traits, source, storage=storage, block_size=block_size,
                                   threads=threads, memory_bytes=memory_bytes)
    from contextlib import ExitStack
    with ExitStack() as owner:
        op = GenotypeOperator(source, traits, plan, backend=backend)
        owner.callback(op.release)
        if op.native is not None:
            if getattr(op.native, 'prediction_mixture_version', 0) != 2:
                raise ImportError('rebuild SUMMIT with native mixture support')
            if getattr(op.native, 'prediction_residual_version', 0) != 1:
                raise ImportError('rebuild SUMMIT with native mixture residual ownership')
            build = op.native.build_info()
            if build.get('blas_vendor') == 'BLIS' and not (build.get('gemm_integrity_enabled') and build.get('gemm_checksum_enabled')):
                raise ValueError('mixture BLIS fits require both GEMM integrity and checksum guards')
        python_identity = digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        initialization = (None if initial_weights is None else
                          digest({str(k):array_digest(v) for k,v in initial_weights.items()}))
        identity = digest(dict(kind='mixture_v3_native_state', fit=plan.fit_identity, solver=asdict(solver),
            python=python_identity, native=None if op.native is None else file_digest(op.native.__file__),
            block_size=block_size, threads=threads, storage=storage))
        state = None if checkpoint is None else owner.enter_context(_Checkpoint(checkpoint, identity, resume=resume))
        if state is not None:
            state.initialization = initialization
        started = time.monotonic()

        def product_into(left,right,out):
            if op.native is None:
                np.matmul(left,right,out=out)
            else:
                op.native.prediction_product(np.asfortranarray(left),np.asfortranarray(right),
                    out,False,threads,True)

        def product(left,right,*,transpose=False):
            out=np.empty((left.shape[1] if transpose else left.shape[0],right.shape[1]),order='F')
            if op.native is None:
                np.matmul(left.T if transpose else left,right,out=out)
            else:
                op.native.prediction_product(np.asfortranarray(left),np.asfortranarray(right),
                    out,transpose,threads,True)
            return out

        def fixed_projection(basis,value):
            if not basis.shape[1]:
                return np.zeros_like(value)
            return product(basis,product(basis,value,transpose=True))

        groups = []
        design_buffers = {}
        for t in traits:
            shape = (len(t.rows),min(block_size,len(t.variants))*t.phi.shape[1])
            design_buffers[t.id] = (np.empty(shape,order='F'),np.empty(shape,order='F'))
            by_surface = {}
            for c in t.candidates:
                by_surface.setdefault(array_digest(c.residual), []).append(c)
            for candidates in by_surface.values():
                scale = np.sqrt(candidates[0].residual)
                fixed = t.fixed/scale[:, None]
                basis = np.asfortranarray(thin_rank_revealing_fixed_effect_basis(fixed, rtol=solver.qr_rtol))
                if basis.shape[1] >= len(t.rows):
                    raise ValueError('fixed effects exhaust the sample space')
                if basis.shape[1] and not np.allclose(product(basis,basis,transpose=True),
                        np.eye(basis.shape[1]),rtol=0,atol=1e-10):
                    raise FloatingPointError('mixture fixed-effect basis is not orthonormal')
                yw = t.y/scale
                k, m, q = len(candidates), len(t.variants), t.phi.shape[1]
                workspace = None
                if op.native is None:
                    yp = yw-fixed_projection(basis,yw[:,None])[:,0]
                else:
                    workspace = op.native.PredictionMixtureResidual(np.ascontiguousarray(yw),basis,k,threads)
                    yp = np.empty_like(yw)
                    workspace.copy_phenotype(yp)
                groups.append(dict(trait=t, candidates=candidates, scale=scale, fixed=fixed, basis=basis,
                    yw=yw, yp=yp, weights=np.zeros((k, m, q)), penalty=np.zeros((k, m)),
                    workspace=workspace, weighted_phi=np.asfortranarray(t.phi/scale[:,None]),
                    update_buffer=np.empty((len(t.rows),k),order='F'),
                    fixed_update_buffer=np.empty((len(t.rows),k),order='F'),
                    residual=np.asfortranarray(np.repeat(yp[:, None], k, axis=1)), cache={},
                    threshold=max(solver.atol, solver.rtol*np.linalg.norm(yp))))

        def design(group, variants, raw):
            t = group['trait']
            lo, hi, g = op._group_block(op.trait_group[t.id], variants, raw)
            if g is None:
                return lo, hi, None
            w = design_buffers[t.id][0][:,:(hi-lo)*t.phi.shape[1]]
            if op.native is None:
                for r in range(t.phi.shape[1]):
                    np.multiply(g,group['weighted_phi'][:,r:r+1],out=w[:,r::t.phi.shape[1]])
            else:
                op.native.prediction_interaction_design(g,group['weighted_phi'],w,threads)
            return lo, hi, w

        for _, variants, raw in op.stream.blocks('mixture_setup', build_cache=storage == 'compact'):
            for group in groups:
                lo, hi, w = design(group, variants, raw)
                if w is None:
                    continue
                t = group['trait']; q = t.phi.shape[1]; m = len(t.variants)
                # Cache the small projection once. Subsequent passes apply
                # P W to candidate vectors, never to every SNP column again.
                projection = (product(group['basis'], w, transpose=True)
                    if group['basis'].shape[1] else np.empty((0,w.shape[1])))
                projected = design_buffers[t.id][1][:,:w.shape[1]]
                if group['basis'].shape[1]:
                    product_into(group['basis'],projection,projected)
                    np.subtract(w,projected,out=projected)
                else:
                    projected[...] = w
                gram = np.ascontiguousarray(product(projected, projected, transpose=True))
                del projected
                diagonal = np.array([gram[j*q:(j+1)*q, j*q:(j+1)*q] for j in range(hi-lo)])
                posteriors, normalizers = [], []
                for c in group['candidates']:
                    covariance = (np.repeat(c.covariance[None], hi-lo, axis=0) if c.annotation_prior is None
                                  else c.annotation_prior.block(lo, hi).reshape(hi-lo, q, q))/m
                    pc, ln = _posterior(diagonal, covariance, mixtures[t.id, c.id],
                        4 if any(isinstance(mixtures[t.id,cc.id],SeparateSparsitySpec) for cc in group['candidates']) else 2)
                    posteriors.append(pc.ravel()); normalizers.append(ln.ravel())
                group['cache'][lo] = dict(gram=gram, covariance=np.ascontiguousarray(posteriors),
                                        normalizer=np.ascontiguousarray(normalizers), projection=projection)
        op.ready = True
        setup_seconds = time.monotonic()-started
        sweep, history, prior_seconds = 0, [], 0.
        if state is not None and resume:
            sweep, history, prior_seconds = state.load(groups)
            for group in groups:
                if group['workspace'] is not None:
                    group['workspace'].restore(group['residual'])

        def update(group, lo, hi, w, independent=False):
            q = group['trait'].phi.shape[1]; k = len(group['candidates']); cache = group['cache'][lo]
            workspace = group['workspace']
            if workspace is not None:
                score = np.empty((w.shape[1],k),order='F')
                workspace.score(w,cache['projection'],score)
            else:
                score = product(w, group['residual'], transpose=True)
                if group['basis'].shape[1]:
                    score -= product(cache['projection'],
                        product(group['basis'], group['residual'], transpose=True), transpose=True)
            score = np.ascontiguousarray(score.T)
            # A one-candidate slice can already be contiguous; retain a copy
            # before assigning the new weights so the residual update is real.
            old = np.array(group['weights'][:, lo:hi].reshape(k, -1), order='C', copy=True)
            beta = old.copy(); penalty = np.empty((k, hi-lo))
            tolerance = group['threshold']/max(1., np.sqrt(len(group['trait'].variants)))
            arguments = (cache['gram'], score, cache['covariance'], cache['normalizer'], beta, penalty)
            if op.native is None:
                metrics = _numpy_update(*arguments, q, solver.block_sweeps, tolerance, independent)
            else:
                metrics = np.empty((k, 2))
                op.native.prediction_mixture_block(*arguments, metrics, q, solver.block_sweeps,
                                                  tolerance, independent, threads)
            if not independent:
                group['weights'][:, lo:hi] = beta.reshape(k, hi-lo, q)
                group['penalty'][:, lo:hi] = penalty
                delta = (beta-old).T
                if workspace is not None:
                    workspace.update(w,np.asfortranarray(delta),cache['projection'])
                else:
                    product_into(w,delta,group['update_buffer'])
                    group['residual'] -= group['update_buffer']
                    if group['basis'].shape[1]:
                        product_into(group['basis'],product(cache['projection'],delta),group['fixed_update_buffer'])
                        group['residual'] += group['fixed_update_buffer']
            return metrics[:, 0]

        def reconstruct():
            for group in groups:
                group['raw_prediction'] = np.zeros_like(group['residual'], order='F')
                if group['workspace'] is not None:
                    group['workspace'].begin_reconstruction()
            for variants, raw in op.blocks('mixture_reconstruct'):
                for group in groups:
                    lo, hi, w = design(group, variants, raw)
                    if w is not None:
                        weights = group['weights'][:, lo:hi].reshape(len(group['candidates']), -1)
                        if group['workspace'] is not None:
                            group['workspace'].add_prediction(w,np.asfortranarray(weights.T))
                        else:
                            product_into(w,weights.T,group['update_buffer'])
                            group['raw_prediction'] += group['update_buffer']
            for group in groups:
                if group['workspace'] is not None:
                    # Retain the incremental state until the independent
                    # reconstruction passes. A failed guard must not discard
                    # a costly full-data sweep or its diagnostic evidence.
                    group['workspace'].copy_residual(group['update_buffer'])
                    group['residual_drift'] = group['workspace'].finish_reconstruction()
                    group['workspace'].copy_residual(group['residual'])
                    group['workspace'].copy_prediction(group['raw_prediction'])
                    continue
                basis = group['basis']; fitted = group['raw_prediction']
                residual = group['yp'][:, None]-fitted+fixed_projection(basis,fitted)
                group['residual_drift'] = float(np.linalg.norm(residual-group['residual']))
                group['residual'][...] = residual

        def verify():
            reconstruct()
            errors = [np.zeros(len(g['candidates'])) for g in groups]
            for variants, raw in op.blocks('mixture_verification'):
                for j, group in enumerate(groups):
                    lo, hi, w = design(group, variants, raw)
                    if w is not None:
                        errors[j] += update(group, lo, hi, w, independent=True)
            return [np.sqrt(x) for x in errors]

        if initial_weights is not None:
            for group in groups:
                for j,c in enumerate(group['candidates']):
                    group['weights'][j]=initial_weights[group['trait'].id,c.id]
            reconstruct()
        errors = None
        for sweep in range(sweep+1, solver.max_sweeps+1):
            previous = [g['weights'].copy() for g in groups]
            for variants, raw in op.blocks('mixture_sweep'):
                for group in groups:
                    lo, hi, w = design(group, variants, raw)
                    if w is not None:
                        update(group, lo, hi, w)
            refreshed = sweep == 1 or sweep % solver.residual_refresh == 0
            if not refreshed:
                for group in groups:
                    if group['workspace'] is not None:
                        group['workspace'].copy_residual(group['residual'])
            if refreshed:
                reconstruct()
                if any(g['residual_drift'] > 1e-9*max(np.linalg.norm(g['yp']),1.) for g in groups):
                    if state is not None:
                        failure = state.path.with_name(state.path.stem+'.arithmetic_failure.npz')
                        arrays = {}
                        for j,g in enumerate(groups):
                            for name in ('weights','residual','raw_prediction','yp','penalty'):
                                arrays[f'{name}_{j}'] = g[name]
                            if g['workspace'] is not None:
                                arrays[f'incremental_residual_{j}'] = g['update_buffer']
                        metadata = dict(kind='mixture_arithmetic_failure',schema=1,
                            passed=False,resumable=False,sweep=sweep,identity=identity,
                            elapsed=prior_seconds+time.monotonic()-started,setup_seconds=setup_seconds,
                            drifts=[float(g['residual_drift']) for g in groups],
                            phenotype_norms=[float(np.linalg.norm(g['yp'])) for g in groups],
                            arrays={key:dict(shape=list(value.shape),sha256=array_digest(value)) for key,value in arrays.items()})
                        arrays['metadata'] = np.frombuffer(canonical(metadata).encode(),dtype=np.uint8)
                        with failure.open('xb') as stream:
                            np.savez(stream,**arrays)
                            stream.flush();os.fsync(stream.fileno())
                    raise FloatingPointError('mixture residual drift exceeds arithmetic tolerance: '+
                        repr([(g['trait'].id,g['residual_drift'],float(np.linalg.norm(g['yp']))) for g in groups]))
            bounds = np.concatenate([-.5*np.sum(g['residual']**2, axis=0)-g['penalty'].sum(axis=1) for g in groups])
            if not np.isfinite(bounds).all():
                raise FloatingPointError('nonfinite mixture objective')
            if history and np.any(bounds < np.array(history[-1])-1e-8*np.maximum(abs(bounds), 1)):
                raise FloatingPointError('mixture variational objective decreased')
            history.append(bounds.tolist())
            change = max(np.linalg.norm(g['weights']-old)/max(np.linalg.norm(g['weights']), 1e-30)
                         for g, old in zip(groups, previous))
            del previous
            if state is not None:
                state.save(groups, sweep, history, prior_seconds+time.monotonic()-started)
            if progress is not None:
                progress(dict(sweep=sweep, elbo=bounds.tolist(), relative_weight_change=float(change),
                              seconds=time.monotonic()-started, setup_seconds=setup_seconds,
                              residual_refresh_error=([g['residual_drift'] for g in groups] if refreshed else None)))
            if change < max(solver.rtol*10, 1e-9) or sweep == solver.max_sweeps:
                errors = verify()
                if all(np.all(e <= g['threshold']) for e, g in zip(errors, groups)):
                    break
        if errors is None:
            errors = verify()
        reports, fixed = {}, {}
        for group, error in zip(groups, errors):
            t = group['trait']; basis = group['basis']
            residual = group['yw'][:, None]-group['raw_prediction']
            alpha = (linalg.lstsq(product(basis,group['fixed'],transpose=True), product(basis,residual,transpose=True), cond=0., lapack_driver='gelsy')[0]
                     if basis.shape[1] else np.zeros((t.fixed.shape[1], len(group['candidates']))))
            projection = (np.linalg.norm(product(basis,group['residual'],transpose=True), axis=0)/np.maximum(np.linalg.norm(group['residual'], axis=0), 1e-30)
                if basis.shape[1] else np.zeros(len(group['candidates'])))
            for j, c in enumerate(group['candidates']):
                key = t.id, c.id
                good = bool(error[j] <= group['threshold'] and projection[j] <= max(1e-12, 10*solver.qr_rtol))
                reports[key] = dict(converged=good, method='mixture_vb_fixed_point', iterations=sweep,
                    true_residual_norm=float(error[j]), threshold=float(group['threshold']),
                    fixed_projection=float(projection[j]), fixed_rank=basis.shape[1],
                    residual_reconstruction_error=group['residual_drift'],
                    reason='frozen_simultaneous_fixed_point' if good else 'fixed_point_failed')
                fixed[key] = alpha[:, j]
        if not all(r['converged'] for r in reports.values()):
            raise ConvergenceError(reports)
        result = SolveResult({}, fixed, reports, prior_seconds+time.monotonic()-started)
        provenance = dict(source=source.identity, backend=backend, arithmetic='affine64_fp64',
            method='block_multivariate_mixture_variational_bayes', fixed_effects='integrated',
            prediction_python_sha256=python_identity, native_build=None if op.native is None else op.native.build_info(),
            products='deterministic_native' if op.native is not None else 'numpy',
            warm_start=(state.initialization is not None if state is not None else initialization is not None),
            initialization_sha256=(state.initialization if state is not None else initialization),
            solver=asdict(solver), fit_identity=identity)
        writer = ModelWriter(output, traits, source, result, provenance)
        owner.callback(writer.close)
        for group in groups:
            t = group['trait']
            for j, c in enumerate(group['candidates']):
                for lo in range(0, len(t.variants), block_size):
                    hi = min(lo+block_size, len(t.variants))
                    writer.write((t.id, c.id), lo, hi, group['weights'][j, lo:hi])
        writer.finish(dict(plan=plan.to_dict(), ledger=asdict(op.ledger),
            setup_seconds=setup_seconds, elapsed_seconds=time.monotonic()-started,
            cumulative_seconds=result.elapsed_seconds, objective_history=history,
            resumed_solver=bool(resume), solver_checkpoint_enabled=state is not None))
    return load_prediction_models(output)
