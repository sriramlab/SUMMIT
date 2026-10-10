"""Streamed research implementation of the conditional polygenic calculation.

No sample covariance or full standardized genotype matrix is constructed.
Estimated-covariance calibration remains under investigation. Genotypes,
products, projection, CG and checkpoints reuse
SUMMIT's existing implementation.
"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.prediction._validation import array_digest, digest
from summit.prediction.genotype import RawBlockStream, StandardizedBlock, native_module
from summit.prediction.operator import GenotypeOperator
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.solver import solve
from summit.prediction.spec import SolverSpec


def fit_kernel_scales(source, rows, variants, *, threads=1, block_size=128, memory_bytes=4*2**30):
    """Training-only additive and heterozygote scales, same imputation rule."""
    if not source.hard_calls:
        raise ValueError("heterozygote covariance requires observed hard calls")
    if 40*len(rows)*min(block_size, len(variants))+32*len(variants)+256*2**20 > memory_bytes:
        raise MemoryError("additive/dominance scaling exceeds memory budget")
    native = native_module()
    configure_prediction_threads(native, threads)
    stream = RawBlockStream(source, rows, variants, block_size=block_size, threads=threads)
    means = np.empty((2, len(stream.variants)))
    inverse = np.empty_like(means)
    for start, selected, raw in stream.blocks("polygenic_scale"):
        end = start + len(selected)
        heterozygote = np.asfortranarray(np.where(raw == -127, -127, raw == 1), dtype=np.int8)
        for i, value in enumerate((raw, heterozygote)):
            native.prediction_hardcall_scale(value, means[i, start:end], inverse[i, start:end], 1, threads)
    return dict(mean=means, inverse_scale=inverse, source=source.identity,
        variants=source.variants.subset(stream.variants).identity,
        samples=digest([source.samples[i] for i in stream.rows]),
        definition="additive_and_observed_heterozygote_training_mean_imputed_ddof1")


class PolygenicKernels:
    """Additive, prespecified covariate-dependent additive, dominance and noise.

    Every additive context has its own nonnegative covariance component.
    Contexts and diagonal variance surfaces must be fixed without outcomes.
    Missing calls use the saved training means in both cohorts.
    """
    product = GenotypeOperator.product

    def __init__(self, source, rows, variants, scales, contexts, noise, *,
                 threads=1, block_size=128, memory_bytes=4*2**30, storage="stream", rhs_columns=None):
        self.native = native_module()
        configure_prediction_threads(self.native, threads)
        if not source.hard_calls or storage not in ("stream", "packed"):
            raise ValueError("research covariance requires hard calls and stream/packed storage")
        self.rows, self.variants = np.array(rows, copy=True), np.array(variants, copy=True)
        if (np.any(np.diff(self.rows) <= 0) or np.any(np.diff(self.variants) <= 0)
                or len(self.rows) < 2 or not len(self.variants)):
            raise ValueError("nonempty ordered participant and variant axes required")
        if (scales["source"] != source.identity
                or scales["variants"] != source.variants.subset(self.variants).identity):
            raise ValueError("frozen kernel scale source/variant identity differs")
        self.contexts, self.noise = np.array(contexts, float, copy=True), np.array(noise, float, copy=True)
        if (self.contexts.ndim != 2 or self.noise.ndim != 2
                or len(self.contexts) != len(rows) or len(self.noise) != len(rows)
                or self.contexts.shape[1] < 1 or self.noise.shape[1] < 1
                or not np.array_equal(self.contexts[:, 0], np.ones(len(rows)))
                or not np.all(np.isfinite(self.contexts)) or not np.all(np.isfinite(self.noise))
                or np.any(self.noise < 0)):
            raise ValueError("finite aligned contexts and nonnegative variance surfaces required")
        self.mean, self.inverse = (np.array(scales[k], float, copy=True) for k in ("mean", "inverse_scale"))
        if (self.mean.shape != (2, len(variants)) or self.inverse.shape != self.mean.shape
                or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.inverse))
                or np.any(self.inverse <= 0)):
            raise ValueError("invalid frozen additive/dominance scales")
        self.count = self.contexts.shape[1] + 1 + self.noise.shape[1]
        if rhs_columns is not None and (type(rhs_columns) is not int or rhs_columns < self.contexts.shape[1]):
            raise ValueError('RHS tile must hold all declared contexts')
        self.rhs_columns = rhs_columns
        self.memory_bytes = memory_bytes
        self.cache_bytes = ((len(rows)+3)//4)*len(variants) if storage == "packed" else 0
        self.base_bytes = (self.cache_bytes + 64*len(rows)*min(block_size, len(variants))
            + self.contexts.nbytes + self.noise.nbytes + self.mean.nbytes + self.inverse.nbytes
            + 8*self.count*len(rows) + 128*(len(source.samples)+len(source.variants.ids))
            + 256*2**20)
        if self.base_bytes >= memory_bytes:
            raise MemoryError("polygenic genotype workspace exceeds memory budget")
        self.plan = SimpleNamespace(threads=threads)
        self.workspace = self.native.PredictionWorkspace()
        self.standard = StandardizedBlock(self.native, threads)
        self.stream = RawBlockStream(source, rows, variants, block_size=block_size,
            threads=threads, native=self.native, storage=storage)
        self.identity = digest([source.identity, array_digest(self.rows), scales["samples"],
            array_digest(self.variants), array_digest(self.mean), array_digest(self.inverse),
            array_digest(self.contexts), array_digest(self.noise), scales["definition"]])
        self._diagonal = None
        # Training needs a packed cache and a Jacobi diagonal. Confirmation
        # products need neither: do not traverse its complete marker axis just
        # to calculate an unused preconditioner. A later solve requests the
        # same exact diagonal lazily, with a separately recorded traversal.
        if storage == 'packed':
            self._prepare_diagonal(build_cache=True)
        for value in (self.rows,self.variants,self.contexts,self.noise,self.mean,self.inverse):
            value.setflags(write=False)

    @property
    def diagonal(self):
        if self._diagonal is None:
            self._prepare_diagonal()
        return self._diagonal

    def _prepare_diagonal(self, *, build_cache=False):
        if self._diagonal is not None:
            raise RuntimeError('covariance diagonal is already prepared')
        diagonal = np.zeros((self.count, len(self.rows)))
        for start, selected, raw in self.stream.blocks("polygenic_setup", build_cache=build_cache):
            for kind in range(2):
                g = self._block(raw, start, len(selected), kind)
                row_square = np.einsum("ij,ij->i", g, g)/len(self.variants)
                if kind == 0:
                    diagonal[:self.contexts.shape[1]] += (self.contexts**2*row_square[:, None]).T
                else:
                    diagonal[self.contexts.shape[1]] += row_square
        diagonal[-self.noise.shape[1]:] = self.noise.T
        diagonal.setflags(write=False)
        self._diagonal = diagonal

    def _block(self, raw, start, width, kind):
        if kind:
            raw = np.asfortranarray(np.where(raw == -127, -127, raw == 1), dtype=np.int8)
        return self.standard.prepare(raw, np.arange(len(self.rows)), np.arange(width),
            self.mean[kind, start:start+width], self.inverse[kind, start:start+width])

    def apply(self, vectors, coefficients=None, *, phase="polygenic_product"):
        v = np.asarray(vectors, float)
        if v.ndim != 2 or len(v) != len(self.rows) or not np.all(np.isfinite(v)):
            raise ValueError("finite aligned matrix of covariance right-hand sides required")
        n, b = v.shape
        q = self.contexts.shape[1]
        theta = None if coefficients is None else np.asarray(coefficients, float)
        if theta is not None and (theta.shape != (self.count, b)
                or not np.all(np.isfinite(theta)) or np.any(theta < 0)):
            raise ValueError("nonnegative kernel coefficients required per right-hand side")
        available=self.memory_bytes-self.base_bytes-8*n*b*(self.count+4)
        affordable=int(available//(8*n*(8*q+8)))
        if affordable<1:
            raise MemoryError('polygenic covariance products exceed memory budget; reduce RHS count')
        width=min(b,affordable) if self.rhs_columns is None else max(1,self.rhs_columns//q)
        needed = self.base_bytes + 8*n*(b*(self.count+4)+min(b,width)*(8*q+8))
        if needed > self.memory_bytes:
            raise MemoryError("polygenic covariance products exceed memory budget; reduce RHS count")
        out = np.zeros((self.count, n, b)) if theta is None else np.zeros(v.shape,order='F')
        self.stream.ledger.operator_calls += 1
        active = np.arange(q) if theta is None else np.flatnonzero(np.any(theta[:q] != 0, axis=1))
        dominance_active = theta is None or np.any(theta[q] != 0)
        self.stream.ledger.active_rhs.append((len(active)+int(dominance_active))*b)
        if theta is not None:
            # The existing prediction kernel mixes each RHS's diagonal context
            # prior and accumulates its product without a Python N x Q x B array.
            phi=np.asfortranarray(self.contexts[:,active])
            one=np.ones((n,1),order='F')
        blocks = self.stream.blocks(phase) if len(active) or dominance_active else ()
        for start, selected, raw in blocks:
            for kind in range(2):
                if (kind == 0 and not len(active)) or (kind == 1 and not dominance_active):
                    continue
                g = self._block(raw, start, len(selected), kind)
                for begin in range(0,b,width):
                    end=min(b,begin+width)
                    if theta is not None:
                        context=phi if kind==0 else one
                        qk=context.shape[1]
                        packed=np.asfortranarray((v[:,begin:end,None]*context[:,None,:]).reshape(n,-1))
                        prior=np.zeros((end-begin,qk,qk))
                        coefficients=theta[active,begin:end].T if kind==0 else theta[q,begin:end,None]
                        prior[:,np.arange(qk),np.arange(qk)]=coefficients
                        self.native.prediction_covariance_block(g,packed,prior.reshape(end-begin,-1),
                            context,out[:,begin:end],float(len(self.variants)),self.plan.threads,self.workspace)
                        continue
                    left=(self.contexts[:,active,None]*v[:,None,begin:end]).reshape(n,len(active)*(end-begin)) if kind==0 else v[:,begin:end]
                    inner = self.product(g, left, transpose=True)/len(self.variants)
                    product = self.product(g, inner)
                    if kind == 0:
                        product = product.reshape(n, len(active), end-begin)
                        for column, k in enumerate(active):
                            value = product[:, column]*self.contexts[:, k, None]
                            out[k,:,begin:end] += value
                    else:
                        out[q,:,begin:end] += product
        for j in range(self.noise.shape[1]):
            k = q+1+j
            value = self.noise[:, j, None]*v
            if theta is None:
                out[k] = value
            else:
                out += value*theta[k]
        return out

    def quadratic_forms(self, vectors, *, phase="polygenic_quadratic_forms"):
        """All component quadratics in one pass, without forward products.

        Workspace scales with a genotype block and the supplied contrasts,
        never with the square of the participant count.
        """
        v = np.asarray(vectors, float)
        if v.ndim != 2 or len(v) != len(self.rows) or not np.all(np.isfinite(v)):
            raise ValueError('finite aligned covariance contrasts required')
        n, b = v.shape
        q = self.contexts.shape[1]
        if self.base_bytes + 8*n*b*(q+2) > self.memory_bytes:
            raise MemoryError('covariance quadratic forms exceed memory budget')
        out = np.zeros((self.count, b))
        self.stream.ledger.operator_calls += 1
        self.stream.ledger.active_rhs.append((q+1)*b)
        for start, selected, raw in self.stream.blocks(phase):
            for kind in range(2):
                g = self._block(raw, start, len(selected), kind)
                # One trait at a time bounds context packing independently of
                # batch size; every trait still shares the genotype traversal.
                for j in range(b):
                    packed = self.contexts*v[:,j,None] if kind == 0 else v[:,j,None]
                    inner = self.product(g, packed, transpose=True)
                    value = np.sum(inner*inner, axis=0)/len(self.variants)
                    if kind == 0:
                        out[:q,j] += value
                    else:
                        out[q,j] += value[0]
        out[q+1:] = self.noise.T@(v*v)
        return out

    def cross_products(self, vectors, input_index, output_index, coefficients=None,
                       *, phase='polygenic_cross_product'):
        """K[output,input] times vectors and each input quadratic form.

        This avoids products against zero-padded cohort rows. The quadratic
        form uses marker inner products, without a forward multiply on input
        rows. Both disjoint and overlapping row selections retain the declared
        diagonal-noise contribution. No marker weights are stored globally.
        """
        v=np.asarray(vectors,float)
        take,give=map(np.asarray,(input_index,output_index))
        n=len(self.rows);q=self.contexts.shape[1]
        for index in (take,give):
            if (index.ndim!=1 or not np.issubdtype(index.dtype,np.integer)
                    or not len(index) or np.any(np.diff(index)<=0)
                    or index[0]<0 or index[-1]>=n):
                raise ValueError('ordered unique covariance row selections required')
        if v.ndim!=2 or len(v)!=len(take) or not np.all(np.isfinite(v)):
            raise ValueError('finite covariance inputs must align with input rows')
        b=v.shape[1]
        theta=None if coefficients is None else np.asarray(coefficients,float)
        if theta is not None and (theta.shape!=(self.count,b)
                or not np.all(np.isfinite(theta)) or np.any(theta<0)):
            raise ValueError('nonnegative covariance coefficients required per column')
        # A weighted action retains one summed output per RHS. Only the
        # unweighted path retains all component outputs. Keep the existing
        # temporary-buffer allowance and tile bound in both cases.
        output_count=self.count if theta is None else 1
        available=self.memory_bytes-self.base_bytes-8*n*b*(output_count+4)
        width=min(b,int(available//(8*n*(8*q+8))))
        if width<1:
            raise MemoryError('cross-cohort covariance products exceed memory budget')
        if self.rhs_columns is not None:
            width=min(width,max(1,self.rhs_columns//q))
        out=np.zeros((self.count,len(give),b)) if theta is None else np.zeros((len(give),b))
        quadratic=np.zeros((self.count,b)) if theta is None else np.zeros(b)
        active=np.arange(q) if theta is None else np.flatnonzero(np.any(theta[:q]!=0,axis=1))
        dominance=theta is None or np.any(theta[q]!=0)
        phi0=self.contexts[take][:,active];phi1=self.contexts[give][:,active]
        input_standard=StandardizedBlock(self.native,self.plan.threads)
        output_standard=StandardizedBlock(self.native,self.plan.threads)
        self.stream.ledger.operator_calls+=1
        self.stream.ledger.active_rhs.append((len(active)+int(dominance))*b)
        blocks=self.stream.blocks(phase) if len(active) or dominance else ()
        mass=float(len(self.variants))
        for start,selected,raw in blocks:
            for kind in range(2):
                if (kind==0 and not len(active)) or (kind==1 and not dominance):
                    continue
                calls=raw if kind==0 else np.asfortranarray(np.where(raw==-127,-127,raw==1),dtype=np.int8)
                columns=np.arange(len(selected))
                mean=self.mean[kind,start:start+len(selected)]
                inverse=self.inverse[kind,start:start+len(selected)]
                # Native column-wise gathers avoid a large row-major advanced
                # indexing copy followed by a second Fortran conversion.
                g0=input_standard.prepare(calls,take,columns,mean,inverse)
                g1=output_standard.prepare(calls,give,columns,mean,inverse)
                for begin in range(0,b,width):
                    end=min(b,begin+width);count=end-begin
                    packed=(phi0[:,:,None]*v[:,None,begin:end]).reshape(len(take),-1) if kind==0 else v[:,begin:end]
                    inner=self.product(g0,packed,transpose=True)/mass
                    if kind==0:
                        inner=inner.reshape(len(selected),len(active),count)
                        diagonal=mass*np.sum(inner**2,axis=0)
                        if theta is None:
                            quadratic[active,begin:end]+=diagonal
                        else:
                            quadratic[begin:end]+=np.sum(diagonal*theta[active,begin:end],axis=0)
                            inner*=theta[active,begin:end][None,:,:]
                        product=self.product(g1,inner.reshape(len(selected),-1)).reshape(len(give),len(active),count)
                        product*=phi1[:,:,None]
                        if theta is None:
                            for column,k in enumerate(active):
                                out[k,:,begin:end]+=product[:,column]
                        else:
                            out[:,begin:end]+=product.sum(axis=1)
                    else:
                        diagonal=mass*np.sum(inner**2,axis=0)
                        if theta is None:
                            quadratic[q,begin:end]+=diagonal
                            out[q,:,begin:end]+=self.product(g1,inner)
                        else:
                            quadratic[begin:end]+=diagonal*theta[q,begin:end]
                            out[:,begin:end]+=self.product(g1,inner*theta[q,begin:end])
        common,source_index,target_index=np.intersect1d(take,give,assume_unique=True,return_indices=True)
        for j in range(self.noise.shape[1]):
            k=q+1+j
            diagonal=np.sum(v*v*self.noise[take,j,None],axis=0)
            value=v[source_index]*self.noise[common,j,None]
            if theta is None:
                quadratic[k]=diagonal
                out[k,target_index]+=value
            else:
                quadratic+=diagonal*theta[k]
                out[target_index]+=value*theta[k]
        return out,quadratic


def project(basis, values):
    return values-basis@(basis.T@values)


def he_geometry(operator, fixed, *, probes=128, seed=871631, exact=False, keep_products=True,
                moment_weighting='none'):
    """Genotype-only projected kernel traces; exact option is bounded reference."""
    n = len(operator.rows)
    if not isinstance(probes, int) or probes < 2:
        raise ValueError("at least two genotype-only covariance probes required")
    if exact and n > 2048:
        raise ValueError("identity-probe covariance reference is bounded to N<=2048")
    count=n if exact else probes
    if operator.base_bytes+8*n*count*((1+bool(keep_products))*operator.count+3)+256*2**20>operator.memory_bytes:
        raise MemoryError('covariance trace and preconditioner sketches exceed memory budget')
    if moment_weighting not in ('none','genotype_diagonal'):
        raise ValueError('unknown covariance moment weighting')
    fixed = np.asarray(fixed,float)
    if fixed.ndim!=2 or len(fixed)!=n:
        raise ValueError('aligned fixed-effect matrix required for covariance moments')
    weights = None
    if moment_weighting == 'genotype_diagonal':
        diagonal = operator.diagonal
        mass = diagonal.mean(axis=1)
        if np.any(mass<=0) or not np.all(np.isfinite(mass)):
            raise ValueError('positive finite covariance component diagonal mass required')
        working = (diagonal/mass[:,None]).mean(axis=0)
        if np.any(working<=0) or not np.all(np.isfinite(working)):
            raise ValueError('positive finite working covariance diagonal required')
        weights = 1/np.sqrt(working)
    u = thin_rank_revealing_fixed_effect_basis(
        fixed if weights is None else np.asarray(fixed)*weights[:,None], rtol=1e-11)
    z = np.eye(n) if exact else np.random.default_rng(seed).choice([-1., 1.], (n, probes))
    z = project(u, z)
    if weights is not None:
        z *= weights[:,None]
    products = operator.apply(z, phase="polygenic_trace")
    unprojected=products.copy() if keep_products else None
    for value in products:
        if weights is not None:
            value *= weights[:,None]
        value[:] = project(u, value)
    # Each probe gives a PSD Gram contribution. Common probes preserve symmetry.
    probe_gram = np.einsum("anb,cnb->bac", products, products)
    gram = probe_gram.sum(0)/(1 if exact else probes)
    norms = np.sqrt(np.diag(gram))
    if np.any(norms <= 0):
        raise ValueError("absorbed covariance component")
    h = gram/norms[:, None]/norms[None, :]
    if np.linalg.cond(h) > 1e8:
        raise ValueError("covariance components are not separately identifiable")
    result = dict(basis=u, gram=gram, norms=norms, h=h,
        probe_gram=probe_gram,
        operator_identity=operator.identity, probes=n if exact else probes,
        seed=None if exact else seed, exact=exact)
    if keep_products:
        result.update(preconditioner_products=unprojected,preconditioner_probes=z)
    if weights is not None:
        # The covariance model and downstream solves stay in original units.
        # Only the phenotype-independent moment equations use D K D and D C.
        result.update(moment_weighting=moment_weighting,moment_weights=weights)
    return result


class NystromPreconditioner:
    """An approximate inverse for computation, never a fitted null covariance.

    Reuse the genotype-only HE sketches U_k=K_k Z. For a fitted theta, the
    Nystrom matrix is U (Z' U)^+ U'. Its diagonal remainder plus a positive
    computational floor gives a diagonal-plus-low-rank preconditioner. The
    actual solve and its residual check must still use the full marker operator.
    """
    def __init__(self,operator,geometry,theta):
        if geometry['operator_identity']!=operator.identity:
            raise ValueError('preconditioner genotype reference changed')
        theta=np.asarray(theta,float)
        if theta.shape!=(operator.count,) or np.any(theta<0) or not np.all(np.isfinite(theta)):
            raise ValueError('aligned nonnegative preconditioner coefficients required')
        z=np.asarray(geometry['preconditioner_probes'])
        products=np.asarray(geometry['preconditioner_products'])
        if (z.ndim!=2 or len(z)!=len(operator.rows) or products.shape!=(operator.count,*z.shape)
                or not np.all(np.isfinite(z)) or not np.all(np.isfinite(products))):
            raise ValueError('finite aligned genotype-only preconditioner sketches required')
        u=np.einsum('knb,k->nb',products,theta)
        w=z.T@u; w=(w+w.T)/2
        eigen,vectors=np.linalg.eigh(w)
        tolerance=64*len(w)*np.finfo(float).eps*max(float(eigen[-1]),np.finfo(float).tiny)
        if eigen[0]<-tolerance:
            raise ValueError('indefinite preconditioner sketch')
        keep=eigen>tolerance
        self.low_rank=u@(vectors[:,keep]/np.sqrt(eigen[keep]))
        diagonal=operator.diagonal.T@theta
        if np.any(diagonal<=0):
            raise ValueError('positive covariance diagonal required for preconditioning')
        remainder=diagonal-np.sum(self.low_rank**2,axis=1)
        if np.any(remainder < -1e-8*diagonal):
            raise ValueError('Nystrom sketch and covariance diagonal disagree')
        # This floor regularizes only the computational inverse. It never
        # enters phenotype covariance, coefficient uncertainty or a P value.
        self.remainder=np.maximum(remainder,.01*diagonal)
        self.weighted=self.low_rank/self.remainder[:,None]
        from scipy.linalg import cho_factor
        self.factor=cho_factor(np.eye(self.low_rank.shape[1])+self.low_rank.T@self.weighted,lower=True)
        self.identity=digest([operator.identity,array_digest(theta),array_digest(z),
            array_digest(products),'nystrom_diagonal_remainder_001_v1'])
        self.operator_identity=operator.identity
        self.theta=np.array(theta,copy=True)
        self.theta.setflags(write=False)
        self.nbytes=sum(v.nbytes for v in (self.low_rank,self.remainder,self.weighted,self.factor[0]))
        for value in (self.low_rank,self.remainder,self.weighted,self.factor[0]):
            value.setflags(write=False)

    def apply(self,values):
        from scipy.linalg import cho_solve
        v=np.asarray(values,float)
        if v.ndim not in (1,2) or len(v)!=len(self.remainder) or not np.all(np.isfinite(v)):
            raise ValueError('finite aligned preconditioner right-hand sides required')
        scaled=v/self.remainder if v.ndim==1 else v/self.remainder[:,None]
        return scaled-self.weighted@cho_solve(self.factor,self.low_rank.T@scaled)


def prepare_preconditioners(operator,geometry,coefficients):
    """Share identical covariance inverses within a bounded fitting batch."""
    theta=np.asarray(coefficients,float)
    if theta.ndim!=2 or theta.shape[0]!=operator.count:
        raise ValueError('one covariance coefficient column per trait required')
    owned={}
    result=[]
    for column in theta.T:
        key=array_digest(column)
        if key not in owned:
            projected_bytes=8*len(operator.rows)*(3*geometry['preconditioner_probes'].shape[1]+2)
            if (operator.base_bytes+sum(p.nbytes for p in owned.values())+projected_bytes
                    +256*2**20>operator.memory_bytes):
                raise MemoryError('conditional preconditioner batch exceeds memory budget')
            owned[key]=NystromPreconditioner(operator,geometry,column)
        result.append(owned[key])
    return result


def estimate_components(operator, y, geometry, *, return_uncertainty=False):
    """Fit nonnegative HE, optionally retaining a full or directional diagnostic.

    ``'directional'`` stores H^-1 B y for a later scalar variance contraction;
    unlike ``True`` it performs no extra genotype product at this stage.
    """
    if return_uncertainty not in (False,True,'directional'):
        raise ValueError('unknown covariance uncertainty representation')
    if geometry["operator_identity"] != operator.identity:
        raise ValueError("covariance reference inputs changed")
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    weights = geometry.get('moment_weights')
    if weights is None and geometry.get('moment_weighting','none')!='none':
        raise ValueError('missing covariance moment weights')
    if weights is None:
        residual = project(geometry["basis"], y)
    else:
        weights = np.asarray(weights,float)
        if (weights.shape!=(len(operator.rows),) or np.any(weights<=0)
                or not np.all(np.isfinite(weights))
                or geometry.get('moment_weighting')!='genotype_diagonal'):
            raise ValueError('invalid covariance moment weights')
        residual = weights[:,None]*project(geometry['basis'],weights[:,None]*y)
    products = operator.apply(residual, phase="polygenic_trait_moments")
    moments = np.einsum("nb,knb->kb", residual, products)/geometry["norms"][:, None]
    values, vectors = np.linalg.eigh(geometry["h"])
    root = np.sqrt(values)[:, None]*vectors.T
    rhs = (vectors.T@moments)/np.sqrt(values)[:, None]
    theta = np.column_stack([nnls(root, r)[0]/geometry["norms"] for r in rhs.T])
    if not return_uncertainty:
        return theta
    # B_k = T K_k T, T = D P_(DC) D. Reuse K_k T y from the fit.
    # With known Gaussian V, 2 (B_k y)' V (B_l y) is unbiased for
    # 2 tr(B_k V B_l V). Here V is fitted, so this is a plug-in diagnostic.
    for value in products:
        if weights is None:
            value[:] = project(geometry['basis'], value)
        else:
            value[:] = weights[:,None]*project(geometry['basis'],weights[:,None]*value)
    norms = geometry['norms']
    inverse = np.linalg.solve(geometry['h'],np.eye(operator.count))/norms[:,None]/norms[None,:]
    trace = []
    for j in range(y.shape[1]):
        if geometry['exact']:
            trace.append(np.zeros_like(inverse))
        else:
            # Independent genotype-only probe randomness, linearized at the
            # fitted theta. This measures Monte Carlo error, not sample error.
            shifts = (geometry['probe_gram']@theta[:,j])@inverse.T
            shifts -= shifts.mean(axis=0)
            count = len(shifts)
            trace.append(shifts.T@shifts/(count*(count-1)))
    if return_uncertainty == 'directional':
        if operator.base_bytes + 2*products.nbytes + residual.nbytes > operator.memory_bytes:
            raise MemoryError('covariance moment influences exceed memory budget')
        influence = (inverse@products.reshape(operator.count,-1)).reshape(products.shape)
        return theta, dict(influence=influence,trace=np.asarray(trace))
    # The full matrix is useful for bounded independent comparisons. Production
    # uses the equivalent directional contraction, avoiding K RHSs per trait.
    n = len(operator.rows)
    minimum = (operator.base_bytes + products.nbytes + residual.nbytes
        + 8*n*(operator.count*(operator.count+4)+8*operator.contexts.shape[1]+8))
    if minimum > operator.memory_bytes:
        raise MemoryError('covariance-component uncertainty exceeds memory budget; reduce trait batch size')
    sampling = []
    for j in range(y.shape[1]):
        score = products[:,:,j].T
        budget = operator.memory_bytes
        try:
            operator.memory_bytes -= products.nbytes + residual.nbytes
            applied = operator.apply(score,np.repeat(theta[:,j,None],operator.count,axis=1),
                phase='polygenic_covariance_precision')
        finally:
            operator.memory_bytes = budget
        omega = 2*score.T@applied
        covariance = inverse@omega@inverse.T
        sampling.append((covariance+covariance.T)/2)
    return theta, dict(sampling=np.asarray(sampling),trace=np.asarray(trace))


def conditional_variance_precision(complete, i0, i1, fitted, theta, uncertainty, *, training=None):
    """Local variance precision holding each fitted confirmation contrast fixed.

    Does not account for changing the learned contrast, NNLS boundary selection,
    model misspecification, or higher-order covariance-mean error. No P-value or
    standard-error correction follows from this diagnostic alone.
    """
    b = theta.shape[1]
    vectors = np.empty((len(complete.rows),b))
    vectors[i0] = fitted['training_contrasts']
    vectors[i1] = fitted['contrasts']
    gradients = complete.quadratic_forms(vectors,phase='polygenic_variance_precision')
    variance = fitted['variance']
    if not np.allclose(np.sum(gradients*theta,axis=0),variance,rtol=2e-7,atol=0):
        raise ValueError('conditional variance and component quadratic forms disagree')
    sampling = None
    if 'influence' in uncertainty:
        if (training is None or not np.array_equal(training.rows,complete.rows[i0])
                or uncertainty['influence'].shape != (training.count,len(i0),b)
                or not np.all(np.isfinite(uncertainty['influence']))):
            raise ValueError('directional precision requires aligned training moment influences')
        direction = np.einsum('knb,kb->nb',uncertainty['influence'],gradients)
        budget = training.memory_bytes
        try:
            training.memory_bytes -= uncertainty['influence'].nbytes + direction.nbytes
            applied = training.apply(direction,theta,phase='polygenic_directional_precision')
        finally:
            training.memory_bytes = budget
        sampling = 2*np.sum(direction*applied,axis=0)
    result = []
    for j in range(b):
        g = gradients[:,j]
        terms = {}
        for name in ('sampling','trace'):
            if name=='sampling' and sampling is not None:
                value = float(sampling[j])
                scale = float(2*np.sum(np.abs(direction[:,j]*applied[:,j])))
            else:
                value = float(g@uncertainty[name][j]@g)
                scale = float(np.abs(g)@np.abs(uncertainty[name][j])@np.abs(g))
            if not np.isfinite(value) or value < -1e-10*max(scale,np.finfo(float).tiny):
                raise ValueError('invalid estimated covariance-component uncertainty')
            terms[name+'_relative_sd'] = float(np.sqrt(max(0.,value))/variance[j])
        relative = float(np.hypot(*terms.values()))
        result.append(dict(terms,fixed_contrast_variance_relative_sd=relative,
            fixed_contrast_effective_df=None if relative==0 else 2/relative**2,
            nnls_boundary_components=np.flatnonzero(theta[:,j]==0).tolist(),
            method='Gaussian HE plug-in moment covariance plus independent trace-probe linearization',
            scope='local unconstrained HE approximation with fitted confirmation contrast fixed; '
                'does not qualify adaptive learning, NNLS boundaries, model misspecification or tail calibration; '
                'P values and confidence intervals are unchanged'))
    return result


def projected_solve(operator, y, fixed, coefficients, *, spec=SolverSpec(rtol=1e-8),
                    checkpoint=None, resume=False, preconditioners=None, recycle_spaces=None):
    """Reuse the native prediction solver and its verified residual/restart path."""
    y, theta = np.asarray(y, float), np.asarray(coefficients, float)
    if y.ndim == 1:
        y = y[:, None]
    if theta.ndim == 1:
        theta = np.repeat(theta[:, None], y.shape[1], axis=1)
    if theta.shape != (operator.count, y.shape[1]):
        raise ValueError("covariance coefficients must align with solver right-hand sides")
    if preconditioners is not None and (len(preconditioners) != y.shape[1]
            or any(p.operator_identity != operator.identity
                   or not np.array_equal(p.theta,theta[:,j]) for j,p in enumerate(preconditioners))):
        raise ValueError('preconditioners must match each fitted covariance')
    if recycle_spaces is not None and (len(recycle_spaces)!=y.shape[1]
            or any(p.operator_identity!=operator.identity or not np.array_equal(p.theta,theta[:,j])
                   for j,p in enumerate(recycle_spaces))):
        raise ValueError('recycling spaces must match each fitted covariance')
    diagonal = operator.diagonal.T@theta
    if np.any(diagonal <= 0):
        raise ValueError("estimated null covariance has nonpositive diagonal")
    traits = [SimpleNamespace(id=str(j), rows=operator.rows, y=y[:, j], fixed=fixed,
        phi=np.ones((len(y), 1)), candidates=[SimpleNamespace(id="null",
            covariance=np.zeros((1, 1)), residual=diagonal[:, j])]) for j in range(y.shape[1])]
    class Adapter:
        def apply(self, vectors, *, phase):
            keys = sorted(vectors)
            indices = [int(k[0]) for k in keys]
            product = operator.apply(np.column_stack([vectors[k] for k in keys]),
                theta[:, indices], phase=phase)
            if recycle_spaces is not None and phase in ('cg','cg_and_verification'):
                for j,key in enumerate(keys):
                    recycle_spaces[int(key[0])].add(vectors[key],product[:,j])
            return {key: product[:, i] for i, key in enumerate(keys)}
    adapter = Adapter()
    if preconditioners is not None:
        adapter.precondition = lambda key,value: preconditioners[int(key[0])].apply(value)
    adapter.traits = traits
    adapter.row_diagonal = {0: np.zeros(len(y))}
    adapter.trait_group = {t.id: 0 for t in traits}
    if checkpoint is None:
        if resume:
            raise ValueError("resume requires a checkpoint")
        result = solve(adapter, spec)
    else:
        from summit.prediction.checkpoint import SolverCheckpoint
        from summit.prediction.artifacts import file_digest
        from pathlib import Path
        import summit.prediction.solver as solver_module
        identity = digest([operator.identity, array_digest(y), array_digest(fixed),
            array_digest(theta), asdict(spec),file_digest(Path(__file__)),
            file_digest(Path(solver_module.__file__)),file_digest(Path(operator.native.__file__)),
            None if preconditioners is None else [p.identity for p in preconditioners]])
        with SolverCheckpoint(checkpoint, identity, resume=resume) as state:
            result = solve(adapter, spec, checkpoint=state)
    return np.column_stack([result.solutions[(str(j), "null")] for j in range(y.shape[1])]), result


def _panel_projected_solve(operator, rhs, fixed, theta):
    """Solve the identifiable training RHS span without iterating roundoff.

    A real LD panel can contain products exactly in the finite main-effect
    span. Relative CG residuals on their floating-point projection are not a
    meaningful accuracy target. Solve an orthonormal basis, reconstruct every
    original RHS, then verify those equations with a fresh covariance product.
    This changes no covariance or inferential threshold.
    """
    basis = thin_rank_revealing_fixed_effect_basis(fixed, rtol=1e-11)
    rhs = np.asarray(rhs, float)
    residual = project(basis, project(basis, rhs))
    original_norm = np.linalg.norm(rhs, axis=0)
    scale = np.where(original_norm > 0, original_norm, 1.)
    left, singular, _ = np.linalg.svd(residual/scale, full_matrices=False)
    keep = singular > 1e-11*max(1., singular[0] if len(singular) else 0.)
    if np.any(keep):
        selected = left[:, keep]
        # Basis errors can add on reconstruction. A two-order margin bounds
        # that accumulation for the at-most-513 RHS span; original equations
        # below retain the 1e-8 relative residual requirement.
        inverse, report = projected_solve(operator, selected, fixed, theta,
            spec=SolverSpec(rtol=1e-10))
        result = inverse@(selected.T@residual)
    else:
        result = np.zeros_like(rhs)
        report = SimpleNamespace(reports={})
    product = operator.apply(result, np.repeat(theta[:, None], rhs.shape[1], axis=1),
        phase='panel_reconstructed_residual')
    error = np.linalg.norm(project(basis, residual-product), axis=0)
    tolerance = 1e-8*np.linalg.norm(residual,axis=0)+1e-11*original_norm
    if np.any(error > tolerance):
        ratio=np.divide(error,tolerance,out=np.full_like(error,np.inf),where=tolerance>0)
        ratio[(error==0)&(tolerance==0)]=0.
        raise ArithmeticError('reconstructed panel solve failed its original-equation residual check; '
            f'maximum error/tolerance={float(np.max(ratio)):.6g}')
    report.panel_span = dict(rhs_columns=rhs.shape[1],solved_rank=int(keep.sum()),
        original_residual_norm=error.tolist(),original_residual_tolerance=tolerance.tolist())
    return result, report


def conditional_score_memory_plan(training, complete, *, feature_count,
                                  training_fixed_count, confirmation_fixed_count,
                                  mean_tangents=False):
    """Plan a joint panel before covariance estimation or phenotype fitting.

    Counts are stored columns (not fitted ranks). The total covers both
    operators, panel inputs and conservative simultaneous workspaces. Unrelated
    caller-owned arrays must be reserved separately. Per-operator memory limits
    are local product/solver caps, not the aggregate budget for this plan.
    """
    counts = (feature_count, training_fixed_count, confirmation_fixed_count)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            or v < 1 for v in counts) or feature_count > 512):
        raise ValueError('positive fixed-column counts and 1..512 panel features required')
    p, d0, d1 = map(int, counts)
    n0, n = len(training.rows), len(complete.rows)
    n1, k = n-n0, training.count
    if n0 < 2 or n1 < 2 or k != complete.count:
        raise ValueError('compatible training and complete operator dimensions required')
    inputs = 8*(n0*(1+p+d0)+n1*(1+p+d1)+k)
    # Confirmation QR holds the tangent-augmented nuisance matrix, its
    # normalized copies, basis and factorization workspace simultaneously.
    workspace = 8*(10*n0*(p+1)+6*n*p+6*n1*(d1+k*bool(mean_tangents))+6*p*p)
    workspace += 8*n0*(k*(24+2*d0) if mean_tangents else 4*d0)
    if mean_tangents:
        workspace += 24*k*n1
    # Reserve the widest native product even when the operator can afford to
    # process every RHS at once. Smaller operator caps may still force tiling.
    width = max(p+1, k-1 if mean_tangents else 0)
    products = max(8*len(op.rows)*width*(op.count+4+8*op.contexts.shape[1]+8)
                   for op in (training, complete))
    operators = int(training.base_bytes+complete.base_bytes)
    return dict(operator_bytes=operators, input_bytes=inputs,
        workspace_bytes=workspace, product_workspace_bytes=products,
        total_bytes=operators+inputs+workspace+products)


def conditional_score(training, complete, training_index, confirmation_index,
                      y0, y1, f0, f1, c0, c1, theta, *, mean_tangents=False,
                      memory_bytes=None):
    """Joint finite-panel response and full covariance through native products.

    Feature columns belong to one phenotype, unlike the independent scalar
    experiments in conditional_scores_batch. Exact duplicate directions retain
    only their estimable span. Near singular independent directions are
    rejected, rather than silently changing the tested hypothesis. Estimated
    covariance and tangent adjustment still require statistical validation.

    memory_bytes is the aggregate allowance for both operators and this panel,
    after reserving unrelated caller-owned arrays. If omitted, the smaller
    operator cap is retained as the conservative legacy aggregate allowance.
    Explicit aggregate admission never raises either operator's own cap.
    """
    i0, i1 = np.asarray(training_index), np.asarray(confirmation_index)
    n = len(complete.rows)
    if (any(v.ndim != 1 or not np.issubdtype(v.dtype, np.integer)
            or len(v) < 2 or np.any(v < 0) or np.any(v >= n)
            or np.any(np.diff(v) <= 0) for v in (i0, i1))
            or len(i0)+len(i1) != n or len(np.unique(np.r_[i0, i1])) != n
            or not np.array_equal(complete.rows[i0], training.rows)
            or training.stream.source.identity != complete.stream.source.identity
            or not np.array_equal(training.variants, complete.variants)
            or not np.array_equal(training.mean, complete.mean)
            or not np.array_equal(training.inverse, complete.inverse)
            or not np.array_equal(training.contexts, complete.contexts[i0])
            or not np.array_equal(training.noise, complete.noise[i0])):
        raise ValueError("training and complete covariance definitions or sample separation differ")
    y0, y1, f0, f1, c0, c1, theta = (
        np.asarray(v, float) for v in (y0, y1, f0, f1, c0, c1, theta))
    if f0.ndim == 1 and f1.ndim == 1:
        f0, f1 = f0[:, None], f1[:, None]
    if (y0.shape != (len(i0),) or y1.shape != (len(i1),)
            or f0.ndim != 2 or f1.shape != (len(i1), f0.shape[1])
            or len(f0) != len(i0) or not 1 <= f0.shape[1] <= 512
            or c0.ndim != 2 or c1.ndim != 2 or len(c0) != len(i0) or len(c1) != len(i1)
            or not c0.shape[1] or not c1.shape[1]
            or theta.shape != (training.count,) or np.any(theta < 0)
            or not all(np.all(np.isfinite(v)) for v in (y0,y1,f0,f1,c0,c1,theta))):
        raise ValueError("finite aligned phenotype, nuisance, covariance and 1..512 panel features required")
    memory_plan = conditional_score_memory_plan(training, complete,
        feature_count=f0.shape[1], training_fixed_count=c0.shape[1],
        confirmation_fixed_count=c1.shape[1], mean_tangents=mean_tangents)
    planned = memory_plan['total_bytes']
    budget = min(training.memory_bytes, complete.memory_bytes) if memory_bytes is None else memory_bytes
    if (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer))
            or budget <= 0):
        raise ValueError('positive integer aggregate memory_bytes required')
    if planned > budget:
        raise MemoryError(f'joint conditional panel requires {planned} bytes; aggregate budget is {budget}; '
            'reserve caller arrays separately and supply memory_bytes when operator caps are partitioned')
    # Use training and confirmation mean-free coordinates separately.
    # Q1'(Y1 - V10 P0 Y0) is independent of Q0'Y0 for known covariance,
    # where Qj'Cj=0 and P0=Q0 (Q0'V00 Q0)^-1 Q0'. This does not require
    # confirmation nuisance coefficients to be identifiable in training.
    rhs = np.column_stack([y0, f0])
    alpha, first = _panel_projected_solve(training, rhs, c0, theta)
    prediction, _ = complete.cross_products(alpha, i0, i1,
        np.repeat(theta[:, None], rhs.shape[1], axis=1))
    tangent_report = None
    if mean_tangents:
        tangents, tangent_report = conditional_tangents_batch(training, complete,
            i0, i1, alpha[:, :1], c0, theta[:, None])
        # Normalize nonzero nuisance columns before rank determination.
        c1 = np.column_stack([c1, tangents[:, 0]])
    norms = np.linalg.norm(c1, axis=0)
    basis = thin_rank_revealing_fixed_effect_basis(c1[:, norms>0]/norms[norms>0], rtol=1e-11)
    unprojected = f1-prediction[:, 1:]
    response = project(basis, project(basis, unprojected))
    column_norm = np.linalg.norm(response, axis=0)
    absorbed = column_norm <= 1e-11*np.linalg.norm(unprojected,axis=0)
    response[:,absorbed] = 0.
    column_norm[absorbed] = 1.
    left, values, right = np.linalg.svd(response/column_norm, full_matrices=False)
    if not len(values) or values[0] <= 1e-11:
        raise ValueError("absorbed conditional interaction response")
    keep = values > values[0]*1e-11
    retained_values = values[keep]
    condition = float((retained_values[0]/retained_values[-1])**2)
    if condition > 1e8:
        raise ValueError("unidentified conditional interaction response")
    # Use a scale-aware generalized inverse. H beta = D' r and H Cov(beta) H
    # retain the original feature units and weights, even for an overlapping
    # panel. Nonidentifiable individual coefficients remain unavailable.
    response = ((left[:,keep]*retained_values)@right[keep])*column_norm
    contrast = ((left[:, keep]/retained_values)@right[keep])/column_norm
    response[:,absorbed] = 0.
    contrast[:,absorbed] = 0.
    beta = contrast.T@(y1-prediction[:, 0])
    padded = np.zeros((n, f0.shape[1]))
    padded[i1] = contrast
    va = complete.apply(padded, np.repeat(theta[:, None], f0.shape[1], axis=1))
    inverse, second = _panel_projected_solve(training, va[i0], c0, theta)
    covariance = contrast.T@va[i1]-va[i0].T@inverse
    covariance = (covariance+covariance.T)/2
    scaled_covariance = covariance*column_norm[:,None]*column_norm[None,:]
    if np.min(np.linalg.eigvalsh(right[keep]@scaled_covariance@right[keep].T)) <= 0:
        raise ValueError("nonpositive conditional interaction covariance")
    leverage = np.sum(basis**2,axis=1)+np.sum(left[:,keep]**2,axis=1)
    rank = basis.shape[1]+int(keep.sum())
    # Support is measured on retained orthonormal directions, not claimed as
    # the worst case over every linear combination.
    effective = float(np.min(1/np.sum(left[:,keep]**4,axis=0)))
    outside=[]
    for failed, reason in [(len(i1)<1000,'N below 1000'),(rank/len(i1)>.05,'fitted rank exceeds 5% of N'),
            (leverage.max()>.1,'maximum leverage exceeds .1'),(effective<100,'feature effective support below 100'),
            (condition>1e6,'normalized information condition exceeds 1e6')]:
        if failed: outside.append(reason)
    reports=[first.reports,second.reports]
    if tangent_report is not None:
        reports.insert(1,tangent_report.reports)
    return dict(beta=beta, covariance=covariance, contrast=contrast.T, response=response,
        information=response.T@response, prediction=prediction, solver_reports=reports,
        rhs_spans=[first.panel_span,second.panel_span],memory_plan_bytes=planned,
        memory_plan=memory_plan, memory_budget_bytes=int(budget),
        diagnostics=dict(nuisance_training_n=len(i0),confirmation_n=len(i1),
            fixed_rank=basis.shape[1],feature_rank=int(keep.sum()),
            max_leverage=float(leverage.max()),minimum_feature_effective_support=effective,
            information_condition=condition,outside_confirmation_design=outside))


def conditional_tangents_batch(training, complete, i0, i1, alpha, c0, theta, *, checkpoint=None,
                               preconditioners=None):
    """Stream covariance-mean derivatives after the training P Y0 solve.

    No confirmation outcomes enter. Each trait retains its own covariance;
    all components, including diagonal noise, contribute to the derivative.
    The extra solve and its restart authenticate the actual component RHSs.
    """
    n,b,k=len(complete.rows),alpha.shape[1],training.count
    if (theta.shape!=(k,b) or not np.all(np.isfinite(theta)) or np.any(theta<0)
            or np.any(np.max(theta,axis=0)<=0)):
        raise ValueError('positive finite covariance required for mean derivatives')
    # Sum_k theta_k T_k=0 because P V00 P=P. Omit one redundant derivative
    # per trait. The largest coefficient bounds every reconstruction ratio by
    # one, including boundary estimates with other components equal to zero.
    omitted=np.argmax(theta,axis=0)
    retained=np.flatnonzero(np.arange(b*k)%k!=np.repeat(omitted,k))
    component,_=complete.cross_products(alpha,i0,np.arange(n),phase='polygenic_mean_derivative')
    rhs=component[:,i0,:].transpose(1,2,0).reshape(len(i0),b*k)[:,retained]
    repeated=np.repeat(theta,k-1,axis=1)
    derivative,report=projected_solve(training,rhs,c0,repeated,checkpoint=checkpoint,
        resume=checkpoint is not None and checkpoint.exists(),
        preconditioners=None if preconditioners is None else [p for p in preconditioners for _ in range(k-1)])
    # Keep only the needed cross-products before the wide complete-cohort call.
    tangents=component[:,i1,:].transpose(1,2,0).copy()
    del component,rhs
    correction,_=complete.cross_products(derivative,i0,i1,repeated,phase='polygenic_mean_derivative_transfer')
    tangents.reshape(len(i1),b*k)[:,retained]-=correction
    for j in range(b):
        tangents[:,j,omitted[j]]=0.
        tangents[:,j,omitted[j]]=-(tangents[:,j]@(theta[:,j]/theta[omitted[j],j]))
    return tangents,report


def conditional_scores_batch(training, complete, i0, i1, y0, y1, f0, f1,
                             c0, common_c1, extra_c1, theta, *, checkpoint_dir=None,
                             mean_tangents=False, preconditioners=None, recycle=False):
    """Independent scalar tests sharing products and common nuisance projection.

    Each column has its own fitted covariance and extra confirmation main
    effects. Columns are separate experiments, not an independence assumption
    for a joint test. A joint direction test uses ``conditional_score`` above.
    """
    from pathlib import Path
    y0, y1, f0, f1, theta = map(np.asarray, (y0,y1,f0,f1,theta))
    b = y0.shape[1]
    if (y0.shape != f0.shape or y1.shape != f1.shape or y1.shape[1] != b
            or len(extra_c1) != b or theta.shape != (training.count,b)
            or not np.array_equal(complete.rows[i0], training.rows)
            or len(np.unique(np.r_[i0,i1])) != len(complete.rows)
            or not np.array_equal(training.mean, complete.mean)
            or not np.array_equal(training.inverse, complete.inverse)
            or not np.array_equal(training.contexts, complete.contexts[i0])
            or not np.array_equal(training.noise, complete.noise[i0])):
        raise ValueError('incompatible batched conditional inputs')
    _, singular, right = np.linalg.svd(c0, full_matrices=False)
    retained = right[singular>singular[0]*1e-11]
    if (common_c1.shape[1] != c0.shape[1]
            or np.linalg.norm(common_c1-common_c1@retained.T@retained)
            > 1e-8*max(1.,np.linalg.norm(common_c1))):
        raise ValueError('confirmation finite mean exceeds the training fixed span')
    def checkpoint(name):
        return None if checkpoint_dir is None else Path(checkpoint_dir)/(name+'.npz')
    first_path, second_path = checkpoint('outcome_feature'), checkpoint('covariance_contrast')
    rhs = np.stack([y0,f0],axis=2).reshape(len(y0),2*b)
    repeated = np.repeat(theta,2,axis=1)
    spaces=None;recycled_path=checkpoint('recycled_inverse')
    if recycle:
        from .krylov import KrylovSpace, load_recycled_inverses, write_recycled_inverses
        from summit.prediction.artifacts import file_digest
        if preconditioners is not None:
            raise ValueError('choose one computational inverse')
        recycle_identity=digest([training.identity,array_digest(rhs),array_digest(c0),array_digest(theta),
            file_digest(Path(__file__).with_name('krylov.py')),'conditional_recycling_v1'])
        if recycled_path is not None and recycled_path.exists():
            preconditioners=load_recycled_inverses(recycled_path,identity=recycle_identity,
                operator=training,theta=theta,memory_bytes=training.memory_bytes)
        else:
            if any(p is not None and p.exists() for p in (second_path,checkpoint('mean_derivative'))):
                raise ValueError('dependent solver checkpoint is missing its recycled inverse')
            # At most 64 exact product pairs per trait. A restart before their
            # publication may collect fewer pairs; this changes speed only.
            basis=thin_rank_revealing_fixed_effect_basis(c0,rtol=1e-11)
            spaces=[KrylovSpace(training,theta[:,j],basis,capacity=64) for j in range(b)]
            if training.base_bytes+3*sum(s.reserved_bytes for s in spaces)+256*2**20>training.memory_bytes:
                raise MemoryError('recycled covariance batch exceeds preparation memory')
    alpha, first = projected_solve(training,rhs,c0,repeated,checkpoint=first_path,
        resume=first_path is not None and first_path.exists(),
        # First solve always uses the original inverse. Recycling is for later
        # RHSs, including when a completed first checkpoint is being resumed.
        preconditioners=None if recycle or preconditioners is None else [p for p in preconditioners for _ in range(2)],
        recycle_spaces=None if spaces is None else [s for s in spaces for _ in range(2)])
    if spaces is not None:
        preconditioners=[s.freeze(training.diagonal.T@theta[:,j]) for j,s in enumerate(spaces)]
        if recycled_path is not None:
            write_recycled_inverses(recycled_path,preconditioners,identity=recycle_identity)
        del spaces
    prediction,_=complete.cross_products(alpha,i0,i1,repeated)
    prediction=prediction.reshape(len(y1),b,2)
    tangent_report=None
    if mean_tangents:
        tangents,tangent_report=conditional_tangents_batch(training,complete,i0,i1,
            alpha[:,::2],c0,theta,checkpoint=checkpoint('mean_derivative'),preconditioners=preconditioners)
        extra_c1=[np.column_stack([extra_c1[j],tangents[:,j]]) for j in range(b)]
    base = thin_rank_revealing_fixed_effect_basis(common_c1,rtol=1e-11)
    base_leverage = np.sum(base**2,axis=1)
    response = project(base,f1-prediction[:,:,1])
    contrasts = np.empty_like(response)
    diagnostics = []
    for j in range(b):
        extra = np.asarray(extra_c1[j])
        norms = np.linalg.norm(extra,axis=0)
        extra = extra[:,norms>0]/norms[norms>0]
        extra = project(base,project(base,extra))
        extra = extra[:,np.linalg.norm(extra,axis=0)>1e-11]
        basis = thin_rank_revealing_fixed_effect_basis(extra,rtol=1e-11)
        r = project(basis,response[:,j])
        r = project(base,project(basis,r))
        energy = r@r
        if energy <= 1e-20*max(float(f1[:,j]@f1[:,j]),np.finfo(float).tiny):
            raise ValueError('absorbed conditional interaction response')
        response[:,j] = r
        contrasts[:,j] = r/energy
        leverage = base_leverage+np.sum(basis**2,axis=1)+r*r/energy
        rank = base.shape[1]+basis.shape[1]+1
        effective = float(energy**2/np.sum(r**4))
        outside=[]
        for failed,reason in [(len(r)<1000,'N below 1000'),(rank/len(r)>.05,'fitted rank exceeds 5% of N'),
            (leverage.max()>.1,'maximum leverage exceeds .1'),(effective<100,'feature effective support below 100')]:
            if failed: outside.append(reason)
        diagnostics.append(dict(rank=rank,max_leverage=float(leverage.max()),feature_effective_support=effective,
            outside_confirmation_design=outside))
    beta = np.sum(contrasts*(y1-prediction[:,:,0]),axis=0)
    rhs,quadratic=complete.cross_products(contrasts,i1,i0,theta)
    inverse, second = projected_solve(training,rhs,c0,theta,checkpoint=second_path,
        resume=second_path is not None and second_path.exists(),preconditioners=preconditioners)
    variance = quadratic-np.sum(rhs*inverse,axis=0)
    if np.any(variance<=0):
        raise ValueError('nonpositive conditional interaction variance')
    reports=[first.reports,second.reports]
    if tangent_report is not None:
        reports.insert(1,tangent_report.reports)
    return dict(beta=beta,variance=variance,contrasts=contrasts,training_contrasts=-inverse,
        response=response,prediction=prediction[:,:,0],diagnostics=diagnostics,
        solver_reports=reports,recycled_ranks=None if not recycle else [p.d.shape[1] for p in preconditioners])
