"""One decoded traversal for ordered cross-trait scores and residual moments.

Score reduction is one batched matrix product per target-block fragment; it
has no Python loop over SNPs or pairs. Exact residual moments require the
intersection of each pair's masks and its two covariate projectors. Their
cost is recorded separately and must not be hidden in the score benchmark.
"""
from __future__ import annotations

from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import count
import os
from threadpoolctl import threadpool_limits
import numpy as np

from summit.context.cross_trait_zpass import write_array_artifact


def _pooled_missing_groups(bits, flags, *, budget, minimum_rows, addition_penalty):
    """Plan disjoint cached row groups, using masks only (no genotype data).

    A group can serve a pair only when every person in it misses both traits.
    Split distinct missingness patterns on balanced trait bits, then select a
    disjoint tree cut with at most ``budget`` groups. The cost model counts
    avoided row products and charges a configurable row-equivalent cost for
    adding a cached product. It changes work allocation, never the moments.
    """
    if not len(flags):
        return np.array([], dtype=np.uint64), []
    patterns, inverse, counts = np.unique(bits, return_inverse=True, return_counts=True)
    bitflags = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))

    def build(indices):
        common = np.bitwise_and.reduce(patterns[indices])
        support = int(np.count_nonzero((common & flags) == flags))
        mass = int(counts[indices].sum())
        gain = mass * (support - 1) - addition_penalty * support
        eligible = mass >= minimum_rows and support > 1 and gain > 0
        values = np.array([0., gain]) if eligible else np.array([0.])
        choice = np.array([-1, -2]) if eligible else np.array([-1])
        node = dict(indices=indices, common=common)
        if len(indices) > 1:
            frequency = counts[indices] @ ((patterns[indices, None] & bitflags) != 0)
            bit = bitflags[np.argmax(np.minimum(frequency, mass-frequency))]
            take = (patterns[indices] & bit) != 0
            left, right = build(indices[take]), build(indices[~take])
            a, b = left['values'], right['values']
            size = max(len(values), min(budget, len(a)+len(b)-2)+1)
            best = np.full(size, -np.inf)
            split = np.full(size, -1, dtype=int)
            for i in range(min(len(a), size)):
                candidate = a[i]+b[:size-i]
                better = candidate > best[i:i+len(candidate)]
                best[i:i+len(candidate)][better] = candidate[better]
                split[i:i+len(candidate)][better] = i
            if eligible and gain > best[1]:
                best[1], split[1] = gain, -2
            values, choice = best, split
            node['children'] = left, right
        node.update(values=values, choice=choice)
        return node

    root = build(np.arange(len(patterns)))
    selected = []

    def recover(node, count):
        if count == 0:
            return
        split = int(node['choice'][count])
        if split == -2:
            selected.append(node)
        else:
            left, right = node['children']
            recover(left, split)
            recover(right, count-split)

    recover(root, int(np.argmax(root['values'])))
    labels = np.full(len(patterns), -1, dtype=int)
    for i, node in enumerate(selected):
        labels[node['indices']] = i
    rows = [np.flatnonzero(labels[inverse] == i) for i in range(len(selected))]
    return np.array([node['common'] for node in selected], dtype=np.uint64), rows


class CrossTraitScoreAccumulator:
    """Annotation-major ordered Q x Q score sums, on paired target blocks."""

    def __init__(self, *, trait_names, num_basis, num_annotations, block_ids, pairs=None):
        self.names=tuple(trait_names);self.q=int(num_basis);self.k=int(num_annotations)
        self.block_ids=np.asarray(block_ids,dtype=np.int64)
        t=len(self.names)
        if (not t or len(set(self.names))!=t or min(self.q,self.k)<1
                or self.block_ids.ndim!=1 or not len(self.block_ids)
                or np.any(np.diff(self.block_ids)<=0)):
            raise ValueError('invalid trait/basis/annotation/block design')
        self.pairs=np.asarray([(x,y) for x in range(t) for y in range(x+1,t)]
                              if pairs is None else pairs,dtype=np.int64).reshape(-1,2)
        if (not len(self.pairs) or np.any(self.pairs<0) or np.any(self.pairs>=t)
                or len(np.unique(self.pairs,axis=0))!=len(self.pairs)):
            raise ValueError('pair indices must be nonempty, unique and in range')
        self.rhs=np.zeros((len(self.block_ids),len(self.pairs),self.k,self.q,self.q))
        self.masses=np.zeros((len(self.block_ids),self.k))
        self.seconds=0.;self.variants=0

    def add(self,scores,annotations,groups):
        start=perf_counter()
        s=np.asarray(scores,dtype=float);a=np.asarray(annotations,dtype=float)
        groups=np.asarray(groups)
        if (s.shape!=(len(groups),len(self.names),self.q) or a.shape!=(len(groups),self.k)
                or not np.isfinite(s).all() or not np.isfinite(a).all() or np.any(a<0)):
            raise ValueError('invalid score/annotation tile')
        flat=s.reshape(len(s),-1)
        labels=np.unique(groups)
        if not np.isin(labels,self.block_ids).all():
            raise ValueError('unknown target block')
        for label in labels:
            keep=groups==label;f=flat[keep];weights=a[keep]
            gram=f.T[None] @ (weights.T[:,:,None]*f[None])
            gram=gram.reshape(self.k,len(self.names),self.q,len(self.names),self.q)
            gram=gram.transpose(1,3,0,2,4)
            b=np.searchsorted(self.block_ids,label)
            self.rhs[b]+=gram[self.pairs[:,0],self.pairs[:,1]]
            self.masses[b]+=weights.sum(axis=0)
        self.variants+=len(s);self.seconds+=perf_counter()-start


class CrossTraitBatch:
    """Wrap MaskedTraitBatch, retaining all scores in the same traversal.

    The master residual span may be redundant on an overlap. The solver must
    rank-reveal its projected cross-residual Gram, not invert the raw span.
    Both zero overlap and partially overlapping fixed bases are supported.
    """

    def __init__(self, masked, *, block_ids, annotation_names, pairs=None,max_cached_missing_patterns=64,
                 minimum_cached_pattern_rows=64,missing_cache_strategy='patterns',
                 cache_addition_penalty_rows=8.,parallel_cache_products=False,
                 residual_workers=1,residual_cpus=None):
        self.masked=masked;self.annotation_names=tuple(annotation_names)
        if int(minimum_cached_pattern_rows)!=minimum_cached_pattern_rows or minimum_cached_pattern_rows<1:
            raise ValueError('cached missing patterns require a positive minimum row count')
        if missing_cache_strategy not in ('patterns','pooled'):
            raise ValueError('unknown missing-row cache strategy')
        if not np.isfinite(cache_addition_penalty_rows) or cache_addition_penalty_rows<0:
            raise ValueError('cache addition penalty must be finite and nonnegative')
        if int(max_cached_missing_patterns)!=max_cached_missing_patterns or max_cached_missing_patterns<0:
            raise ValueError('cache group budget must be a nonnegative integer')
        self.residual_workers=int(residual_workers)
        self.residual_cpus=tuple(sorted(os.sched_getaffinity(0) if residual_cpus is None else residual_cpus))
        if (self.residual_workers<1 or self.residual_workers>len(self.residual_cpus)
                or len(set(self.residual_cpus))!=len(self.residual_cpus)):
            raise ValueError('residual workers require distinct reserved CPU IDs')
        self._executor=None;self.residual_worker_affinity={}
        self.parallel_cache_products=bool(parallel_cache_products)
        self.scores=CrossTraitScoreAccumulator(trait_names=[t['name'] for t in masked.traits],
            num_basis=masked.q,num_annotations=len(annotation_names),block_ids=block_ids,pairs=pairs)
        self.geometry=[];self.residual_seconds=0.
        self.residual_phase_seconds={name:0. for name in ('patterns','raw_products','span_expansion','projection','reduction')}
        h=masked.h;q=masked.q
        self.genetic_residual=np.zeros((*self.scores.rhs.shape[:-2],q*q,h))
        self.residual_gram=np.zeros((len(self.scores.pairs),h,h))
        self.residual_rhs=np.zeros((len(self.scores.pairs),h))
        for pair_index,(ix,iy) in enumerate(self.scores.pairs):
            x,y=masked.traits[ix],masked.traits[iy]
            overlap,lx,ly=np.intersect1d(x['indices'],y['indices'],return_indices=True)
            d=masked.residual_basis[overlap]
            # Inclusion/exclusion of missing-person products reuses the
            # already computed per-trait fixed moments. Nested masks need no
            # additional large covariate product at all.
            nested=ix if len(overlap)==len(x['indices']) else (iy if len(overlap)==len(y['indices']) else None)
            joint_missing=np.setdiff1d(np.arange(masked.n),np.union1d(x['indices'],y['indices']),assume_unique=True)
            if nested is not None:
                joint=masked.traits[nested]['master_residual']
            else:
                u0=masked.fixed_basis[joint_missing];d0=masked.residual_basis[joint_missing]
                joint=(x['master_residual']+y['master_residual']-masked.master_residual
                    +np.stack([u0.T@(d0[:,j,None]*u0) for j in range(h)]))
            cross=x['transform'].T@joint@y['transform']
            # These are the actual supplied trait projectors' row norms,
            # already computed once by MaskedTraitBatch. Reconstructing two
            # overlap-by-fixed-rank matrices per pair is unnecessary.
            leverage=1-x['fixed_leverage'][lx]-y['fixed_leverage'][ly]
            self.residual_gram[pair_index]=d.T@(d*leverage[:,None])+np.einsum('hcd,kcd->hk',cross,cross)
            self.residual_rhs[pair_index]=d.T@(x['common'].normalized_phenotypes[lx,0]
                                                         *y['common'].normalized_phenotypes[ly,0])
            use_missing=len(joint_missing)<len(overlap)
            # Keep only rows needed by the tile correction. The full overlap
            # is used above for fixed residual moments and otherwise costs
            # O(number_of_pairs * cohort_size) unused persistent storage.
            self.geometry.append(dict(correction=joint_missing if use_missing else overlap,
                inclusion_exclusion=use_missing,nested=nested,
                cross_gemm=np.ascontiguousarray(cross.transpose(1,0,2)).reshape(x['fixed_rank'],h*y['fixed_rank'])))
        self._ordered_lookup=np.empty((q,q),dtype=int)
        for i,(a,b) in enumerate(masked.pairs):
            self._ordered_lookup[a,b]=self._ordered_lookup[b,a]=i
        # People with the same missing-trait bit pattern contribute identical
        # raw overlap corrections to many pairs. Cache a bounded number of
        # disjoint groups per tile; each genotype is decoded only once.
        # Small/private patterns remain in the exact per-pair correction.
        self.missing_pattern_rows=[]
        if len(masked.traits)<=64 and max_cached_missing_patterns>0:
            bits=np.zeros(masked.n,dtype=np.uint64)
            for t,trait in enumerate(masked.traits):
                absent=np.ones(masked.n,dtype=bool);absent[trait['indices']]=False
                bits[absent]|=np.uint64(1)<<np.uint64(t)
            if missing_cache_strategy=='pooled':
                flags=np.array([(np.uint64(1)<<np.uint64(ix))|(np.uint64(1)<<np.uint64(iy))
                    for geom,(ix,iy) in zip(self.geometry,self.scores.pairs)
                    if geom['nested'] is None and geom['inclusion_exclusion']],dtype=np.uint64)
                selected,self.missing_pattern_rows=_pooled_missing_groups(bits,flags,
                    budget=int(max_cached_missing_patterns),minimum_rows=minimum_cached_pattern_rows,
                    addition_penalty=cache_addition_penalty_rows)
            else:
                patterns,counts=np.unique(bits,return_counts=True)
                support=np.zeros(len(patterns),dtype=np.int64)
                for ix,iy in self.scores.pairs:
                    flag=(np.uint64(1)<<np.uint64(ix))|(np.uint64(1)<<np.uint64(iy))
                    support+=(patterns&flag)==flag
                eligible=np.flatnonzero((counts>=minimum_cached_pattern_rows)&(support>1))
                eligible=eligible[np.argsort((counts*support)[eligible])[::-1][:max_cached_missing_patterns]]
                selected=patterns[eligible]
                self.missing_pattern_rows=[np.flatnonzero(bits==pattern) for pattern in selected]
            for geom,(ix,iy) in zip(self.geometry,self.scores.pairs):
                geom['cached_patterns']=np.array([],dtype=int)
                if geom['nested'] is not None or not geom['inclusion_exclusion']:continue
                flag=(np.uint64(1)<<np.uint64(ix))|(np.uint64(1)<<np.uint64(iy))
                groups=np.flatnonzero((selected&flag)==flag);geom['cached_patterns']=groups
                if len(groups):
                    cached=np.sort(np.concatenate([self.missing_pattern_rows[j] for j in groups]))
                    geom['correction']=np.setdiff1d(geom['correction'],cached,assume_unique=True)

    def block(self,genotype,annotations,groups,*,z_callback=None,shared_tn_operator=None):
        """Yield unchanged within-trait statistics; accumulate cross statistics."""
        x=np.asarray(genotype,dtype=float);a=np.asarray(annotations,dtype=float);g=np.asarray(groups)
        projections={};shared=[];raw_traits={}
        def collect(index,value):
            projections[index]=value
        yield from self.masked.block(x,score_callback=lambda s:self.scores.add(s,a,g),
            z_callback=z_callback,projection_callback=collect,
            shared_tn_operator=shared_tn_operator,
            shared_callback=lambda linear,square:shared.extend((linear,square)),
            raw_callback=lambda i,linear,square:raw_traits.update({i:(linear,square)}))
        start=perf_counter();m=self.masked;q=m.q;h=m.h;p=len(m.pairs)
        # Only complements of overlaps are multiplied. Reuse shared products
        # across every pair; no decoded genotype is revisited.
        common_linear,common_square=shared
        squared=x*x
        def pattern_product(rows):
            return x[:,rows]@m.fixed_weights[rows],squared[:,rows]@m.square_weights[rows]
        pattern_products=(self._map_residual_work(pattern_product,self.missing_pattern_rows)
                          if self.parallel_cache_products else
                          list(map(pattern_product,self.missing_pattern_rows)))
        self.residual_phase_seconds['patterns']+=perf_counter()-start
        # Contract the small trait transform before applying all residual
        # multipliers. This is algebraically identical and avoids Q*H copies
        # of the C by C transform for every pair and SNP.
        master_projection={i:z@m.traits[i]['transform'].T for i,z in projections.items()}
        residual_coefficients=np.ascontiguousarray(m.fixed_coefficients[:,q:])
        def pair_work(pair_index):
            ix,iy=self.scores.pairs[pair_index]
            phases={name:0. for name in self.residual_phase_seconds}
            tick=perf_counter()
            geom=self.geometry[pair_index];rows=geom['correction']
            if geom['nested'] is not None:
                linear,square=raw_traits[geom['nested']]
            elif len(rows):
                linear=x[:,rows]@m.fixed_weights[rows]
                square=squared[:,rows]@m.square_weights[rows]
            else:
                linear=np.zeros_like(common_linear);square=np.zeros_like(common_square)
            for pattern in geom.get('cached_patterns',()):
                linear+=pattern_products[pattern][0];square+=pattern_products[pattern][1]
            if geom['nested'] is None and geom['inclusion_exclusion']:
                linear=raw_traits[ix][0]+raw_traits[iy][0]-common_linear+linear
                square=raw_traits[ix][1]+raw_traits[iy][1]-common_square+square
            phases['raw_products']+=perf_counter()-tick;tick=perf_counter()
            linear=linear.reshape(len(x),m.multiplier_rank,m.fixed_rank)
            raw=(square@m.square_coefficients).reshape(len(x),h+1,p)
            phases['span_expansion']+=perf_counter()-tick
            tick=perf_counter()
            zx,zy=projections[ix],projections[iy]
            value=raw[:,1:,self._ordered_lookup].transpose(0,2,3,1).copy()
            # Contract C while the multiplier axis is still rank-compressed,
            # then expand only width*rank*Q to width*H*Q². This avoids the
            # width*H*Q*C intermediate without changing either contraction.
            left=linear@master_projection[iy].transpose(0,2,1)
            left=left.transpose(0,2,1).reshape(len(x)*q,m.multiplier_rank)@residual_coefficients
            value-=left.reshape(len(x),q,h,q).transpose(0,3,1,2)
            right=linear@master_projection[ix].transpose(0,2,1)
            right=right.transpose(0,2,1).reshape(len(x)*q,m.multiplier_rank)@residual_coefficients
            value-=right.reshape(len(x),q,h,q).transpose(0,1,3,2)
            # An unconstrained einsum path makes a width*Q²*C² outer product.
            # Explicit GEMMs contract C first, using width*Q*H*C workspace.
            cx,cy=zx.shape[-1],zy.shape[-1]
            if cx and cy:
                projected=(zx.reshape(len(x)*q,cx)@geom['cross_gemm']).reshape(len(x),q*h,cy)
                value+=(projected@zy.transpose(0,2,1)).reshape(len(x),q,h,q).transpose(0,1,3,2)
            value=value.reshape(len(x),q*q,h)
            phases['projection']+=perf_counter()-tick;tick=perf_counter()
            for label in np.unique(g):
                take=g==label;b=np.searchsorted(self.scores.block_ids,label)
                self.genetic_residual[b,pair_index]+=np.einsum('jk,jph->kph',a[take],value[take],optimize=True)
            phases['reduction']+=perf_counter()-tick
            return phases
        for phases in self._map_residual_work(pair_work,range(len(self.geometry))):
            for name,value in phases.items():self.residual_phase_seconds[name]+=value
        self.residual_seconds+=perf_counter()-start

    def _map_residual_work(self,function,items):
        """Finish every independent task before restoring the global BLAS limit."""
        if self.residual_workers>1 and self._executor is None:
            next_cpu=count()
            def initialize_worker():
                cpu=self.residual_cpus[next(next_cpu)]
                os.sched_setaffinity(0,{cpu})
                if os.sched_getaffinity(0)!={cpu}:raise RuntimeError('residual worker affinity differs')
                self.residual_worker_affinity[cpu]=[cpu]
            self._executor=ThreadPoolExecutor(max_workers=self.residual_workers,
                initializer=initialize_worker,thread_name_prefix='summit-cross-residual')
        # Only this independent residual section runs concurrently. NumPy's
        # process-wide BLAS limit is restored before another shared/native call.
        limit=threadpool_limits(limits=1,user_api='blas') if self._executor else nullcontext()
        with limit:
            try:
                return list(self._executor.map(function,items) if self._executor else map(function,items))
            except BaseException:
                # Join workers before restoring a process-wide BLAS limit.
                self.close()
                raise

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True,cancel_futures=True);self._executor=None

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def write(self,path,*,provenance):
        write_array_artifact(path,kind='summit.cross_trait.summary',arrays=dict(
            block_ids=self.scores.block_ids,block_masses=self.scores.masses,pairs=self.scores.pairs,
            trait_names=np.array(self.scores.names),annotation_names=np.array(self.annotation_names),
            block_rhs=self.scores.rhs,block_genetic_residual=self.genetic_residual,
            residual_gram=self.residual_gram,residual_rhs=self.residual_rhs),
            provenance=dict(provenance,variants=self.scores.variants,
                score_accumulation_seconds=self.scores.seconds,residual_seconds=self.residual_seconds,
                mask_projector_policy='exact_own_projectors_overlap_residuals'))
