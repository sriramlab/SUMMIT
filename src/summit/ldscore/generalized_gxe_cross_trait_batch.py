"""One decoded traversal for ordered cross-trait scores and residual moments.

Score reduction is one batched matrix product per target-block fragment; it
has no Python loop over SNPs or pairs. Exact residual moments require the
intersection of each pair's masks and its two covariate projectors. Their
cost is recorded separately and must not be hidden in the score benchmark.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from summit.context.cross_trait_zpass import write_array_artifact


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

    def __init__(self, masked, *, block_ids, annotation_names, pairs=None):
        self.masked=masked;self.annotation_names=tuple(annotation_names)
        self.scores=CrossTraitScoreAccumulator(trait_names=[t['name'] for t in masked.traits],
            num_basis=masked.q,num_annotations=len(annotation_names),block_ids=block_ids,pairs=pairs)
        self.geometry=[];self.residual_seconds=0.
        h=masked.h;q=masked.q
        self.genetic_residual=np.zeros((*self.scores.rhs.shape[:-2],q*q,h))
        self.residual_gram=np.zeros((len(self.scores.pairs),h,h))
        self.residual_rhs=np.zeros((len(self.scores.pairs),h))
        for pair_index,(ix,iy) in enumerate(self.scores.pairs):
            x,y=masked.traits[ix],masked.traits[iy]
            overlap,lx,ly=np.intersect1d(x['indices'],y['indices'],return_indices=True)
            ux=masked.fixed_basis[overlap]@x['transform']
            uy=masked.fixed_basis[overlap]@y['transform']
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
            leverage=1-np.sum(ux*ux,axis=1)-np.sum(uy*uy,axis=1)
            self.residual_gram[pair_index]=d.T@(d*leverage[:,None])+np.einsum('hcd,kcd->hk',cross,cross)
            self.residual_rhs[pair_index]=d.T@(x['common'].normalized_phenotypes[lx,0]
                                                         *y['common'].normalized_phenotypes[ly,0])
            use_missing=len(joint_missing)<len(overlap)
            self.geometry.append(dict(overlap=overlap,correction=joint_missing if use_missing else overlap,
                inclusion_exclusion=use_missing,nested=nested,
                cross_gemm=np.ascontiguousarray(cross.transpose(1,0,2)).reshape(masked.fixed_rank,-1)))
        self._ordered_lookup=np.empty((q,q),dtype=int)
        for i,(a,b) in enumerate(masked.pairs):
            self._ordered_lookup[a,b]=self._ordered_lookup[b,a]=i

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
        # Contract the small trait transform before applying all residual
        # multipliers. This is algebraically identical and avoids Q*H copies
        # of the C by C transform for every pair and SNP.
        master_projection={i:z@m.traits[i]['transform'].T for i,z in projections.items()}
        cache={}
        for pair_index,(ix,iy) in enumerate(self.scores.pairs):
            geom=self.geometry[pair_index];rows=geom['correction']
            key=(int(ix),int(iy))
            # Identical masks share their raw moments in this tile. Retain
            # only the most recent mask to bound storage independently of T².
            if key not in cache:
                if geom['nested'] is not None:
                    linear,square=raw_traits[geom['nested']]
                elif len(rows):
                    linear=x[:,rows]@m.fixed_weights[rows]
                    square=squared[:,rows]@m.square_weights[rows]
                else:
                    linear=np.zeros_like(common_linear);square=np.zeros_like(common_square)
                if geom['nested'] is None and geom['inclusion_exclusion']:
                    linear=raw_traits[ix][0]+raw_traits[iy][0]-common_linear+linear
                    square=raw_traits[ix][1]+raw_traits[iy][1]-common_square+square
                linear=np.einsum('jrc,rs->jsc',linear.reshape(len(x),m.multiplier_rank,m.fixed_rank),
                                 m.fixed_coefficients,optimize=True).reshape(len(x),h+1,q,m.fixed_rank)
                raw=(square@m.square_coefficients).reshape(len(x),h+1,p)
                cache={key:(linear,raw)}
            linear,raw=cache[key]
            zx,zy=projections[ix],projections[iy]
            value=raw[:,1:,self._ordered_lookup].transpose(0,2,3,1).copy()
            value-=np.einsum('jhqc,jrc->jqrh',linear[:,1:],master_projection[iy],optimize=True)
            value-=np.einsum('jqc,jhrc->jqrh',master_projection[ix],linear[:,1:],optimize=True)
            # An unconstrained einsum path makes a width*Q²*C² outer product.
            # Explicit GEMMs contract C first, using width*Q*H*C workspace.
            projected=(zx.reshape(-1,m.fixed_rank)@geom['cross_gemm']).reshape(len(x),q*h,m.fixed_rank)
            value+=(projected@zy.transpose(0,2,1)).reshape(len(x),q,h,q).transpose(0,1,3,2)
            value=value.reshape(len(x),q*q,h)
            for label in np.unique(g):
                take=g==label;b=np.searchsorted(self.scores.block_ids,label)
                self.genetic_residual[b,pair_index]+=np.einsum('jk,jph->kph',a[take],value[take],optimize=True)
        self.residual_seconds+=perf_counter()-start

    def write(self,path,*,provenance):
        write_array_artifact(path,kind='summit.cross_trait.summary',arrays=dict(
            block_ids=self.scores.block_ids,block_masses=self.scores.masses,pairs=self.scores.pairs,
            trait_names=np.array(self.scores.names),annotation_names=np.array(self.annotation_names),
            block_rhs=self.scores.rhs,block_genetic_residual=self.genetic_residual,
            residual_gram=self.residual_gram,residual_rhs=self.residual_rhs),
            provenance=dict(provenance,variants=self.scores.variants,
                score_accumulation_seconds=self.scores.seconds,residual_seconds=self.residual_seconds,
                mask_projector_policy='exact_own_projectors_overlap_residuals'))
