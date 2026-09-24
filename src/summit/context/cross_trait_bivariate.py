"""Run ordinary, unweighted SUMMIT from single-annotation block moments.

Replacing every SNP's LD/score moment by its block mean preserves all sums
used by the existing HE fitter and its fixed target-block deletions. This
adapter expands those means to the original block cardinalities: annotation
squares, masses and jackknife sizes remain correct. The expanded positions
are bookkeeping indices, not reconstructed SNP-level observations. They
must not be used for IRWLS, SNP filtering or per-SNP diagnostics.
"""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.h2core import prepare_h2, fit_h2
from summit.inference.rgcore import prepare_rg, fit_intercept, fit_rg
from summit.ldscore.generalized_gxe_chromosome import joint_chromosome_equations


def baseline_reference_block_ld(chromosomes):
    """Profile the constant residual through the existing chromosome assembler.

    Inputs must already select one annotation. The returned LD numerators
    retain its inherited within-chromosome, frozen-source approximation and
    any reference probe noise. No study or cross-trait fitted Gram is used.
    """
    chunks=tuple(chromosomes)
    if not chunks or any(len(c.annotation_names)!=1 for c in chunks):
        raise ValueError('one selected annotation is required')
    baseline=tuple(replace(c,num_basis=1,block_directed=c.block_directed[:,:1,:1],
        block_genetic_rhs=c.block_genetic_rhs[:,:1],
        block_genetic_residual=c.block_genetic_residual[:,:1,:1]) for c in chunks)
    ids=np.unique(np.concatenate([c.block_ids for c in baseline]))
    masses=np.zeros(len(ids))
    for c in baseline: masses[np.searchsorted(ids,c.block_ids)]+=c.block_masses[:,0]
    rank=baseline[0].residual_rank
    def numerator(deleted=()):
        e=joint_chromosome_equations(baseline,residual_gram=np.array([[rank]]),
            residual_rhs=np.zeros((1,len(baseline[0].trait_names))),
            residual_traces=np.array([rank]),residual_names=('constant',),deleted_blocks=deleted)
        profiled=e.matrix[0,0]-e.matrix[0,1]**2/e.matrix[1,1]
        return profiled*e.annotation_masses[0]**2/rank**2
    full=numerator()
    ld=np.array([full-numerator((int(b),)) for b in ids])
    np.testing.assert_allclose(ld.sum(),full,rtol=1e-10,atol=1e-10)
    return ids,masses,ld


class BivariateBlockAdapter:
    """Ordinary SUMMIT HE with exact supplied block sums and fixed overlap c."""
    def __init__(self,block_ids,counts,ld_sums):
        ids=np.asarray(block_ids);counts=np.asarray(counts);ld=np.asarray(ld_sums,dtype=float)
        if (ids.ndim!=1 or len(ids)<2 or np.any(np.diff(ids)<=0)
                or counts.shape!=ids.shape or ld.shape!=ids.shape
                or not np.isfinite(counts).all() or not np.isfinite(ld).all()
                or np.any(counts<=0) or np.any(counts!=np.floor(counts))):
            raise ValueError('sorted blocks, positive integer SNP counts and finite LD sums required')
        self.ids=ids;self.counts=counts.astype(np.int64)
        ends=np.cumsum(self.counts);starts=np.r_[0,ends[:-1]]
        m=int(ends[-1]);b=len(ids);axis=np.arange(m)
        self.trace=SimpleNamespace(nsnps=m,nbins=1,snps=axis,annot=np.ones((m,1)),
            ldscores=np.repeat(ld/self.counts,self.counts)[:,None])
        self.jackknife=JackknifeDesign(spec=JackknifeSpec.parse(b),nrep=b,nunit=b,
            unit_id=np.repeat(np.arange(b),self.counts),unit_labels=ids,
            delete_sets=np.arange(b)[:,None],D=np.eye(b),starts=starts,ends=ends,nsnps=m)

    def summary(self,sums):
        sums=np.asarray(sums,dtype=float)
        if sums.shape!=self.counts.shape or not np.isfinite(sums).all():
            raise ValueError('finite score sums on the block axis are required')
        return np.repeat(sums/self.counts,self.counts)

    def matched(self,n,rank):
        if not 0<rank<=n: raise ValueError('invalid sample/residual rank')
        return SimpleNamespace(nsnps=self.trace.nsnps,snps=self.trace.snps,nsamp=n,n_scale=rank)

    def within(self,score_sums,*,n,rank):
        matched=self.matched(n,rank)
        prepared=prepare_h2(self.trace,matched,self.jackknife,
            summary_y=self.summary(np.asarray(score_sums)/rank))
        return fit_h2(prepared,report_tau=False),matched

    def cross(self,score_sums,*,left,right,overlap_rhs):
        h1,m1=left;h2,m2=right;scale=np.sqrt(m1.n_scale*m2.n_scale)
        y=self.summary(np.asarray(score_sums)/scale)
        prepared=prepare_rg(self.trace,m1,m2,self.jackknife,summary_y=y)
        intercept=fit_intercept(self.trace,m1,m2,self.jackknife,h1,h2,summary_y=y,
            fixed_c=float(overlap_rhs)/scale)
        return fit_rg(prepared,h1,h2,intercept)
