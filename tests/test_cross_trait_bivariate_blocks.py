"""Block means preserve the independent ordinary SUMMIT fit and every deletion."""
from types import SimpleNamespace
import numpy as np
from summit.context.cross_trait_bivariate import BivariateBlockAdapter,baseline_reference_block_ld
from summit.ldscore.generalized_gxe_chromosome import ChromosomeMoments
from summit.inference.h2core import prepare_h2,fit_h2
from summit.inference.rgcore import prepare_rg,fit_intercept,fit_rg


def test_block_adapter_preserves_snp_axis_ordinary_fits():
    rng=np.random.default_rng(230924);n,m=311,210
    counts=np.array([21,33,29,37,41,49]);ids=np.array([1,3,4,6,8,9])
    g=rng.normal(size=(n,m));g-=g.mean(0);g*=np.sqrt(n/np.sum(g*g,axis=0))
    beta=rng.multivariate_normal([0,0],[[.6,.3],[.3,.5]],size=m)/np.sqrt(m)
    y=g@beta+rng.normal(size=(n,2))*.5;y*=np.sqrt(n/np.sum(y*y,axis=0))
    scores=g.T@y;ld=np.sum((g.T@g)**2,axis=1)/n**2-m/n
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    reduce=lambda x:np.add.reduceat(x,starts,axis=0)
    adapter=BivariateBlockAdapter(ids,counts,reduce(ld))
    trace=SimpleNamespace(**dict(vars(adapter.trace),ldscores=ld[:,None]))
    matched=adapter.matched(n,n);jk=adapter.jackknife
    ordinary=[];compact=[]
    for t in range(2):
        h=fit_h2(prepare_h2(trace,matched,jk,summary_y=scores[:,t]**2/n),report_tau=False)
        other,meta=adapter.within(reduce(scores[:,t]**2),n=n,rank=n)
        np.testing.assert_allclose(other.sigma_reps,h.sigma_reps,rtol=1e-12,atol=1e-12)
        ordinary.append(h);compact.append((other,meta))
    product=scores[:,0]*scores[:,1]/n;c=y[:,0]@y[:,1]/n
    prepared=prepare_rg(trace,matched,matched,jk,summary_y=product)
    intercept=fit_intercept(trace,matched,matched,jk,*ordinary,summary_y=product,fixed_c=c)
    expected=fit_rg(prepared,*ordinary,intercept)
    actual=adapter.cross(reduce(scores[:,0]*scores[:,1]),left=compact[0],right=compact[1],overlap_rhs=c*n)
    np.testing.assert_allclose(actual.rg_reps,expected.rg_reps,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual.rg,expected.rg,rtol=1e-12,atol=1e-12)


def test_reference_block_ld_uses_frozen_chromosome_sources():
    rng=np.random.default_rng(218);n,m=81,41
    g=rng.normal(size=(n,m));g-=g.mean(0);rank=n-1
    groups=np.arange(m)//8;expected=np.zeros(groups.max()+1);chunks=[]
    for ch,(lo,hi) in enumerate(((0,19),(19,m)),1):
        f=g[:,lo:hi];norm=np.sum(f*f,axis=0);dot=f.T@f
        raw=np.sum(dot*dot,axis=1)/rank**2
        profile=raw-norm*norm.sum()/rank**3
        ids=np.unique(groups[lo:hi]);mass=[];directed=[];cross=[]
        for b in ids:
            take=groups[lo:hi]==b;mass.append(take.sum())
            directed.append(raw[take].sum());cross.append(norm[take].sum())
            expected[b]+=profile[take].sum()
        chunks.append(ChromosomeMoments(str(ch),'same',('all',),('x',),1,n,rank,ids,
            np.array(mass)[:,None],np.array(directed)[:,None,None],
            np.zeros((len(ids),1,1)),np.array(cross)[:,None,None]))
    ids,masses,actual=baseline_reference_block_ld(chunks)
    np.testing.assert_array_equal(ids,np.unique(groups))
    np.testing.assert_array_equal(masses,np.bincount(groups))
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
