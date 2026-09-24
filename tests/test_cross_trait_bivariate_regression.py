"""Q=1 limiting case against the independent bivariate SUMMIT fitter."""
from types import SimpleNamespace
import numpy as np
from summit.context.cross_trait_fit import assemble_cross_trait_normal_equations,solve_cross_trait_normal_equations
from summit.inference.jackknife import JackknifeDesign,JackknifeSpec
from summit.inference.rgcore import prepare_rg,fit_intercept,fit_rg


def test_baseline_rg_matches_bivariate_sumcore_on_same_dense_panel():
    rng=np.random.default_rng(9167);n,m=501,400
    g=rng.normal(size=(n,m));g-=g.mean(0);g*=np.sqrt(n/np.sum(g*g,axis=0))
    effects=rng.multivariate_normal([0,0],[[.55,.30],[.30,.50]],size=m)/np.sqrt(m)
    y=g@effects+rng.multivariate_normal([0,0],[[.45,.12],[.12,.50]],size=n)
    y*=np.sqrt(n/np.sum(y*y,axis=0))
    scores=g.T@y;kernel=g@g.T/m;gg=np.sum(kernel*kernel)
    def context(ix,iy):
        equations=assemble_cross_trait_normal_equations(genetic_gram=[[gg]],
            genetic_rhs=[scores[:,ix]@scores[:,iy]/m],genetic_residual=[[np.trace(kernel)]],
            residual_gram=[[n]],residual_rhs=[y[:,ix]@y[:,iy]],num_basis=1,annotation_masses=[m])
        return solve_cross_trait_normal_equations(equations).coefficients[0]
    xx,yy,xy=context(0,0),context(1,1),context(0,1)
    snps=np.array([str(j) for j in range(m)])
    # Residual-profiled exact LD moments, independently from SNP dot products.
    ld=(np.sum((g.T@g)**2,axis=1)/n**2-m/n)[:,None]
    trace=SimpleNamespace(nsnps=m,nbins=1,snps=snps,annot=np.ones((m,1)),ldscores=ld)
    matched=SimpleNamespace(nsnps=m,snps=snps,nsamp=n,n_scale=n)
    jk=JackknifeDesign.from_trace_view(trace,JackknifeSpec.parse(20))
    product=scores[:,0]*scores[:,1]/n
    prepared=prepare_rg(trace,matched,matched,jk,summary_y=product)
    def within(value):
        return SimpleNamespace(sigma_reps=np.full((jk.nrep+1,1),value),h2_reps=np.full((jk.nrep+1,1),value))
    h1,h2=within(xx),within(yy)
    intercept=fit_intercept(trace,matched,matched,jk,h1,h2,summary_y=product,fixed_c=y[:,0]@y[:,1]/n)
    ordinary=fit_rg(prepared,h1,h2,intercept)
    expected=xy/np.sqrt(xx*yy);observed=ordinary.rg_reps[-1,0]
    np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12)
    assert abs(observed-expected)<ordinary.rg[0,1]
