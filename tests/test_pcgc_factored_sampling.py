"""Exact annotation/context factorization checked against explicit pair products."""
from itertools import combinations_with_replacement
import numpy as np
import pytest

from summit import gxeldcore
from summit.pcgc.gxe import context_pairs
from summit.pcgc.sampling import build_sampling_moments,partner_proposal,sample_partners
from prediction_helpers import prediction_threads
from test_pcgc_gxe_sampling import fixture


@pytest.mark.parametrize('degree',[2,3,4])
@pytest.mark.parametrize('importance',[False,True])
def test_factored_native_panels(degree,importance):
    rng=np.random.default_rng(7751)
    n,l,k,q=21,5,3,2
    base=rng.normal(size=(n,l,k))
    features=rng.normal(size=(n,q))
    response=rng.normal(size=n)
    partners=rng.integers(0,n-1,(n,l)); partners+=partners>=np.arange(n)[:,None]
    probability=rng.uniform(.1,1,n); probability/=probability.sum()
    leverage=rng.uniform(.3,1.4,n)
    ap=np.array(list(combinations_with_replacement(range(k),degree)),dtype=np.int64)
    cp=np.array(list(combinations_with_replacement(range(q),degree)),dtype=np.int64)
    first,last=3,17
    args=(base.reshape(n*l,k),partners,features,response,probability if importance else np.empty(0),
          leverage,ap,cp,first,last,degree==4,2**28,prediction_threads())
    left,right=map(np.asarray,gxeldcore.pcgc_factored_pair_panel(*args))
    neighbor=partners[first:last]
    weight=(1-probability[first:last,None])/probability[neighbor] if importance else np.full(neighbor.shape,n-1.)
    y=response[first:last,None]*response[neighbor]
    own=np.prod(features[first:last][:,cp],axis=-1)
    other=np.prod(features[neighbor][:,:,cp],axis=-1)
    context=own[:,None,:,None]*other[:,:,None,:]
    i,j=np.triu_indices(len(cp))
    context=(context[:,:,i,j]+context[:,:,j,i]*(i!=j)).reshape(-1,len(i))
    expected=context*(weight*y**(4-degree)/l).reshape(-1,1)
    if degree==4:
        expected=np.column_stack((expected,context*(weight**2*leverage[first:last,None]/l).reshape(-1,1)))
    np.testing.assert_allclose(left,np.prod(base[first:last][:,:,ap],axis=-1).reshape(left.shape),rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(right,expected,rtol=2e-14,atol=2e-13)
    with pytest.raises(RuntimeError,match='workspace'):
        gxeldcore.pcgc_factored_pair_panel(*args[:-2],1,args[-1])
    broken=partners.copy(); broken[first,0]=first
    with pytest.raises(ValueError,match='partners'):
        gxeldcore.pcgc_factored_pair_panel(args[0],broken,*args[2:])
    invalid=ap.copy(); invalid[0,0]=k
    with pytest.raises(ValueError,match='power'):
        gxeldcore.pcgc_factored_pair_panel(*args[:6],invalid,*args[7:])
    bad_features=features.copy(); bad_features[first,0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):
        gxeldcore.pcgc_factored_pair_panel(*args[:2],bad_features,*args[3:])


@pytest.mark.parametrize('fitted',[False,True])
@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('scheme',['exact','uniform','importance'])
def test_factored_covariance_matches_direct_sampled_polynomial(fitted,inverse,scheme,monkeypatch):
    import summit.pcgc.sampling as sampling
    data,_,_=fixture(n=37,m=31,fitted=fitted,inverse=inverse)
    x=data.pop('x'); n,m=x.shape
    contexts=np.column_stack((data['contexts'],data['contexts'][:,1]**2))
    features=contexts if inverse else contexts*(data['risk'].sensitivity/data['sd'])[:,None]
    # Overlapping, nonbinary annotations exercise signed relatedness products.
    annotations=np.column_stack((np.ones(m),np.linspace(.2,1.7,m),np.linspace(1.5,.1,m)))
    bases=np.stack([(x*a)@x.T/a.sum() for a in annotations.T],axis=-1)
    probabilities=partner_proposal(features,data['response']) if scheme=='importance' else np.full(n,1/n)
    partners=(np.array([np.delete(np.arange(n),i) for i in range(n)]) if scheme=='exact'
              else sample_partners(probabilities,7,75161))
    kernels=[]
    for base in np.moveaxis(bases,-1,0):
        for u,v in context_pairs(3):
            matrix=base*np.outer(features[:,u],features[:,v])
            if u!=v: matrix+=matrix.T.copy()
            np.fill_diagonal(matrix,0.)
            kernels.append(matrix)
    data.update(contexts=contexts,features=features,
        kernel_actions=np.asarray([v@data['response'] for v in kernels]).T,
        genotype_diagonal=np.stack([np.diag(bases[:,:,i]) for i in range(3)],axis=1),
        pair_kernels=bases[np.arange(n)[:,None],partners],partners=partners,
        partner_probabilities=probabilities if scheme=='importance' else None,sampled=scheme!='exact')
    expected=build_sampling_moments(**data,native=False)
    # Force several participant tiles as well as a final partial tile.
    monkeypatch.setattr(sampling,'sampling_tile_people',lambda n,l:11)
    evidence={}
    actual=build_sampling_moments(**data,native=True,threads=prediction_threads(),execution=evidence)
    assert evidence['pair_reduction']=='annotation_context_monomials'
    for name in ('constant','linear','quadratic','pair_constant','pair_linear','pair_quadratic'):
        np.testing.assert_allclose(getattr(actual,name),getattr(expected,name),rtol=2e-10,atol=3e-14)
    for theta in (np.zeros(18),np.linspace(-.2,.3,18)):
        np.testing.assert_allclose(actual.covariance(theta),expected.covariance(theta),rtol=2e-10,atol=3e-14)


def test_native_pair_means_match_explicit_kernels():
    rng=np.random.default_rng(19403)
    n,l,k,q=21,5,2,3
    base=rng.normal(size=(n,l,k)); features=rng.normal(size=(n,q))
    partners=rng.integers(0,n-1,(n,l)); partners+=partners>=np.arange(n)[:,None]
    probabilities=rng.uniform(.2,1,n); probabilities/=probabilities.sum()
    pairs=np.asarray(context_pairs(q),dtype=np.int64); c=k*len(pairs)
    upper=np.column_stack(np.triu_indices(c)).astype(np.int64)
    first,last=2,19
    args=(base.reshape(n*l,k),partners,features,probabilities,pairs,upper,first,last,2**26,prediction_threads())
    actual=np.asarray(gxeldcore.pcgc_pair_means(*args))
    neighbor=partners[first:last]
    f=features[first:last]
    context=np.stack([f[:,u,None]*features[neighbor,v] if u==v else
        f[:,u,None]*features[neighbor,v]+f[:,v,None]*features[neighbor,u] for u,v in pairs],axis=-1)
    kernels=(base[first:last,:,:,None]*context[:,:,None,:]).reshape(last-first,l,c)
    square=kernels[:,:,upper[:,0]]*kernels[:,:,upper[:,1]]
    weight=(1-probabilities[first:last,None])/probabilities[neighbor]
    np.testing.assert_allclose(actual,(square*weight[:,:,None]).mean(1),rtol=3e-14,atol=2e-13)
    with pytest.raises(RuntimeError,match='workspace'):
        gxeldcore.pcgc_pair_means(*args[:-2],1,args[-1])
    broken=partners.copy(); broken[first,0]=-1
    with pytest.raises(ValueError,match='partners'):
        gxeldcore.pcgc_pair_means(args[0],broken,*args[2:])


@pytest.mark.parametrize('exponent',[-400,400])
def test_reciprocal_kernel_scaling_uses_direct_products(exponent):
    data,_,base=fixture(n=24,m=19)
    data.pop('x')
    n=len(base)
    other=np.array([np.delete(np.arange(n),i) for i in range(n)])
    # Three annotations and three contexts admit the factorized reducer.
    phi=np.column_stack((data['contexts'],data['contexts'][:,1]**2))
    features=phi*(data['risk'].sensitivity/data['sd'])[:,None]
    kernels=[]
    for _ in range(3):
        for u,v in context_pairs(3):
            value=base*np.outer(features[:,u],features[:,v])
            if u!=v: value+=value.T.copy()
            np.fill_diagonal(value,0.)
            kernels.append(value)
    data.update(contexts=phi,features=features,partners=other,sampled=False,
        pair_kernels=np.repeat(base[np.arange(n)[:,None],other,None],3,axis=2),
        kernel_actions=np.asarray([value@data['response'] for value in kernels]).T,
        genotype_diagonal=np.repeat(data['genotype_diagonal'],3,axis=1))
    expected=build_sampling_moments(**data,native=False)
    data['pair_kernels']=np.ldexp(data['pair_kernels'],exponent)
    data['genotype_diagonal']=np.ldexp(data['genotype_diagonal'],exponent)
    data['contexts']=np.ldexp(data['contexts'],-exponent//2)
    data['features']=np.ldexp(data['features'],-exponent//2)
    evidence={}
    actual=build_sampling_moments(**data,native=True,threads=prediction_threads(),execution=evidence)
    assert evidence['pair_reduction']=='direct_pair_products'
    for name in ('constant','linear','quadratic','pair_constant','pair_linear','pair_quadratic'):
        np.testing.assert_allclose(getattr(actual,name),getattr(expected,name),rtol=2e-10,atol=2e-14)
