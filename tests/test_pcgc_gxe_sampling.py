"""Independent dense and finite-difference checks of sampling inference."""
import numpy as np
import pytest
from scipy.special import ndtr,ndtri
from summit.pcgc.sampling import build_sampling_moments,center_strata,nuisance_derivatives
from summit.pcgc.gxe import context_pairs
from summit.sumstats.binary import prepare_binary_risk,fit_binary_risk


def fixture(n=40,m=31,*,fitted=False,inverse=False):
    rng = np.random.default_rng(61573)
    x = rng.normal(size=(n,m))
    env = rng.uniform(-1,1,n)
    phi = np.column_stack((np.ones(n),env))
    y = np.arange(n)%2
    cov = env[:,None]
    risk = fit_binary_risk(y,.1,cov) if fitted else prepare_binary_risk(y,.1,population_risk=ndtr(-1.3+.3*env))
    sd = np.sqrt(1+.1*env)
    method = 'pcgc-inverse' if inverse else 'pcgc'
    features = phi if inverse else phi*(risk.sensitivity/sd)[:,None]
    response = risk.z/(risk.sensitivity/sd) if inverse else risk.z
    base = x@x.T/m
    kernels = []
    for u,v in context_pairs(2):
        matrix = base*np.outer(features[:,u],features[:,v])
        if u != v:
            matrix += matrix.T.copy()
        np.fill_diagonal(matrix,0)
        kernels.append(matrix)
    kernels = np.asarray(kernels)
    actions = np.einsum('cij,j->ic',kernels,response)
    return dict(x=x,contexts=phi,features=features,response=response,risk=risk,sd=sd,method=method,
                kernel_actions=actions,genotype_diagonal=(x*x).mean(1)[:,None],risk_covariates=cov if fitted else None),kernels,base


def panel(data,base,partners,sampled):
    args = {key:value for key,value in data.items() if key != 'x'}
    return build_sampling_moments(**args,pair_kernels=base[np.arange(len(base))[:,None],partners,None],partners=partners,sampled=sampled)


@pytest.mark.parametrize('inverse',[False,True])
def test_exact_sampling_polynomial_matches_dense_u_statistic(inverse):
    data,kernels,base = fixture(inverse=inverse)
    n = len(base)
    partners = np.array([np.delete(np.arange(n),i) for i in range(n)])
    moments = panel(data,base,partners,False)
    for theta in (np.zeros(3),np.array([.2,.06,-.01]),np.array([-.02,.01,.03])):
        y = data['response']
        error = np.outer(y,y)-np.einsum('c,cij->ij',theta,kernels)
        residual = kernels*error
        rows = center_strata(residual.sum(2).T,data['risk'].z>0)
        expected = (4*rows.T@rows-2*np.einsum('cij,dij->cd',residual,residual))/(n*(n-1))**2
        np.testing.assert_allclose(moments.covariance(theta)[:3,:3],expected,rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(moments.pair_covariance(theta),
            2*np.einsum('cij,dij->cd',residual,residual)/(n*(n-1))**2,rtol=2e-12,atol=1e-15)


@pytest.mark.parametrize('overlapping',[False,True])
def test_multi_annotation_covariance_polynomial_matches_dense_pairs(overlapping):
    data,_,_ = fixture(n=24,m=31)
    x,risk,sd,response = (data[name] for name in ('x','risk','sd','response'))
    n,m = x.shape
    phi = np.column_stack((data['contexts'],data['contexts'][:,1]**2))
    features = phi*(risk.sensitivity/sd)[:,None]
    annotations = np.column_stack((np.arange(m)<m//2,np.arange(m)>=m//2)).astype(float)
    if overlapping:
        annotations[:,0] = 1.
        annotations[:,1] = np.linspace(.2,1.8,m)
    bases = np.stack([(x*column)@x.T/column.sum() for column in annotations.T],axis=-1)
    kernels = []
    for base in np.moveaxis(bases,-1,0):
        for u,v in context_pairs(phi.shape[1]):
            matrix = base*np.outer(features[:,u],features[:,v])
            if u != v:
                matrix += matrix.T.copy()
            np.fill_diagonal(matrix,0.)
            kernels.append(matrix)
    kernels = np.asarray(kernels)
    c = len(kernels)
    partners = np.array([np.delete(np.arange(n),i) for i in range(n)])
    moments = build_sampling_moments(
        pair_kernels=bases[np.arange(n)[:,None],partners],partners=partners,sampled=False,
        kernel_actions=np.einsum('cij,j->ic',kernels,response),
        genotype_diagonal=np.stack([np.diag(bases[:,:,a]) for a in range(2)],axis=1),
        contexts=phi,features=features,response=response,risk=risk,sd=sd,method='pcgc')
    for theta in [np.zeros(c),np.linspace(-.03,.08,c),np.linspace(.15,-.1,c)]:
        residual = kernels*(np.outer(response,response)-np.einsum('c,cij->ij',theta,kernels))
        rows = center_strata(residual.sum(2).T,risk.z>0)
        pair = 2*np.einsum('cij,dij->cd',residual,residual)/(n*(n-1))**2
        expected = 4*rows.T@rows/(n*(n-1))**2-pair
        np.testing.assert_allclose(moments.covariance(theta)[:c,:c],expected,rtol=3e-11,atol=2e-15)
        np.testing.assert_allclose(moments.pair_covariance(theta),pair,rtol=3e-11,atol=2e-15)


@pytest.mark.parametrize('inverse',[False,True])
def test_complete_nuisance_derivative_matches_finite_difference(inverse):
    data,kernels,base = fixture(fitted=True,inverse=inverse)
    risk,sd,phi = data['risk'],data['sd'],data['contexts']
    C,IF,t,dz = nuisance_derivatives(risk,data['risk_covariates'],data['method'],sd)
    theta = np.array([.13,.05,.01])
    Hrow = np.einsum('cij,dij->icd',kernels,kernels)
    analytic = 2*data['kernel_actions'].T@(dz+data['response'][:,None]*t)-4*np.einsum('ir,icd,d->cr',t,Hrow,theta)
    for r in range(C.shape[1]):
        values = []
        for sign in (-1,1):
            alternate = prepare_binary_risk((risk.z>0).astype(float),risk.population_prevalence,
                population_risk=ndtr(ndtri(risk.population_risk)+sign*1e-5*C[:,r]))
            features = phi if inverse else phi*(alternate.sensitivity/sd)[:,None]
            response = alternate.z/(alternate.sensitivity/sd) if inverse else alternate.z
            bks = []
            for u,v in context_pairs(2):
                bk = base*np.outer(features[:,u],features[:,v])
                if u != v: bk += bk.T.copy()
                np.fill_diagonal(bk,0)
                bks.append(bk)
            bks = np.asarray(bks)
            H = np.einsum('cij,dij->cd',bks,bks)
            values.append(np.einsum('i,cij,j->c',response,bks,response)-H@theta)
        np.testing.assert_allclose((values[1]-values[0])/2e-5,analytic[:,r],rtol=2e-7,atol=2e-8)


@pytest.mark.parametrize('fitted',[False,True])
def test_partner_noise_correction_is_unbiased_for_full_pair_covariance(fitted):
    data,_,base = fixture(n=20,m=17,fitted=fitted)
    n = len(base)
    partners = np.array([np.delete(np.arange(n),i) for i in range(n)])
    exact = panel(data,base,partners,False)
    theta = np.array([.6,.12,.08])
    target = exact.covariance(theta)
    rng = np.random.default_rng(71389)
    draws = []
    for rep in range(400):
        other = rng.integers(0,n-1,(n,4))
        other += other >= np.arange(n)[:,None]
        draws.append(panel(data,base,other,True).covariance(theta))
    draws = np.asarray(draws)
    error = np.abs(draws.mean(0)-target)
    mcse = draws.std(0,ddof=1)/np.sqrt(len(draws))
    assert np.all(error <= 4.5*mcse+1e-14)


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('method',['pcgc','pcgc-inverse','pcgc-basis'])
def test_streamed_sampling_preserves_reference_and_matches_dense_actions(native,method):
    from summit.pcgc.gxe import prepare_gxe_moments,fit_gxe
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
    from prediction_helpers import prediction_threads
    data,kernels,base = fixture(n=84,m=79,inverse=method=='pcgc-inverse')
    x,phi,risk,sd = data['x'],data['contexts'],data['risk'],data['sd']
    options = dict(probes=113,seed=129,block_size=19,native=native,threads=prediction_threads())
    if method == 'pcgc-basis':
        options.update(basis=risk.sensitivity[:,None],coefficients=[1.])
    fixed,_ = prepare_gxe_moments(Operator(x),np.ones((x.shape[1],1)),risk,phi,method,liability_sd=sd,**options)
    actual,diagnostics = prepare_gxe_moments(Operator(x),np.ones((x.shape[1],1)),risk,phi,method,liability_sd=sd,
        sampling_partners=32,sampling_seed=472,**options)
    np.testing.assert_array_equal(actual.ldscores,fixed.ldscores)
    np.testing.assert_array_equal(actual.rhs_rows,fixed.rhs_rows)
    from summit.pcgc.sampling import partner_proposal,sample_partners
    probabilities = partner_proposal(data['features'],data['response'])
    partner = sample_partners(probabilities,32,472)
    expected = build_sampling_moments(**{key:value for key,value in data.items() if key != 'x'},
        pair_kernels=base[np.arange(len(base))[:,None],partner,None],partners=partner,
        partner_probabilities=probabilities)
    for name in ('constant','linear','quadratic'):
        np.testing.assert_allclose(getattr(actual.sampling_moments,name),getattr(expected,name),atol=3e-14,rtol=3e-11)
    result = fit_gxe(actual,block_ids=np.arange(x.shape[1])%8)
    assert result['uncertainty_method'] == 'individual_sampling_fixed_genome_v1'
    assert np.isfinite(result['population_heritability_se'])
    assert diagnostics['genotype_passes'] == 2
    assert diagnostics['peak_planned_workspace_bytes'] <= 2**30


def test_sampling_admission_fails_before_genotype_io():
    from summit.pcgc.gxe import prepare_gxe_moments,plan_gxe_reference
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
    data,_,_ = fixture(n=40,m=31)
    op = Operator(data['x'])
    with pytest.raises(MemoryError):
        prepare_gxe_moments(op,np.ones((31,1)),data['risk'],data['contexts'],liability_sd=data['sd'],
            sampling_partners=128,architecture_probes=32,memory_bytes=16*1024**2,native=False)
    assert op.observed_passes == 0
    plan = plan_gxe_reference(num_samples=40,num_variants=31,num_contexts=2,num_annotations=1,
        probes=71,sampling_partners=32,architecture_probes=8,memory_bytes=2**30,block_size=13)
    _,diag = prepare_gxe_moments(op,np.ones((31,1)),data['risk'],data['contexts'],liability_sd=data['sd'],
        probes=71,sampling_partners=32,architecture_probes=8,memory_bytes=2**30,block_size=13,native=False)
    assert plan['peak_planned_workspace_bytes'] == diag['peak_planned_workspace_bytes']


@pytest.mark.parametrize('fitted',[False,True])
def test_importance_partner_covariance_is_unbiased_and_basis_invariant(fitted):
    from summit.pcgc.sampling import partner_proposal,sample_partners
    data,_,base = fixture(n=20,m=17,fitted=fitted,inverse=True)
    n = len(base)
    probabilities = partner_proposal(data['features'],data['response'])
    changed = partner_proposal(data['features']@np.array([[1.,.4],[-.2,1.3]]),data['response'])
    np.testing.assert_allclose(probabilities,changed,rtol=2e-14,atol=1e-16)
    exact = panel(data,base,np.array([np.delete(np.arange(n),i) for i in range(n)]),False)
    theta = np.array([.6,.12,.08]); target = exact.covariance(theta)
    draws = []
    for seed in range(600):
        other = sample_partners(probabilities,6,seed+810391)
        mom = build_sampling_moments(**{key:value for key,value in data.items() if key != 'x'},
            pair_kernels=base[np.arange(n)[:,None],other,None],partners=other,partner_probabilities=probabilities)
        draws.append(mom.covariance(theta))
    draws = np.asarray(draws)
    assert np.all(abs(draws.mean(0)-target) <= 4.5*draws.std(0,ddof=1)/np.sqrt(len(draws))+1e-14)
