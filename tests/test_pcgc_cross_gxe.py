"""Independent person-kernel checks; deliberately small dense matrices only."""
import numpy as np
import pytest

from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec
from summit.pcgc.cross_gxe_model import binary_trait, quantitative_trait
from summit.pcgc.cross_gxe import prepare_cross_gxe, pair_layout


def fixture(kind="qq", overlap=8, qa=2, qb=3, seed=913, projected=True):
    rng = np.random.default_rng(seed)
    na, nb, m = 17, 19, 24
    n = na+nb-overlap
    ids = tuple((str(i), str(i)) for i in range(n))
    rows = [np.arange(na), np.r_[np.arange(overlap), np.arange(na, n)]]
    x = rng.normal(size=(n, m))
    a = rng.uniform(.2, 1, (m, 2))
    traits = []
    for index, (code, q, rr) in enumerate(zip(kind, (qa, qb), rows)):
        phi = np.column_stack([np.ones(len(rr)), rng.uniform(-1, 1, (len(rr), q-1))])
        samples = tuple(ids[i] for i in rr)
        options = dict(context_names=tuple("context"+str(i) for i in range(q)))
        if code == "b":
            t = binary_trait(str(index), samples, np.arange(len(rr)) % 2, phi,
                liability_sd=np.linspace(.7, 1.5, len(rr)), selection="outcome_independent",
                population_risk=np.linspace(.002, .8, len(rr)), **options)
        else:
            design = np.column_stack((np.ones(len(rr)), np.column_stack([np.cos((j+1)*np.arange(len(rr))*2*np.pi/len(rr)) for j in range(1+index)]))) if projected else None
            t = quantitative_trait(str(index), samples, rng.normal(size=len(rr)), phi,
                population_sd=1.7, fixed_effects=design, **options)
        traits.append(t)
    return x, a, traits, ids, rows


def dense_oracle(x, a, traits, rows, probes=37, seed=41):
    """Materialize each rectangular test matrix and projected diagonal basis."""
    z = GlobalVariantProbeSpec(seed, 0, probes).generate(np.arange(len(a)))
    features, projectors, response = [], [], []
    for trait, rr in zip(traits, rows):
        p = np.eye(len(rr))-trait.fixed_basis@trait.fixed_basis.T
        projectors.append(p)
        features.append([p@(x[rr]*(trait.sensitivity*trait.contexts[:, q])[:, None])
                         for q in range(trait.q)])
        response.append(p@trait.response)
    result = {}
    for left in range(len(traits)):
        for right in range(left, len(traits)):
            layout = pair_layout(traits[left].q, traits[right].q, left == right)
            pa, pb = projectors[left], projectors[right]
            _, ia, ib = np.intersect1d(rows[left], rows[right], return_indices=True)
            nuisance = [np.outer(pa[:, i], pb[j]) for i, j in zip(ia, ib)]
            c = np.array([[h[i, j] for h in nuisance] for i, j in zip(ia, ib)])
            source = []
            random_source = []
            for col in a.T:
                for orientations in layout:
                    source.append(sum((features[left][u]*col)@features[right][v].T for u,v in orientations))
                    random_source.append(sum(
                        (features[left][u]@(np.sqrt(col)[:,None]*z))@
                        (features[right][v]@(np.sqrt(col)[:,None]*z)).T/probes
                        for u,v in orientations))
            rhs = np.empty((len(a), len(layout)))
            exact = np.empty((len(a), len(layout), len(source)))
            stochastic = np.empty_like(exact)
            tests = []
            for variant in range(len(a)):
                tests.append([])
                for p, orientations in enumerate(layout):
                    raw = sum(np.outer(features[left][u][:, variant], features[right][v][:, variant])
                              for u,v in orientations)
                    corrected = raw.copy()
                    if len(ia):
                        coeff = np.linalg.solve(c, raw[ia,ib])
                        corrected -= sum(w*h for w,h in zip(coeff, nuisance))
                    tests[-1].append(corrected)
                    assert np.max(np.abs(corrected[ia,ib]), initial=0) < 2e-10
                    rhs[variant,p] = response[left]@corrected@response[right]
                    for s, (g, gh) in enumerate(zip(source, random_source)):
                        exact[variant,p,s] = np.sum(corrected*g)
                        # Production removes the EXACT global source diagonal
                        # from the stochastic full product, not its noisy diagonal.
                        stochastic[variant,p,s] = np.sum(raw*gh)-np.sum((raw-corrected)*g)
            result[left,right] = dict(rhs=rhs, exact=exact, stochastic=stochastic,
                                      tests=np.array(tests), source=np.array(source))
    return result


@pytest.mark.parametrize("kind", ["qq", "bq", "bb"])
@pytest.mark.parametrize("overlap", [0, 8, 17])
@pytest.mark.parametrize("native", [False, True])
def test_explicit_rectangular_kernels(kind, overlap, native):
    from prediction_helpers import prediction_threads
    x,a,traits,ids,rows = fixture(kind, overlap)
    actual = prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                              probes=37,seed=41,native=native,block_size=7,probe_tile=11,
                              threads=prediction_threads())
    expected = dense_oracle(x,a,traits,rows)
    for pair in actual.pairs:
        value = expected[pair.left,pair.right]
        np.testing.assert_allclose(pair.rhs_rows,value["rhs"],rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(pair.ldscores*pair.sample_product,value["stochastic"],rtol=2e-10,atol=2e-10)
    assert actual.diagnostics["genotype_passes"] == 2
    assert actual.diagnostics["variant_visits"] == 2*len(a)


def test_projected_diagonal_deletion_leaves_residual_contamination():
    x,a,traits,ids,rows = fixture("qq",17)
    d = dense_oracle(x,a,traits,rows)[0,1]
    pa,pb = [np.eye(len(t.samples))-t.fixed_basis@t.fixed_basis.T for t in traits]
    _,ia,ib = np.intersect1d(*rows,return_indices=True)
    residual = sum((1+i/len(ia))*np.outer(pa[:,u],pb[v]) for i,(u,v) in enumerate(zip(ia,ib)))
    np.testing.assert_allclose(np.einsum("jpab,ab->jp",d["tests"],residual),0,atol=2e-11)
    naive=d["source"][0].copy()
    naive[ia,ib]=0
    assert abs(np.sum(naive*residual)) > .01


def test_chunk_probe_tile_and_native_invariance():
    x,a,traits,ids,rows = fixture()
    from prediction_helpers import prediction_threads
    runs = [prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
             probes=37,seed=41,native=native,block_size=block,probe_tile=tile,context_tile=context_tile,
             threads=prediction_threads()) for native,block,tile,context_tile in
             [(False,5,7,1),(False,24,37,4),(True,9,13,2)]]
    for run in runs[1:]:
        for p,r in zip(runs[0].pairs,run.pairs):
            for name in ("rhs_rows","ldscores","probe_grams"):
                np.testing.assert_allclose(getattr(p,name),getattr(r,name),rtol=2e-10,atol=2e-10)


def test_invalid_selection_and_missing_context_fail_before_io():
    from dataclasses import replace
    x,a,traits,ids,rows = fixture("bb",8)
    bad=replace(traits[0],metadata={**traits[0].metadata,"selection":"own_case_status"})
    op=ArraySequentialGenotypeOperator(x)
    with pytest.raises(ValueError,match="joint selection"):
        prepare_cross_gxe(op,a,[bad,traits[1]],ids,native=False)
    assert op.observed_passes == 0
    with pytest.raises(ValueError,match="selection"):
        replace(traits[0],metadata={**traits[0].metadata,"selection":"joint_controls"})
    phi=traits[0].contexts.copy();phi[0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        replace(traits[0],contexts=phi)


def exact_moments(kind="qq",overlap=8,qa=2,qb=2):
    from dataclasses import replace
    x,a,traits,ids,rows=fixture(kind,overlap,qa,qb)
    moments=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                             probes=19,seed=41,native=False)
    oracle=dense_oracle(x,a,traits,rows,probes=19)
    for pair in moments.pairs:
        pair.ldscores=oracle[pair.left,pair.right]["exact"]/pair.sample_product
        pair.probe_grams[:]=pair.probe_grams.mean(0)
    return moments,oracle,(x,a,traits,ids,rows)


def test_known_covariance_and_frozen_target_deletion():
    from summit.pcgc.cross_gxe_fit import equations,fit_cross_gxe,evaluate_cross_contexts
    moments,oracle,_=exact_moments()
    truths={}
    for pair in moments.pairs:
        k=moments.annotations.shape[1]
        theta=np.tile([.2,.07,.03] if pair.left==pair.right else [.08,.01,-.02,.025],k)
        source=oracle[pair.left,pair.right]["source"]
        mass=np.repeat(moments.annotations.sum(0),pair.p)
        covariance=np.einsum("c,cij->ij",theta/mass,source)
        pair.rhs_rows=np.einsum(
            "jpuv,uv->jp",oracle[pair.left,pair.right]["tests"],covariance)
        truths[pair.left,pair.right]=theta
        keep=np.arange(len(moments.annotations))%3!=0
        test=np.einsum("ja,jpuv->apuv",moments.annotations[keep],
                        oracle[pair.left,pair.right]["tests"][keep])
        retained=np.repeat(moments.annotations[keep].sum(0),pair.p)
        test=test.reshape(len(theta),*covariance.shape)/retained[:,None,None]
        H=np.einsum("cij,dij->cd",test,source/mass[:,None,None])
        b=np.einsum("cij,ij->c",test,covariance)
        actual=equations(moments,pair,keep)
        np.testing.assert_allclose(actual[0],H,rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(actual[1],b,rtol=2e-11,atol=2e-11)
    fit=fit_cross_gxe(moments,block_ids=np.arange(len(moments.annotations))%4)
    for pair in moments.pairs:
        begin,end=fit["pairs"][f"{pair.left}:{pair.right}"]["coefficient_slice"]
        np.testing.assert_allclose(fit["components"][begin:end],truths[pair.left,pair.right],atol=2e-11)
    derived=evaluate_cross_contexts(fit,0,1,[1,.5],[1,-.3])
    assert derived["genetic_correlation"] is not None
    np.testing.assert_allclose(np.asarray(fit["covariance"]),0,atol=2e-20)


@pytest.mark.parametrize("kind",["qq","bq","bb"])
@pytest.mark.parametrize("qb",[2,3])
def test_reversal_signed_scores_and_basis_transformation(kind,qb):
    from dataclasses import replace
    from summit.pcgc.cross_gxe_fit import fit_cross_gxe
    x,a,traits,ids,rows=fixture(kind,8,2,qb)
    a=np.ones((len(a),1))
    def prep(tt):
        return prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,tt,ids,
                                probes=509,seed=88,native=False)
    original=prep(traits)
    reverse=prep(traits[::-1])
    ab=original.pairs[1];ba=reverse.pairs[1]
    index=np.arange(2*qb).reshape(2,qb).T.ravel()
    np.testing.assert_allclose(ab.ldscores[:,index][:,:,index],ba.ldscores,atol=2e-11)
    np.testing.assert_allclose(ab.rhs_rows[:,index],ba.rhs_rows,atol=2e-11)
    scale=np.array([[1.,.3],[0.,1.8]])
    changed_traits=[replace(traits[0],contexts=traits[0].contexts@scale),traits[1]]
    changed=prep(changed_traits)
    # Isolate the fitted basis identity from weak randomized normal equations
    # in this deliberately tiny cohort. The stochastic reversal is checked above.
    for moments,tt in ((original,traits),(changed,changed_traits)):
        oracle=dense_oracle(x,a,tt,rows)
        for pair in moments.pairs:
            pair.ldscores=oracle[pair.left,pair.right]["exact"]/pair.sample_product
            pair.probe_grams[:]=pair.probe_grams.mean(0)
    f0,f1=fit_cross_gxe(original),fit_cross_gxe(changed)
    omega0=np.array(f0["pairs"]["0:1"]["omega_total"])
    omega1=np.array(f1["pairs"]["0:1"]["omega_total"])
    np.testing.assert_allclose(omega1,np.linalg.solve(scale,omega0),rtol=2e-10,atol=2e-10)
    signs=np.where(np.arange(len(a))%2,1.,-1.)
    signed=prepare_cross_gxe(ArraySequentialGenotypeOperator(x*signs),a,traits,ids,
                            probes=19,native=False)
    np.testing.assert_allclose(signed.signed_scores[0],original.signed_scores[0]*signs[:,None],atol=2e-11)
    # Finite probe realizations change under source-allele flips; exact moments do not.
    dense=dense_oracle(x,a,traits,rows)
    flipped=dense_oracle(x*signs,a,traits,rows)
    np.testing.assert_allclose(dense[0,1]["exact"],flipped[0,1]["exact"],atol=2e-10)


def test_additive_unprojected_reduction():
    from summit.pcgc.cross import prepare_pair,QuantitativeResponse,SELECTION_CONTRACT
    x,a,traits,ids,rows=fixture("qq",8,1,1,projected=False)
    actual=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,probes=71,seed=13,native=False)
    old=prepare_pair(ArraySequentialGenotypeOperator(x),a,*[QuantitativeResponse(t.response) for t in traits],*rows,
                     selection_contract=SELECTION_CONTRACT,probes=71,seed=13,native=False)
    cross=actual.pairs[1]
    np.testing.assert_allclose(cross.ldscores[:,0]*cross.sample_product,old.cross.ldscores*len(x)**2,atol=2e-10)
    np.testing.assert_allclose(cross.rhs_rows[:,0],old.cross.rhs_rows,atol=2e-11)


def test_reference_monte_carlo_convergence():
    x,a,traits,ids,rows=fixture("bq",8,1,1)
    exact=dense_oracle(x,a,traits,rows)[0,1]["exact"]
    means=[]
    for seed in range(40):
        out=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                             probes=31,seed=seed,native=False)
        means.append(out.pairs[1].ldscores*out.pairs[1].sample_product)
    errors=np.asarray(means)-exact
    # Test unbiased rows aggregated over seeds, not a favorable single realization.
    norm=np.linalg.norm(errors.mean(0))
    mc=np.sqrt(np.sum(errors.var(0,ddof=1))/len(errors))
    assert norm<3*mc
    assert np.linalg.norm(errors.mean(0))<.35*np.mean(np.linalg.norm(errors.reshape(len(errors),-1),axis=1))


def test_singular_annotations_and_nonpositive_variance_denominators():
    from summit.pcgc.cross_gxe_fit import fit_cross_gxe,evaluate_cross_contexts
    moments,oracle,_=exact_moments(qa=1,qb=1)
    for pair in moments.pairs:
        H=pair.ldscores.copy()
        pair.rhs_rows=pair.sample_product*np.einsum("jpc,c->jp",H,
            np.array([-.1,-.1]) / moments.annotations.sum(0))
    fit=fit_cross_gxe(moments)
    result=evaluate_cross_contexts(fit,0,1,[1],[1])
    assert result["genetic_correlation"] is None
    assert result["correlation_status"]=="undefined_nonpositive_denominator"
    x,a,traits,ids,rows=fixture("qq",0,1,1)
    out=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),np.ones_like(a),traits,ids,
                         probes=71,native=False)
    with pytest.raises(ValueError,match="rank deficient"):
        fit_cross_gxe(out)


def test_supplied_fitted_risks_and_common_heterogeneous_scale():
    from scipy.special import ndtr
    n=120
    ids=tuple((str(i),str(i)) for i in range(n))
    env=np.linspace(-1,1,n);phi=np.column_stack((np.ones(n),env))
    y=(np.arange(n)%5==0).astype(float)
    t=binary_trait("a",ids,y,phi,context_names=("i","e"),liability_sd=1.,
                   selection="outcome_independent",risk_covariates=env[:,None])
    assert t.metadata["risk_source"]=="fitted_probit"
    common=dict(name="a",samples=ids,y=y,contexts=phi,context_names=("i","e"),
                selection="own_case_status",population_risk=ndtr(-2+env),prevalence=.03)
    u=binary_trait(liability_sd=1.,**common)
    v=binary_trait(liability_sd=np.full(n,2.),**common)
    np.testing.assert_allclose(v.sensitivity,u.sensitivity/2)
    np.testing.assert_allclose(v.response,u.response)
    with pytest.raises(ValueError,match="unit"):
        binary_trait("a",ids,y,phi,context_names=("i","e"),liability_sd=2.,
                     selection="outcome_independent",risk_covariates=env[:,None])


@pytest.mark.parametrize("odds",[(1.,1.),(5.,.3),(20.,7.)])
def test_selected_bivariate_normal_derivative(odds):
    from scipy.integrate import quad
    from scipy.special import ndtr,ndtri
    k1,k2=.03,.24
    sd1,sd2=.8,1.7
    eta1,eta2=ndtri(k1),ndtri(k2)
    p1=odds[0]*k1/(1-k1+odds[0]*k1)
    p2=odds[1]*k2/(1-k2+odds[1]*k2)
    z1=(np.arange(2)-p1)/np.sqrt(p1*(1-p1))
    z2=(np.arange(2)-p2)/np.sqrt(p2*(1-p2))
    inclusion=np.outer([1,odds[0]],[1,odds[1]])
    def moment(c):
        rho=c/(sd1*sd2)
        joint=quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*
                   ndtr((eta2-rho*x)/np.sqrt(1-rho*rho)),-np.inf,eta1,epsabs=1e-13)[0]
        table=np.array([[1-k1-k2+joint,k2-joint],[k1-joint,joint]])*inclusion
        return np.sum(table*np.outer(z1,z2))/table.sum()
    ids=(("f","a"),("f","b"),("f","c"))
    common=dict(samples=ids,y=[0,1,0],contexts=np.ones((3,1)),context_names=("i",),selection="own_case_status")
    a=binary_trait("a",population_risk=np.full(3,k1),liability_sd=sd1,sampling_odds=odds[0],**common)
    b=binary_trait("b",population_risk=np.full(3,k2),liability_sd=sd2,sampling_odds=odds[1],**common)
    derivative=(moment(1e-5)-moment(-1e-5))/2e-5
    np.testing.assert_allclose(derivative,a.sensitivity[0]*b.sensitivity[0],rtol=2e-8,atol=1e-10)


def test_nonfactorizing_selection_cannot_use_marginal_weights():
    k1,k2=.1,.2
    population=np.outer([1-k1,k1],[1-k2,k2])
    inclusion=np.array([[.7,.1],[.2,.8]])
    table=population*inclusion;table/=table.sum()
    p1=table[1].sum();p2=table[:,1].sum()
    centered=np.outer(np.arange(2)-p1,np.arange(2)-p2)
    # Selection creates dependence even at zero latent covariance.
    assert abs(np.sum(table*centered))>.01


def test_probe_gram_draws_preserve_cross_pair_dependence():
    x,a,traits,ids,rows=fixture("qq",8,1,1)
    result=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                            probes=71,seed=221,native=False)
    z=GlobalVariantProbeSpec(221,0,71).generate(np.arange(len(a)))
    f=[]
    for trait,index in zip(traits,rows):
        p=np.eye(len(index))-trait.fixed_basis@trait.fixed_basis.T
        f.append(p@x[index])
    for pair in result.pairs:
        target=[(f[pair.left]*w)@f[pair.right].T/w.sum() for w in a.T]
        expected=np.empty_like(pair.probe_grams)
        for probe in range(len(z.T)):
            for source,w in enumerate(a.T):
                left=f[pair.left]@(np.sqrt(w)*z[:,probe])
                right=f[pair.right]@(np.sqrt(w)*z[:,probe])
                kernel=np.outer(left,right)/w.sum()
                expected[probe,:,source]=[np.sum(t*kernel) for t in target]
        np.testing.assert_allclose(pair.probe_grams,expected,atol=2e-10,rtol=2e-10)
    aa=result.pairs[0].probe_grams.reshape(71,-1)
    ab=result.pairs[1].probe_grams.reshape(71,-1)
    assert np.linalg.norm((aa-aa.mean(0)).T@(ab-ab.mean(0)))>1.


def test_memory_and_high_leverage_reject_before_genotype_traversal():
    from dataclasses import replace
    x,a,traits,ids,rows=fixture("qq",8,1,1)
    op=ArraySequentialGenotypeOperator(x)
    with pytest.raises(MemoryError,match="memory budget"):
        prepare_cross_gxe(op,a,traits,ids,native=False,memory_bytes=1024)
    assert op.observed_passes==0
    design=np.zeros((len(traits[0].samples),1));design[0,0]=1
    bad=replace(traits[0],fixed_basis=design)
    with pytest.raises(ValueError,match="leverage"):
        prepare_cross_gxe(op,a,[bad,traits[1]],ids,native=False)
    assert op.observed_passes==0


def test_population_metrics_and_joint_correlation_gradient():
    from summit.pcgc.cross_gxe_fit import derived_covariance
    # Deliberately correlated AA, BB and AB estimates: independent error
    # propagation gives a different result.
    coefficient=np.array([.4,.12,.3])
    covariance=np.array([[.004,.001,.0005],[.001,.002,.0007],[.0005,.0007,.003]])
    pairs={}
    for index,key in enumerate(("0:0","0:1","1:1")):
        pairs[key]=dict(omega=[[[coefficient[index]]]],pair_table=[[0,0]],
                        coefficient_slice=[index,index+1],weak_normal_equations=False)
    fit=dict(components=coefficient,covariance=covariance,pairs=pairs,
             uncertainty_status="experimental")
    unit=np.ones((1,1,1))
    value=derived_covariance(fit,0,1,unit,unit,unit,total_variances=[2.,3.])
    r=coefficient[1]/np.sqrt(coefficient[0]*coefficient[2])
    step=1e-6
    def ratio(theta):
        return theta[1]/np.sqrt(theta[0]*theta[2])
    gradient=np.array([(ratio(coefficient+step*e)-ratio(coefficient-step*e))/(2*step)
                       for e in np.eye(3)])
    np.testing.assert_allclose(value["genetic_correlation"],r)
    np.testing.assert_allclose(value["genetic_correlation_se"]**2,gradient@covariance@gradient,rtol=1e-9)
    np.testing.assert_allclose(value["heritabilities"],[.2,.1])
    assert not np.isclose(gradient@covariance@gradient,np.sum(gradient**2*np.diag(covariance)))
    reversed_value=derived_covariance(fit,1,0,unit,unit,unit)
    np.testing.assert_allclose(reversed_value["genetic_correlation_se"],value["genetic_correlation_se"])
    with pytest.raises(ValueError,match="positive semidefinite"):
        derived_covariance(fit,0,1,unit,unit,2*unit)
    fit["components"]=[.4,.5,.3]
    value=derived_covariance(fit,0,1,unit,unit,unit)
    assert value["genetic_correlation"]>1
    assert value["correlation_outside_unit_interval"]

def test_fit_memory_and_joint_probe_validation():
    from summit.pcgc.cross_gxe_fit import fit_cross_gxe
    moments,_,_=exact_moments(qa=1,qb=1)
    with pytest.raises(MemoryError,match="joint uncertainty"):
        fit_cross_gxe(moments,memory_bytes=1024)
    moments.pairs[0].probe_grams=moments.pairs[0].probe_grams[:-1]
    with pytest.raises(ValueError,match="identical probe counts"):
        moments.validate()

@pytest.mark.parametrize("kind",["bq","bb"])
def test_constant_context_binary_additive_reduction(kind):
    from summit.pcgc.cross import prepare_pair,QuantitativeResponse,SELECTION_CONTRACT
    from summit.sumstats.binary import prepare_binary_risk
    x,a,traits,ids,rows=fixture(kind,0,1,1,projected=False)
    old_traits=[]
    for index,code in enumerate(kind):
        t=traits[index]
        if code=="b":
            y=np.arange(len(t.samples))%2
            k=np.linspace(.02,.4,len(y))
            traits[index]=binary_trait(t.name,t.samples,y,t.contexts,context_names=t.context_names,
                liability_sd=1.,selection="own_case_status",population_risk=k,prevalence=.1)
            old_traits.append(prepare_binary_risk(y,.1,population_risk=k))
        else:
            old_traits.append(QuantitativeResponse(t.response))
    actual=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                            probes=71,seed=13,native=False)
    old=prepare_pair(ArraySequentialGenotypeOperator(x),a,*old_traits,*rows,
        selection_contract=SELECTION_CONTRACT,probes=71,seed=13,native=False)
    pair=actual.pairs[1]
    np.testing.assert_allclose(pair.ldscores[:,0]*pair.sample_product,old.cross.ldscores*len(x)**2,
                               rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(pair.rhs_rows[:,0],old.cross.rhs_rows,rtol=2e-10,atol=2e-10)


def test_nearly_collinear_annotation_failure_is_explicit():
    from summit.pcgc.cross_gxe_fit import fit_cross_gxe
    x,a,traits,ids,rows=fixture("qq",0,1,1)
    a=np.column_stack((np.ones(len(a)),1+1e-6*np.linspace(-1,1,len(a))))
    moments=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,probes=127,native=False)
    with pytest.raises(ValueError,match="rank deficient"):
        fit_cross_gxe(moments)


def test_binary_case_relabeling_changes_cross_sign_only():
    x,a,traits,ids,_=fixture("bq",8,2,2)
    t=traits[0];n=len(t.samples)
    flipped=binary_trait(t.name,t.samples,1-np.arange(n)%2,t.contexts,
        context_names=t.context_names,liability_sd=np.linspace(.7,1.5,n),
        selection="outcome_independent",population_risk=1-np.linspace(.002,.8,n))
    original=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,
                              probes=37,native=False)
    changed=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,[flipped,traits[1]],ids,
                             probes=37,native=False)
    for before,after in zip(original.pairs,changed.pairs):
        np.testing.assert_allclose(before.ldscores,after.ldscores,rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(before.rhs_rows,(-1 if before.left!=before.right else 1)*after.rhs_rows,
                                   rtol=2e-10,atol=2e-10)


def test_finite_block_intervals_and_missing_trait_uncertainty():
    from scipy.stats import t
    from summit.pcgc.cross_gxe_fit import linear_interval
    result=dict(components=[.3,.1],snp_block_covariance=[[.04,.01],[.01,.02]],
                reference_mc_covariance=[[0.,0.],[0.,0.]],jackknife_block_count=8,
                reference_probe_count=32,confidence_level=.95)
    interval=linear_interval(result,[1.,0.])
    np.testing.assert_allclose(interval["interval"],.3+np.array([-1,1])*t.ppf(.975,7)*.2)
    assert interval["degrees_of_freedom"]==pytest.approx(7)
    result["reference_mc_covariance"]=[[.04,0.],[0.,0.]]
    interval=linear_interval(result,[1.,0.])
    assert interval["degrees_of_freedom"]>7
    assert interval["reference_variance_fraction"]==pytest.approx(.5)
    result["jackknife_block_count"]=0
    assert linear_interval(result,[1.,0.])["interval"] is None
    with pytest.raises(ValueError,match="confidence"):
        linear_interval(result,[1.,0.],confidence=1)

def test_inverse_reference_diagnostics_against_exact_inverse_average():
    from itertools import product
    from summit.pcgc.cross_gxe_fit import _reference_inverse_diagnostics
    # Enumerate the complete independent Rademacher draw law. The true bias
    # comes from actual inverses, independently of the Taylor calculation.
    estimates=[];predictions=[]
    for signs in product((-1.,1.),repeat=8):
        draws=1+.05*np.asarray(signs)
        h=np.array([[draws.mean()]])
        theta=np.array([2./h[0,0]])
        diagnostics=_reference_inverse_diagnostics(h,(draws-draws.mean())[:,None,None],theta)
        estimates.append(theta[0]);predictions.append(diagnostics["reference_inverse_second_order_bias"][0])
    exact_bias=np.mean(estimates)-2.
    assert exact_bias>0
    np.testing.assert_allclose(np.mean(predictions),exact_bias,rtol=.01)
    # Directed matrices need their actual ordered inverse products.
    h=np.array([[2.,.5],[-.2,1.5]])
    delta=np.array([[.03,-.02],[.04,.01]])
    theta=np.array([.1,-.04])
    diagnostic=_reference_inverse_diagnostics(h,np.stack((delta,-delta)),theta)
    exact=(np.linalg.solve(h+delta,h@theta)+np.linalg.solve(h-delta,h@theta))/2-theta
    np.testing.assert_allclose(diagnostic["reference_inverse_second_order_bias"],exact,rtol=.002,atol=1e-10)
    assert diagnostic["reference_inverse_rms_perturbation"]>0
    zero=_reference_inverse_diagnostics(h,np.zeros((8,2,2)),theta)
    assert zero["reference_inverse_rms_perturbation"]==0
    assert zero["reference_inverse_second_order_bias"]==[0.,0.]

@pytest.mark.parametrize("native",[False,True])
def test_projection_subtraction_cancellation_uses_direct_products(native):
    from prediction_helpers import prediction_threads
    rng=np.random.default_rng(28729)
    n,m=96,8
    x=rng.normal(size=(n,m));a=np.ones((m,1))
    x[:,0]*=1e4  # Expose cancellation with a large removed direction.
    ids=tuple(("f",str(i)) for i in range(n))
    trait=quantitative_trait("trait",ids,rng.normal(size=n),np.ones((n,1)),
        context_names=("intercept",),population_sd=1.,
        fixed_effects=np.column_stack((np.ones(n),x[:,0])))
    actual=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,[trait],ids,
        probes=19,seed=72,block_size=8,probe_tile=7,context_tile=1,
        native=native,threads=prediction_threads())
    assert actual.diagnostics["projection_cancellation_fallbacks"]>0
    expected=dense_oracle(x,a,[trait],[np.arange(n)],probes=19,seed=72)[0,0]["stochastic"]
    np.testing.assert_allclose(actual.pairs[0].ldscores*n*n,expected,rtol=2e-8,atol=2e-7)
    # This target's genetic feature was removed by the fixed-effect model.
    # A raw-Gram subtraction leaves substantially larger roundoff here.
    assert np.max(np.abs(actual.pairs[0].ldscores[0]))<1e-20
