"""Check the large-N research oracle against independent person kernels."""
import numpy as np
import pytest
from scripts.pcgc.calibrate_cross_scaling import make_design, exact_equations, fit_responses
from test_pcgc_cross_gxe import dense_oracle


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["qq", "bq", "bb"])
@pytest.mark.parametrize("overlap", [0., .5, 1.])
def test_scaling_variant_gram_oracle(kind, overlap, native):
    from prediction_helpers import prediction_threads
    design = make_design(24, 24, kind=kind, overlap=overlap, annotations=2,
                         heterogeneous=True, native=native, threads=prediction_threads())
    models, truth = exact_equations(design, [4, 6])
    total = max(max(r) for r in design["rows"])+1
    x = np.empty((total, 24))
    for rows, base in zip(design["rows"], design["bases"]):
        x[rows] = base
    oracle = dense_oracle(x, design["ann"], design["traits"], design["rows"])
    rng = np.random.default_rng(98)
    z = [rng.normal(size=(24, 3)) for _ in range(2)]
    z = [(np.eye(24)-trait.fixed_basis@trait.fixed_basis.T)@y
         for trait, y in zip(design["traits"], z)]
    fitted = fit_responses(design, models, z)
    for blocks in [4, 6]:
        full, deleted = [], []
        for model in models:
            left, right = model["left"], model["right"]
            value = oracle[left, right]
            p = len(model["layout"])
            source = value["source"] / np.repeat(design["ann"].sum(0), p)[:, None, None]
            work = model["blocks"][blocks]
            def solve(keep):
                test = np.einsum("ja,jpuv->apuv", design["ann"][keep], value["tests"][keep]).reshape(2*p, 24, 24)
                normal = np.einsum("cuv,duv->cd", test, source)
                rhs = np.einsum("ur,cuv,vr->cr", z[left], test, z[right])
                return normal, np.linalg.solve(normal, rhs).T
            normal, point = solve(np.ones(24, dtype=bool))
            np.testing.assert_allclose(normal, work["H"], rtol=2e-11, atol=2e-11)
            full.append(point)
            deleted_pair = []
            for first, last in zip(work["boundaries"][:-1], work["boundaries"][1:]):
                keep = np.ones(24, dtype=bool)
                keep[first:last] = False
                deleted_pair.append(solve(keep)[1])
            deleted.append(deleted_pair)
        np.testing.assert_allclose(fitted[blocks][0], np.concatenate(full, axis=1), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(fitted[blocks][1], np.concatenate(deleted, axis=2), rtol=2e-10, atol=2e-10)



@pytest.mark.parametrize("fitted", [False, True])
def test_retrospective_variant_oracle_matches_person_kernels(fitted):
    from scripts.pcgc.calibrate_cross_ascertainment import one
    from summit.ldscore.matrix_products import MatrixProducts
    from prediction_helpers import prediction_threads
    dense, truth = one(87211, 64, 16, fitted=fitted, rare=False)
    fast, expected = one(87211, 64, 16, fitted=fitted, rare=False, blocks=[4,8],
                         products=MatrixProducts(native=True,threads=prediction_threads()))
    np.testing.assert_array_equal(truth, expected)
    for point, _ in fast.values():
        np.testing.assert_allclose(point[0], dense, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("kind", ["qq", "bq", "bb"])
@pytest.mark.parametrize("overlap", [0., .5, 1.])
def test_randomized_variant_oracle_matches_production(kind, overlap):
    from scripts.pcgc.calibrate_cross_reference import random_equations, joint_covariance
    from summit.pcgc.cross_gxe import prepare_cross_gxe
    from summit.pcgc.cross_gxe_fit import equations, fit_cross_gxe
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
    design=make_design(32,24,kind=kind,overlap=overlap)
    exact,truth=exact_equations(design,[4],retain_grams=True)
    models=random_equations(design,exact,257,389)
    total=max(max(r) for r in design["rows"])+1
    x=np.empty((total,24))
    for rows,base in zip(design["rows"],design["bases"]):
        x[rows]=base
    ids=tuple(("s",str(i)) for i in range(total))
    actual=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),design["ann"],design["traits"],ids,
                             probes=257,seed=389,native=False)
    for model,pair in zip(models,actual.pairs):
        mass=np.repeat(design["ann"].sum(0),pair.p)
        np.testing.assert_allclose(model["blocks"][4]["H"],equations(actual,pair)[0]*mass[:,None],
                                   rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(model["draw"],pair.probe_grams*mass[None,:,None],
                                   rtol=2e-11,atol=2e-11)
    z=[((np.eye(32)-t.fixed_basis@t.fixed_basis.T)@t.response)[:,None] for t in design["traits"]]
    point,loo=fit_responses(design,models,z)[4]
    snp,mc=joint_covariance(models,point[0],loo[:,0],4)
    fitted=fit_cross_gxe(actual,block_ids=np.arange(24)*4//24)
    np.testing.assert_allclose(fitted["components"],point[0],rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(fitted["covariance"],snp+mc,rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize("correlation", [None,.9])
@pytest.mark.parametrize("heterogeneity", [False,True])
def test_overlapping_unequal_annotation_mass_matches_generating_kernel(correlation,heterogeneity):
    design=make_design(24,25,kind="qq",overlap=.5,annotations=2,annotation_mixing=.8,
                       effect_correlation=correlation,shared_context=True,residual_correlation=.8,
                       annotation_heterogeneity=heterogeneity)
    models,truth=exact_equations(design,[5])
    total=max(max(r) for r in design["rows"])+1
    x=np.empty((total,25))
    for rows,base in zip(design["rows"],design["bases"]): x[rows]=base
    oracle=dense_oracle(x,design["ann"],design["traits"],design["rows"])
    estimate=[]
    roots=design["variant_roots"]
    if roots is None:
        roots=np.tile(design["omega"]/5,(25,1,1))
    covariance=roots@roots.transpose(0,2,1)
    for t in range(2):
        raw=design["bases"][t][:,None,:]*design["phi"][t][:,:,None]
        kernel=np.einsum("iqj,jqr,krj->ik",raw,covariance[:,2*t:2*t+2,2*t:2*t+2],raw)
        np.testing.assert_allclose(np.diag(kernel)+design["residual"][t],design["sd"][t]**2,
                                   rtol=2e-12,atol=2e-12)
    for model,matrix in zip(models,design["truth_matrices"]):
        left,right=model["left"],model["right"]
        genetic=np.einsum("iqj,jqr,krj->ik",np.stack(design["features"][left],axis=1),
            covariance[:,2*left:2*left+2,2*right:2*right+2],np.stack(design["features"][right],axis=1))
        tests=oracle[left,right]["tests"]
        rhs=np.einsum("ja,jpuv,uv->ap",design["ann"],tests,genetic).ravel()
        estimate.extend(np.linalg.solve(model["blocks"][5]["H"],rhs))
    np.testing.assert_allclose(estimate,truth,rtol=2e-10,atol=2e-10)
