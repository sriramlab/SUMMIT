"""Check the large-N research oracle against independent person kernels."""
import numpy as np
import pytest
from scripts.pcgc.calibrate_cross_scaling import make_design, exact_equations, fit_responses
from test_pcgc_cross_gxe import dense_oracle


@pytest.mark.parametrize("kind", ["qq", "bq", "bb"])
@pytest.mark.parametrize("overlap", [0., .5, 1.])
def test_scaling_variant_gram_oracle(kind, overlap):
    design = make_design(24, 24, kind=kind, overlap=overlap, annotations=2,
                         heterogeneous=True)
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
