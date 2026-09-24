import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from summit.context.cross_trait_gram import orientation_matrix


def test_common_within_score_expansion_preserves_ordered_oracle():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_pilot_fit.py'
    spec=importlib.util.spec_from_file_location('pilot_fit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(911);q,b,h,m=5,7,13,32
    scores=rng.normal(size=(b,m,q));weights=rng.uniform(size=(b,m))
    rhs=np.einsum('bm,bmq,bmr->bqr',weights,scores,scores).reshape(b,q*q)
    gr=rng.normal(size=(b,q,q,h));gr=(gr+gr.swapaxes(1,2))/2;gr=gr.reshape(b,q*q,h)
    saved=orientation_matrix(q)
    study=SimpleNamespace(num_basis=q,block_ids=np.arange(b),block_masses=weights.sum(1)[:,None],
        block_genetic_rhs=(rhs@saved.T)[...,None],block_genetic_residual=saved@gr)
    result=module.ordered_within_record(study)
    np.testing.assert_allclose(result['block_rhs'][:,0].reshape(b,q*q),rhs,atol=1e-13)
    np.testing.assert_array_equal(result['block_genetic_residual'][:,0],gr)
