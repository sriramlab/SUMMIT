import importlib.util
from pathlib import Path
import numpy as np
from summit.context.cross_trait_fit import cross_trait_derived


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_context_profiles.py'
    spec=importlib.util.spec_from_file_location('context_profiles',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_context_curves_centering_and_trait_orientation():
    rng=np.random.default_rng(103);a=rng.normal(size=(6,6));joint=a@a.T+np.eye(6)
    xx,yy,xy=joint[:3,:3],joint[3:,3:],joint[:3,3:]
    mx=np.array([.2,-.1]);my=np.array([-.1,.3]);offsets=np.array([-1.,0.,1.])
    value=module().profiles(np.stack([xx,yy,xy]),mean_x=mx,mean_y=my,exposure=1,offsets=offsets)
    centered=cross_trait_derived(xy,xx,yy,mean_x=mx,mean_y=my,context_covariance=np.eye(2))
    np.testing.assert_allclose(value['context_rg'][1],centered['centered_baseline_rg'],rtol=1e-14)
    reverse=module().profiles(np.stack([yy,xx,xy.T]),mean_x=my,mean_y=mx,exposure=1,offsets=offsets)
    np.testing.assert_allclose(reverse['context_rg'],value['context_rg'],rtol=1e-14)
    assert np.max(abs(value['context_rg']))<=1


def test_invalid_profile_variance_remains_undefined():
    primitive=np.array([np.diag([-1.,0.]),np.eye(2),np.eye(2)])
    result=module().profiles(primitive,mean_x=[0.],mean_y=[0.],exposure=1,offsets=np.array([-1.,0.,1.]))
    assert np.isnan(result['context_rg']).all()
    assert np.isnan(result['high_minus_low_rg'])
