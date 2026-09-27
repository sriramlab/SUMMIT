import importlib.util
from pathlib import Path
import numpy as np


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_biology.py'
    spec=importlib.util.spec_from_file_location('biology',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_joint_baseline_response_against_full_gaussian_schur_complement():
    bio=module();rng=np.random.default_rng(68);q=3
    a=rng.normal(size=(6,6));sigma=a@a.T+np.eye(6)
    xx,yy,xy=sigma[:q,:q],sigma[q:,q:],sigma[:q,q:]
    sx=np.array([.2,-.4]);sy=np.array([-.1,.5]);s=np.array([[1.,.3],[.3,1.2]])
    options=dict(mean_x=sx,mean_y=sy,context_covariance=s,age=0,bmi=1)
    values=bio.quantities(xy,xx,yy,**options)
    c=np.eye(6);c[0,1:3]=sx;c[3,4:6]=sy;center=c@sigma@c.T
    response=[1,2,4,5];baseline=[0,3]
    conditional=center[np.ix_(response,response)]-center[np.ix_(response,baseline)]@np.linalg.solve(
        center[np.ix_(baseline,baseline)],center[np.ix_(baseline,response)])
    expected=np.trace(s@conditional[:2,2:])/np.sqrt(np.trace(s@conditional[:2,:2])*np.trace(s@conditional[2:,2:]))
    np.testing.assert_allclose(values['joint_baseline_response_rg'],expected,rtol=1e-13)
    np.testing.assert_allclose(values['trace_contribution'].sum(),values['orthogonal_rg'],rtol=1e-13)
    reverse=bio.quantities(xy.T,yy,xx,**dict(options,mean_x=sy,mean_y=sx))
    np.testing.assert_allclose(reverse['age_bmi_rg'],values['bmi_age_rg'],rtol=1e-13)
    np.testing.assert_allclose(reverse['age_bmi_asymmetry'],-values['age_bmi_asymmetry'],rtol=1e-13)


def test_bh_retains_undefined_and_counts_only_testable_entries():
    result=module().bh([.01,.04,np.nan,.2])
    np.testing.assert_allclose(result,[.03,.06,np.nan,.2],equal_nan=True)


def test_adding_followup_sets_preserves_the_original_pilot_fdr_family():
    bio=module()
    rows=[dict(trait_x='ldl_raw',trait_y=y,quantity='orthogonal_rg',p=p)
        for y,p in [('hba1c_raw',.01),('height_raw',.08),('glucose_log',.04),('triglycerides_log',1e-8)]]
    bio.assign_fdr(rows)
    np.testing.assert_allclose([r['fdr'] for r in rows],[.02,.08,.04,1e-8])
    assert [r['analysis_family'] for r in rows]==['pilot','pilot','glycaemic_followup','external_followup']
    assert bio.analysis_family('triglycerides_log','ldl_raw')=='external_followup'


def test_glycaemic_specificity_retains_covariance_across_pairs():
    bio=module();b=200
    common=np.linspace(-10,10,b);difference=np.linspace(-.01,.01,b)
    quantities=('baseline_rg','orthogonal_rg','joint_baseline_response_rg','response_rg','orthogonal_exposure_rg')
    left={k:np.array(.4) for k in quantities};right={k:np.array(.1) for k in quantities}
    il={k:common+difference for k in quantities};ir={k:common for k in quantities}
    lookup={frozenset(('ldl_raw','hba1c_raw')):(left,None,['age','bmi'],None,None,il),
            frozenset(('ldl_raw','glucose_log')):(right,None,['age','bmi'],None,None,ir)}
    rows=bio.glycaemic_specificity(lookup)
    assert len(rows)==5
    expected=np.sqrt((b-1)/b*np.sum(difference**2))
    np.testing.assert_allclose([r['estimate'] for r in rows],.3)
    np.testing.assert_allclose([r['paired_delta_se'] for r in rows],expected,rtol=1e-12)
    assert bio.glycaemic_specificity({})==[]
