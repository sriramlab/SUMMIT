"""Population metrics and supported pre-context ancestry adjustment."""
import numpy as np
import pytest

from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from summit.pcgc.ancestry import AncestryAdjustedOperator
from summit.pcgc.gxe import prepare_gxe_moments, fit_gxe, context_pairs
from summit.pcgc.gxe_io import make_gxe_artifact, write_gxe_artifact, load_gxe_artifact
from test_pcgc_io import fixture
from prediction_helpers import prediction_threads


@pytest.mark.parametrize('native',[False,True])
def test_adjustment_and_actual_annotation_metric(native):
    _,x,axis,risk,scale = fixture(91,73)
    rng = np.random.default_rng(992)
    cov = np.column_stack((np.linspace(-.2,1.,len(x)),rng.normal(size=len(x))))
    phi = np.column_stack((np.ones(len(x)),rng.uniform(-1,1,len(x))))
    a = np.column_stack((np.ones(x.shape[1]),np.linspace(.1,2,x.shape[1])))
    projection = np.eye(len(x))-cov@np.linalg.solve(cov.T@cov,cov.T)
    adjusted = projection@x
    op = AncestryAdjustedOperator(Operator(x),cov,threads=prediction_threads(),native=native)
    actual,diag = prepare_gxe_moments(op,a,risk,phi,liability_sd=1.,probes=113,seed=419,
                                    threads=prediction_threads(),native=native)
    expected,_ = prepare_gxe_moments(Operator(adjusted),a,risk,phi,liability_sd=1.,probes=113,seed=419,native=False)
    np.testing.assert_allclose(actual.rhs_rows,expected.rhs_rows,atol=2e-10)
    np.testing.assert_allclose(actual.ldscores,expected.ldscores,atol=2e-12)
    K,P = risk.population_prevalence,risk.sample_prevalence
    weight = np.where(risk.z>0,K/P,(1-K)/(1-P))/len(x)
    diagonal = adjusted**2@a/a.sum(0)
    metrics = np.array([phi.T@((weight*column)[:,None]*phi) for column in diagonal.T])
    np.testing.assert_allclose(actual.population_kernel_second_moment,metrics,atol=2e-14)
    result = fit_gxe(actual)
    assert result['population_genetic_variance'] == pytest.approx(np.sum(np.asarray(result['omega'])*metrics))
    assert diag['genotype_passes'] == 2


def test_legacy_artifact_retains_declared_unit_genotype_metric(tmp_path):
    from dataclasses import replace
    _,x,axis,risk,scale = fixture(71,55)
    phi = np.column_stack((np.ones(len(x)),np.linspace(-1,1,len(x))))
    m,d = prepare_gxe_moments(Operator(x),np.ones((x.shape[1],1)),risk,phi,liability_sd=1.,native=False)
    for name,moments in [('current',m),('legacy',replace(m,population_kernel_second_moment=None))]:
        artifact = make_gxe_artifact(moments,variant_axis=axis,annotation_names=['all'],context_names=['intercept','E'],
            sample_identity='a'*64,genotype_scale_identity=scale.identity,risk=risk,contexts=phi,liability_sd=1.,diagnostics=d)
        loaded = load_gxe_artifact(write_gxe_artifact(artifact,tmp_path/(name+'.npz')))
        assert loaded.manifest['schema_version'] == (2 if name == 'current' else 1)
        assert fit_gxe(loaded.moments)['population_genetic_variance_metric'] == (
            'annotation_specific_population_kernel_diagonal' if name == 'current' else 'unit_conditional_genotype_variance_assumption')


def test_ancestry_rejects_invalid_design():
    x = np.arange(70,dtype=float).reshape(10,7)
    with pytest.raises(ValueError,match='rank deficient'):
        AncestryAdjustedOperator(Operator(x),np.ones((10,2)),native=False)
    with pytest.raises(ValueError,match='shape'):
        AncestryAdjustedOperator(Operator(x),np.ones((9,1)),native=False)


def test_cli_fitted_risks_with_genotype_pcs_and_sampling(tmp_path):
    import pandas as pd
    from summit.entrypoint import main
    from test_pcgc_gxe_io import write_bed
    raw,x,axis,risk,scale = fixture(180,96)
    rng = np.random.default_rng(68193)
    e,pc = rng.uniform(-1,1,len(x)),rng.normal(size=len(x))
    write_bed(tmp_path/'study.bed',raw,axis)
    pd.DataFrame(dict(FID=['study']*len(x),IID=[f'i{i}' for i in range(len(x))],Y=(risk.z>0).astype(int),
        E=e,PC1=pc,PC1_E=pc*e)).to_csv(tmp_path/'people.tsv',sep='\t',index=False)
    pd.DataFrame(dict(SNP=axis.ids,A1=axis.counted,A2=axis.other,MEAN=scale.mean,INV_SD=scale.inverse_scale)).to_csv(tmp_path/'scale.tsv',sep='\t',index=False)
    assert main(['--binary-method','pcgc','--make-binary-sumstats',str(tmp_path/'people.tsv'),
        '--geno',str(tmp_path/'study.bed'),'--binary-scale',str(tmp_path/'scale.tsv'),'--binary-prevalence','.1',
        '--binary-context-columns','E','--binary-unit-liability','--binary-genotype-covariates','PC1',
        '--binary-covariates','PC1_E','--binary-sampling-partners','32','--binary-architecture-probes','8',
        '--nvecs','113','--memory-gib','1','--num-threads',str(prediction_threads()),'--out',str(tmp_path/'fit')]) == 0
    from summit.pcgc.split_io import load_split_artifact, output_paths
    artifact = load_split_artifact(*output_paths(tmp_path/'fit'))
    diagnostics = artifact.manifest['diagnostics']
    assert diagnostics['genotype_ancestry_adjustment']['rank'] == 1
    assert diagnostics['sampling_inference']['nuisance'] == 'same_study_probit_observed_information'
    assert artifact.moments.sampling_moments is not None
