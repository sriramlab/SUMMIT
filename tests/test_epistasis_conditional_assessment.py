"""Authenticate statistical denominators and prevent learner pseudoreplication."""
import copy
import json

import pytest
import numpy as np

from summit.epistasis.conditional import conditional_mean_summary
from summit.epistasis.robust import robust_score_tests, write_robust_scores
from scripts.epistasis.conditional_assessment import METHOD, assess


def test_simulation_schedule_preserves_ids_and_rejects_ambiguous_batches():
    from scripts.epistasis.replicate_schedule import replicate_ids, simulation_replicates, replicate_columns
    assert simulation_replicates({'replicates':2})==[0,1]
    assert replicate_columns(simulation_replicates({'replicates':2,'replicate_ids':[8,9]}))==['rep008','rep009']
    for count,start in ((0,0),(2,-1),(2,99),(True,0),(1,1.5)):
        with pytest.raises(ValueError,match='schedule'):
            replicate_ids(count,start)
    for values in ([1,1],[2,1],[True,2],[1],[-1,1],[99,100]):
        with pytest.raises(ValueError,match='replicate IDs'):
            simulation_replicates({'replicates':2,'replicate_ids':values})


def _bundle(root, replicates, *, changed_outcome=False, flat=False, comparisons=True):
    prepared = root if flat else root/'prepared'
    prepared.mkdir(parents=True)
    entries, records = [], []
    for i in replicates:
        identities = {k:format(j+i+1, '064x') for j,k in enumerate(
            ('direction','training_outcomes','confirmation_outcomes','null_fit'))}
        identities['genotype_reference'] = 'a'*64
        if changed_outcome:
            identities['training_outcomes'] = 'b'*64
        summary = conditional_mean_summary([.01*(i+1)], [[.04]], [[10.]],
            feature_names=['target_by_frozen_score'], trait_name=f'rep{i:03}', identities=identities,
            trait_unit='test units', diagnostics=dict(nuisance_training_n=100,
                confirmation_n=200, fixed_rank=4, feature_rank=1, outside_confirmation_design=[]))
        name=f'rep{i:03}.robust-score.npz'
        write_robust_scores(summary, prepared/name)
        entries.append(dict(file=name, identity=summary.metadata['preparation_identity']))
        fit=robust_score_tests(summary)
        records.append(dict(setting='null', replicate=i, failed=False, biological_null=True,
            beta=float(fit['beta'][0]), se=.2, p=fit['kernel_p'], error=float(fit['beta'][0]),
            coverage=True, conditional_interaction_truth=0., conditional_mean_bias=0.,
            known_conditional_se=.2, conditional_rejection_probability={'0.05':.05,'0.005':.005},
            outside_confirmation_design=[]))
        if comparisons:
            records[-1]['comparisons'] = {name: records[-1].copy() for name in ('burden','oracle')}
            records[-1]['baselines'] = {name: dict(p=fit['kernel_p'], outside_confirmation_design=[])
                for name in ('local','finite_varying_mean')}
    (prepared/'preparation.json').write_text(json.dumps(dict(method=METHOD,summaries=entries,
        kernel_names=['noise'],covariance_components=[[1.]*len(replicates)])))
    (root/'results.json').write_text(json.dumps(dict(setting='null',scheduled=len(replicates),records=records)))
    (root/'design.json').write_text(json.dumps(dict(phase='development')))


def test_conditional_assessment_authenticates_counts_and_overlaps(tmp_path):
    _bundle(tmp_path/'pair', [0,1], flat=True, comparisons=False)
    _bundle(tmp_path/'batch', [0,1,2])
    design=dict(schema_version=1,method=METHOD,groups=[dict(id='null_dev', setting='null',
        phase='development',scheduled_replicates=[0,1,2,3],completed_results=['pair','batch'])])
    result=assess(design,tmp_path)['groups'][0]
    assert result['completed_replicates']==[0,1,2]
    assert len(result['repeated_executions'])==2
    summary=result['summary']
    assert summary['scheduled']==4 and summary['completed']==3
    assert summary['unexecuted_or_incomplete']==1 and summary['numerical_failures'] is None
    assert summary['0.05']['scheduled_denominator']==4
    assert summary['0.05']['successful_denominator']==3
    for field in ('comparisons','baselines'):
        for child in summary[field].values():
            assert child['unexecuted_or_incomplete']==1 and child['numerical_failures'] is None
            assert 'failures' not in child
    assert 'comparisons.oracle' in result['repeated_executions'][0]['supplemental_fields']
    # Input order must not discard completed comparisons or alter statistics.
    reversed_design=copy.deepcopy(design)
    reversed_design['groups'][0]['completed_results'].reverse()
    assert assess(reversed_design,tmp_path)['groups'][0]['summary']==summary
    _bundle(tmp_path/'different_comparison',[0])
    p=tmp_path/'different_comparison/results.json';v=json.loads(p.read_text())
    v['records'][0]['comparisons']['oracle']['beta'] += .1
    p.write_text(json.dumps(v))
    changed=copy.deepcopy(design);changed['groups'][0]['completed_results'].append('different_comparison')
    with pytest.raises(AssertionError):
        assess(changed,tmp_path)
    _bundle(tmp_path/'different', [0], changed_outcome=True)
    changed=copy.deepcopy(design);changed['groups'][0]['completed_results'].append('different')
    with pytest.raises(ValueError,match='different fitted learners'):
        assess(changed,tmp_path)
    changed=copy.deepcopy(design);changed['groups'][0]['phase']='confirmation'
    with pytest.raises(ValueError,match='phase differs'):
        assess(changed,tmp_path)
    _bundle(tmp_path/'reused', [0,1], changed_outcome=True)
    changed=copy.deepcopy(design);changed['groups'][0]['completed_results']=['reused']
    with pytest.raises(ValueError,match='training outcomes were reused'):
        assess(changed,tmp_path)
    path=tmp_path/'batch/results.json';data=json.loads(path.read_text())
    data['records'][0]['beta'] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        assess(design,tmp_path)


def test_assessment_rejects_training_reused_across_validation_phases(tmp_path):
    _bundle(tmp_path/'development',[8,9])
    _bundle(tmp_path/'confirmation',[8,9])
    (tmp_path/'confirmation/design.json').write_text(json.dumps(dict(phase='confirmation')))
    design=dict(schema_version=1,method=METHOD,groups=[
        dict(id=phase,setting='null',phase=phase,scheduled_replicates=[8,9],completed_results=[phase])
        for phase in ('development','confirmation')])
    with pytest.raises(ValueError,match='development training outcomes were reused'):
        assess(design,tmp_path)


def test_primary_completion_retains_pending_comparator_denominators(tmp_path):
    _bundle(tmp_path/'early',[0,1,2],comparisons=False)
    (tmp_path/'early/results.json').rename(tmp_path/'early/primary_diagnostics.json')
    _bundle(tmp_path/'pair',[0,1])
    design=dict(schema_version=1,method=METHOD,groups=[dict(id='null',setting='null',
        phase='development',scheduled_replicates=[0,1,2,3],completed_results=['early','pair'])])
    group=assess(design,tmp_path)['groups'][0]
    assert group['summary']['completed']==3
    assert group['collected_stages'][0]['stage']=='public_primary_comparators_pending'
    for family in ('comparisons','baselines'):
        for summary in group['summary'][family].values():
            assert summary['completed']==2 and summary['unexecuted_or_incomplete']==2
            assert summary['numerical_failures'] is None
    _bundle(tmp_path/'complete',[0,1,2])
    design['groups'][0]['completed_results'].append('complete')
    completed=assess(design,tmp_path)['groups'][0]
    assert completed['completed_replicates']==[0,1,2]
    assert len(completed['repeated_executions'])==5
    for family in ('comparisons','baselines'):
        for summary in completed['summary'][family].values():
            assert summary['completed']==3 and summary['unexecuted_or_incomplete']==1
    design['groups'][0]['completed_results']=['early']
    summary=assess(design,tmp_path)['groups'][0]['summary']
    assert all(v['completed']==0 and v['scheduled']==4
        for family in ('comparisons','baselines') for v in summary[family].values())


def test_conditional_error_moments_match_independent_centering_reference(tmp_path):
    from scripts.epistasis.public_conditional_validation import summarize
    _bundle(tmp_path/'moments',[0,1,2])
    rows=json.loads((tmp_path/'moments/results.json').read_text())['records']
    mean=np.array([-.4,.1,.8]);sigma=np.array([.1,.7,.3])
    for j,row in enumerate(rows):
        row.update(conditional_mean_bias=float(mean[j]),known_conditional_se=float(sigma[j]))
        row.pop('comparisons');row.pop('baselines')
    actual=summarize(rows,3)['conditional_error_moments']
    centering=np.eye(3)-np.ones((3,3))/3
    second_moment=np.diag(sigma*sigma)+np.outer(mean,mean)
    np.testing.assert_allclose(actual['expected_sample_variance'],
        np.trace(centering@second_moment@centering)/2,rtol=1e-14)
    np.testing.assert_allclose(actual['expected_mean_square'],np.trace(second_moment)/3,rtol=1e-14)
    np.testing.assert_allclose(actual['variance_of_mean'],np.ones(3)@np.diag(sigma*sigma)@np.ones(3)/9,rtol=1e-14)
    one=summarize(rows[:1],1)['conditional_error_moments']
    assert one['expected_sample_variance'] is None
    np.testing.assert_allclose(one['expected_mean_square'],sigma[0]**2+mean[0]**2)
