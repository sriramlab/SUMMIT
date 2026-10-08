"""Collect completed conditional simulations without counting reruns as learners.

The design lists result directories and the scheduled replicate IDs for each
setting. Only portable summaries are opened: no genotype, phenotype or known
mean enters this collector. Missing scheduled learners remain in denominators.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from summit.epistasis.robust import load_robust_scores, robust_score_tests
from summit.prediction.artifacts import file_digest
from scripts.epistasis.public_conditional_validation import summarize
from scripts.epistasis.verify_conditional_batch import compare


METHOD = 'conditional_polygenic_mean_tangent_v1'


def _classify_completion(summary):
    """Missing scheduled comparisons have the same unknown status as primary fits."""
    incomplete = summary['scheduled']-summary['completed']
    summary['numerical_failures'] = 0 if not incomplete else None
    summary['unexecuted_or_incomplete'] = incomplete
    summary['failed_or_unexecuted'] = summary.pop('failures')
    summary['failure_classification'] = ('Missing results are unclassified; '
        'a missing file is not evidence of a numerical failure.')
    for field in ('comparisons', 'baselines'):
        for child in summary.get(field, {}).values():
            _classify_completion(child)


def _merge_diagnostics(previous, repeated):
    """Retain later completed comparisons without selecting on significance."""
    def check(left, right):
        for field in ('beta', 'se', 'p', 'conditional_interaction_truth',
                      'conditional_mean_bias', 'known_conditional_se'):
            if field in left and field in right:
                np.testing.assert_allclose(left[field], right[field], rtol=2e-6, atol=3e-8)
        for field in ('conditional_rejection_probability', 'conditional_coverage_probability'):
            for level in left.get(field, {}).keys() & right.get(field, {}).keys():
                np.testing.assert_allclose(left[field][level], right[field][level],
                    rtol=2e-6, atol=3e-8)
    check(previous, repeated)
    added = []
    for family in ('comparisons', 'baselines'):
        if family not in repeated:
            continue
        target = previous.setdefault(family, {})
        for name, values in repeated[family].items():
            if name not in target:
                target[name] = values.copy()
                added.append(f'{family}.{name}')
                continue
            check(target[name], values)
            for field, value in values.items():
                if field not in target[name]:
                    target[name][field] = value
                    added.append(f'{family}.{name}.{field}')
    for field, value in repeated.items():
        if field not in previous:
            previous[field] = value
            added.append(field)
    return added


def _preparation(root):
    # Collected earlier runs kept portable files beside results.json. Require
    # an unambiguous layout; neither layout needs the cohort-side reference.
    candidates = [p for p in (root/'prepared', root) if (p/'preparation.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('one unambiguous portable preparation directory required')
    return candidates[0]


def _bundle(root):
    root = Path(root).resolve()
    path = root/'results.json'
    if not path.is_file():
        path = root/'primary_diagnostics.json'
    result = json.loads(path.read_text())
    comparisons_complete=all(
        all(name in row.get(family,{}) for name in names)
        for row in result['records']
        for family,names in (('comparisons',('burden','oracle')),
            ('baselines',('local','finite_varying_mean'))))
    result['collected_stage'] = ('all_diagnostics' if comparisons_complete
        else 'public_primary_comparators_pending')
    prepared = _preparation(root)
    preparation = json.loads((prepared/'preparation.json').read_text())
    if preparation['method'] != METHOD:
        raise ValueError('conditional mean public preparation required')
    entries = preparation['summaries']
    if result['scheduled'] != len(entries) or len(result['records']) != len(entries):
        raise ValueError('completed public batch and diagnostic denominators differ')
    rows, hashes = [], {str(path): file_digest(path)}
    for item, row in zip(entries, result['records']):
        name = item['file']
        if Path(name).name != name:
            raise ValueError('portable summary must be local to the preparation')
        source = prepared/name
        summary = load_robust_scores(source)
        if (summary.metadata['preparation_identity'] != item['identity']
                or summary.metadata['method'] != METHOD or len(summary.trait_names) != 1):
            raise ValueError('public summary does not match the preparation')
        fit = robust_score_tests(summary)
        if len(fit['beta']) != 1 or row['failed'] or row['setting'] != result['setting']:
            raise ValueError('completed scalar primary record required')
        se = float(np.sqrt(fit['coefficient_covariance'][0][0]))
        np.testing.assert_allclose([row['beta'], row['se'], row['p']],
            [fit['beta'][0], se, fit['kernel_p']], rtol=2e-6, atol=3e-8)
        if se <= 0 or abs(row['beta']-fit['beta'][0])/se > 1e-4:
            raise ValueError('diagnostic coefficient differs from portable fit')
        if row['outside_confirmation_design'] != fit['diagnostics']['outside_confirmation_design']:
            raise ValueError('diagnostic support differs from portable fit')
        np.testing.assert_allclose(row['error'], row['beta']-row['conditional_interaction_truth'],
            rtol=2e-6, atol=3e-8)
        if bool(row['coverage']) != (abs(row['error']) <= norm.isf(.025)*se):
            raise ValueError('diagnostic coverage disagrees with the portable interval')
        if row['biological_null'] and row['conditional_interaction_truth'] != 0:
            raise ValueError('biological-null interaction truth must be zero')
        hashes[str(source)] = file_digest(source)
        rows.append((row, summary.metadata['identities'], summary.trait_names[0]))
    if len({v[0]['replicate'] for v in rows}) != len(rows):
        raise ValueError('duplicate learner within completed batch')
    return result, rows, hashes


def assess(design, root):
    """Require consistent repeated learners; never choose the smaller P value."""
    if design.get('schema_version') != 1 or design.get('method') != METHOD:
        raise ValueError('declare the conditional assessment schema and method')
    groups = design['groups']
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('assessment group IDs must be distinct')
    output, training_phases = [], {}
    for group in groups:
        scheduled = group['scheduled_replicates']
        if (not scheduled or len(set(scheduled)) != len(scheduled)
                or any(type(i) is not int or i < 0 for i in scheduled)
                or group['phase'] not in ('development', 'confirmation')):
            raise ValueError('declare distinct scheduled learners and their phase')
        chosen, identities, sources, hashes, repetitions = {}, {}, {}, {}, []
        collected_stages=[]
        learner_keys, training_keys = {}, {}
        reference_identity = None
        equivalence = None
        if 'reference_equivalence' in group:
            from scripts.epistasis.conditional_reference_equivalence import verify_equivalence
            equivalence = verify_equivalence(group['reference_equivalence'], root)
            hashes.update(equivalence['files'])
        paths = [Path(root)/p for p in group['completed_results']]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError('a completed result directory was listed twice')
        compared = set()
        for path in paths:
            result, rows, digests = _bundle(path)
            collected_stages.append(dict(path=str(path),stage=result['collected_stage']))
            if result['setting'] != group['setting']:
                raise ValueError('result setting differs from scheduled setting')
            run_design = path/'design.json'
            if run_design.exists():
                if json.loads(run_design.read_text())['phase'] != group['phase']:
                    raise ValueError('run phase differs from assessment phase')
            elif group['phase'] == 'confirmation':
                raise ValueError('confirmation assessment requires each original run design')
            hashes.update(digests)
            for row, identity, trait in rows:
                rep = row['replicate']
                if rep not in scheduled:
                    raise ValueError('unscheduled learner in completed results')
                key = tuple(identity[k] for k in
                    ('genotype_reference', 'direction', 'training_outcomes', 'confirmation_outcomes'))
                if reference_identity is None:
                    reference_identity = identity['genotype_reference']
                if equivalence is not None and identity['genotype_reference'] not in equivalence['reference_ids']:
                    raise ValueError('genotype reference is outside verified equivalence')
                if reference_identity != identity['genotype_reference'] and equivalence is None:
                    raise ValueError('genotype reference differs within assessment group')
                if key in identities and identities[key] != rep:
                    raise ValueError('same learner was assigned different replicate IDs')
                identities[key] = rep
                training_key = identity['training_outcomes']
                previous_phase = training_phases.setdefault(training_key, group['phase'])
                if previous_phase != group['phase']:
                    raise ValueError('development training outcomes were reused for confirmation')
                if training_key in training_keys and training_keys[training_key] != rep:
                    raise ValueError('training outcomes were reused as independent learners')
                training_keys[training_key] = rep
                if rep in chosen:
                    if learner_keys[rep] != (key, trait):
                        raise ValueError('replicate ID refers to different fitted learners')
                    pair = (sources[rep], path)
                    if pair not in compared:
                        compare(_preparation(pair[0]), _preparation(pair[1]))
                        compared.add(pair)
                    previous = chosen[rep]
                    for field in ('conditional_interaction_truth', 'conditional_mean_bias',
                                  'known_conditional_se', 'error'):
                        np.testing.assert_allclose(previous[field], row[field], rtol=2e-6, atol=3e-8)
                    if abs(previous['conditional_interaction_truth']-row['conditional_interaction_truth'])/row['se'] > 1e-4:
                        raise ValueError('repeated learner diagnostic truth differs')
                    supplemental = _merge_diagnostics(previous, row)
                    repetitions.append(dict(replicate=rep, trait=trait, retained=str(sources[rep]),
                        repeated=str(path), supplemental_fields=supplemental,
                        additional_learning_replicates=0))
                else:
                    chosen[rep], sources[rep] = row, path
                    learner_keys[rep] = (key, trait)
        records = [chosen[i] for i in scheduled if i in chosen]
        summary = summarize(records, len(scheduled))
        # The public validation programme prespecifies these comparisons. A
        # completed primary fit can precede its expensive comparator stage.
        # Keep those unfinished diagnostics visible rather than silently
        # dropping their scheduled denominators or calling the whole run done.
        for name in ('burden','oracle'):
            summary.setdefault('comparisons',{}).setdefault(name,summarize([],len(scheduled)))
        for name in ('local','finite_varying_mean'):
            summary.setdefault('baselines',{}).setdefault(name,dict(scheduled=len(scheduled),
                completed=0,failures=len(scheduled),unsupported=0,
                interpretation='Observed rejection; no completed comparison yet.'))
        # The historical summarize.failures field includes unexecuted learners.
        # Report the distinction explicitly; no missing result implies failure.
        _classify_completion(summary)
        output.append(dict(id=group['id'], setting=group['setting'], phase=group['phase'],
            scheduled_replicates=scheduled, completed_replicates=[r['replicate'] for r in records],
            summary=summary, repeated_executions=repetitions, files=hashes,
            collected_stages=collected_stages,records=records,
            reference_equivalence=equivalence))
    return dict(schema_version=1, method=METHOD, groups=output,
        interpretation='Observed and generating-law conditional rejection are separate; conditional probabilities are not additional learning replicates. Learner bootstrap intervals are approximate Monte Carlo intervals, not universal calibration guarantees.')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('design', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = assess(json.loads(a.design.read_text()), a.design.resolve().parent)
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
