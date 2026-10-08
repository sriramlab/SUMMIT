"""Verify overlapping learners in separate and batched public preparations.

This is a numerical comparison, never another statistical replicate. It requires
identical genotype-reference, direction and phenotype identities, and reads only
authenticated portable summaries. Batch-level preparation identities may differ.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summit.epistasis.robust import load_robust_scores, robust_score_tests
from summit.prediction.artifacts import file_digest


def compare(left, right):
    left, right = map(Path, (left, right))
    records = [json.loads((p/'preparation.json').read_text()) for p in (left, right)]
    if any(r['method'] != 'conditional_polygenic_mean_tangent_v1' for r in records):
        raise ValueError('conditional mean preparations required')
    summaries = []
    for root, record in zip((left, right), records):
        values = {}
        for j, item in enumerate(record['summaries']):
            path = root/item['file']
            if path.parent != root or path.name != item['file']:
                raise ValueError('summary filenames must be local to the preparation')
            summary = load_robust_scores(path)
            if (summary.metadata['preparation_identity'] != item['identity']
                    or len(summary.trait_names) != 1):
                raise ValueError('summary does not match its preparation')
            name = summary.trait_names[0]
            if name in values:
                raise ValueError('duplicate trait in preparation')
            values[name] = (j, summary, path)
        summaries.append(values)
    common = sorted(set(summaries[0]) & set(summaries[1]))
    if not common:
        raise ValueError('preparations have no common learners')
    checks = []
    for name in common:
        (j, a, ap), (k, b, bp) = [v[name] for v in summaries]
        for key in ('genotype_reference', 'direction', 'training_outcomes', 'confirmation_outcomes'):
            if a.metadata['identities'][key] != b.metadata['identities'][key]:
                raise ValueError(f'{name}: {key} differs; these are not duplicate learners')
        if a.feature_names != b.feature_names:
            raise ValueError(f'{name}: feature definitions differ')
        for key in ('method', 'trait_unit', 'nuisance_training_n', 'confirmation_n',
                    'fixed_rank', 'feature_rank', 'outside_confirmation_design'):
            if a.metadata[key] != b.metadata[key]:
                raise ValueError(f'{name}: {key} differs')
        if records[0]['kernel_names'] != records[1]['kernel_names']:
            raise ValueError('covariance component definitions differ')
        theta = [np.asarray(r['covariance_components'])[:, index]
                 for r, index in zip(records, (j, k))]
        np.testing.assert_allclose(*theta, rtol=2e-6, atol=3e-8)
        af, bf = map(robust_score_tests, (a, b))
        # Match the bounded public-workflow numerical tolerance. The additional
        # SE-unit comparison prevents an absolute tolerance hiding a meaningful
        # coefficient change for a precisely estimated phenotype.
        for key in ('beta', 'coefficient_covariance', 'kernel_p'):
            np.testing.assert_allclose(af[key], bf[key], rtol=2e-6, atol=3e-8)
        se = np.sqrt(np.diag(np.asarray(af['coefficient_covariance'])))
        if np.any(se <= 0):
            raise ValueError('positive coefficient variances required')
        delta = (np.asarray(af['beta'])-np.asarray(bf['beta']))/se
        if np.max(np.abs(delta)) > 1e-4:
            raise ValueError('batch coefficient difference exceeds 0.0001 SE')
        checks.append(dict(trait=name, coefficient_difference_se=delta.tolist(),
            covariance_max_absolute_difference=float(np.max(np.abs(
                np.asarray(af['coefficient_covariance'])-np.asarray(bf['coefficient_covariance'])))),
            p_absolute_difference=abs(float(af['kernel_p'])-float(bf['kernel_p'])),
            inputs=a.metadata['identities'],
            summaries=[dict(path=str(p.resolve()), sha256=file_digest(p)) for p in (ap, bp)]))
    return dict(verified=True, overlapping_learners=len(common), additional_learning_replicates=0,
        numerical_tolerance=dict(rtol=2e-6, atol=3e-8, maximum_coefficient_difference_se=1e-4),
        checks=checks)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
