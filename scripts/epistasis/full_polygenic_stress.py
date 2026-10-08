"""Matched fixed-architecture stress inputs using an existing intact-row reference.

Fixed genetic means and effect sizes precede fresh errors. The withheld-window
stress reads only bounded local genotypes; all other settings reuse saved means.
Every learned direction and covariance must be re-estimated from that
replicate's training outcome.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.epistasis.full_matched import write_json
from scripts.epistasis.replicate_schedule import replicate_ids, replicate_columns


SETTINGS={
    'fixed_dense':('dense','gaussian'),
    'fixed_sparse':('sparse','gaussian'),
    'fixed_structure':('structure','gaussian'),
    'fixed_heavy':('structure','t5_unit_variance'),
    'fixed_aligned':('aligned','gaussian'),
    'fixed_sparse_interaction':('sparse_interaction','gaussian'),
    'fixed_structure_mixed':('structure_mixed','gaussian'),
    'fixed_local_withheld':('local_withheld','gaussian'),
    'fixed_local_withheld_mixed':('local_withheld_mixed','gaussian'),
}


def run(a):
    ids = replicate_ids(a.replicates, getattr(a, 'replicate_start', 0))
    start=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False);a.out.chmod(0o700)
    selected=a.settings.split(',')
    if (not selected or len(set(selected))!=len(selected) or set(selected)-set(SETTINGS)
            or not 1<=a.replicates<=100 or not np.isfinite(a.signal_multiplier) or a.signal_multiplier<=0):
        raise ValueError('prespecified settings, 1..100 replicates and positive fixed signal strength required')
    metadata=json.loads((a.reference/'reference.json').read_text())
    withheld=any(SETTINGS[s][0].startswith('local_withheld') for s in selected)
    teacher=json.loads((a.signal_reference/'reference.json').read_text())
    with np.load(a.reference/'reference.npz',allow_pickle=False) as archive:
        rows=archive['rows'];x=archive['target'];i0=archive['training_index'];i1=archive['confirmation_index']
    if withheld:
        from scripts.epistasis.full_matched import load_experiment_inputs
        teacher,arrays,_=load_experiment_inputs(a.signal_reference,local_stress=True,
            threads=getattr(a,'num_threads',1))
        take=np.searchsorted(arrays['rows'],rows)
        if not np.array_equal(arrays['rows'][take],rows):raise ValueError('fixed architecture rows changed')
        means=arrays['means'][take];signals=arrays['signals'][take]
    else:
        with np.load(a.signal_reference/'reference.npz',allow_pickle=False) as archive:
            take=np.searchsorted(archive['rows'],rows)
            if not np.array_equal(archive['rows'][take],rows):raise ValueError('fixed architecture rows changed')
            means=archive['means'][take];signals=archive['signals'][take]
    sample_source=pd.read_csv(a.signal_reference/'samples.tsv',sep=r'\s+',dtype=str)
    samples=sample_source.iloc[take].reset_index(drop=True)
    if len(samples)!=len(rows):raise ValueError('reference sample axis differs')
    recipe_path=a.training.resolve();recipe=json.loads(recipe_path.read_text())
    if metadata['source']!=teacher['source_identity']:raise ValueError('genotype references differ')
    if withheld:
        excluded=set(teacher['prediction_exclusion']['variants'])
        if excluded.intersection(recipe.get('local_variants',[])+recipe.get('dominance_variants',[])):
            raise ValueError('withheld window overlaps declared local nuisance terms')
        direction=(recipe_path.parent/recipe['interaction_variants']).read_text().splitlines()
        if excluded.intersection(direction):raise ValueError('withheld window overlaps interaction background')
        original=(recipe_path.parent/recipe['variants']).read_text().splitlines()
        retained=[v for v in original if v not in excluded]
        if not retained:raise ValueError('withheld window removes the entire fitted variant axis')
        (a.out/'training_variants.txt').write_text('\n'.join(retained)+'\n')
        write_json(a.out/'prediction_exclusion.json',dict(teacher['prediction_exclusion'],
            original_fitted_marker_count=len(original),fitted_marker_count=len(retained),
            scope='same genotype-only window excluded from direction-training additive prediction and conditional covariance; causal identity is diagnostic only'))
    definitions={}
    for setting in selected:
        original,errors=SETTINGS[setting];j=teacher['settings'].index(original)
        definitions[setting]=dict(sampling_law='fixed_architecture',error_law=errors,
            biological_null=teacher['definitions'][original]['biological_null'],
            architecture=original,noise_variances=[.4,.6],
            direction=teacher['definitions'][original]['direction'],
            reference_signal_variance=teacher['definitions'][original]['reference_signal_variance']*a.signal_multiplier**2)
        if original.startswith('local_withheld'):
            definitions[setting]['diagnostic_local_cause']=teacher['definitions'][original]
    write_json(a.out/'simulation.json',dict(reference=str(a.reference.resolve()),
        signal_reference=str(a.signal_reference.resolve()),settings=selected,definitions=definitions,
        seed=a.seed,replicates=a.replicates,replicate_ids=ids,signal_multiplier=a.signal_multiplier,
        conditioning='fixed intact distinct participants and marker coefficients; independent new errors in both cohorts',
        strength='original full-reference architecture coefficients; only the prespecified common interaction multiplier changes signal',
        inference_scope='stress of the same random-polygenic estimator; fixed-operator signal response and conditional noise are diagnostic targets',
        status='inputs only; no qualification'))
    columns=replicate_columns(ids);variance=.4+.6*x*x;moments=[]
    for setting in selected:
        original,law=SETTINGS[setting];j=teacher['settings'].index(original)
        signal=signals[:,j]*a.signal_multiplier
        mean=means[:,j]-signals[:,j]+signal
        root=a.out/setting;root.mkdir();outcomes=samples.copy()
        for r,column in zip(ids,columns):
            rng=np.random.default_rng(np.random.SeedSequence([a.seed,list(SETTINGS).index(setting),r]))
            noise=rng.standard_t(5,len(rows))/np.sqrt(5/3) if law=='t5_unit_variance' else rng.normal(size=len(rows))
            outcomes[column]=mean+np.sqrt(variance)*noise
        outcomes.to_csv(root/'phenotypes.tsv',sep='\t',index=False)
        np.savez(root/'diagnostic_truth.npz',rows=rows,mean=mean,signal=signal,variance=variance)
        train=json.loads(json.dumps(recipe));train.pop('phenotype',None)
        train['phenotypes']=dict(file='phenotypes.tsv',columns=columns,unit='fixed reference units')
        for name in ('samples','variants','interaction_variants'):train[name]=str((recipe_path.parent/train[name]).resolve())
        if original.startswith('local_withheld'):
            train['variants']=str((a.out/'training_variants.txt').resolve())
        train['genotypes']['geno']=str((recipe_path.parent/train['genotypes']['geno']).resolve())
        train['covariates']['file']=str((recipe_path.parent/train['covariates']['file']).resolve())
        write_json(root/'train.json',train)
        write_json(root/'prepare.json',dict(kind='summit.epistasis.prepare',schema_version=1,
            training='train.json',samples=str(Path(metadata['confirmation_samples']).resolve()),phenotypes=train['phenotypes'],
            directions=[dict(phenotype=column,direction=f'trained/direction.{r}.json') for r,column in enumerate(columns)],
            inference=dict(method='conditional_polygenic_mean')))
        moments.append(dict(setting=setting,realized_training_signal_variance=float(np.var(signal[i0])),
            realized_confirmation_signal_variance=float(np.var(signal[i1])),mean_variance=float(np.var(mean))))
    write_json(a.out/'resources.json',dict(seconds=time.perf_counter()-start,genotype_traversals=0,
        genotype_access=('bounded target+[200,400] kb calls on frozen reference rows' if withheld else 'none'),
        traversal_scope='complete marker traversals; bounded local reads are reported separately',moments=moments))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('reference','signal-reference','training','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--replicates',type=int,default=12)
    p.add_argument('--replicate-start',type=int,default=0,
        help='First stable learner ID; split batches share the same seed and reference')
    p.add_argument('--signal-multiplier',type=float,default=np.sqrt(.2))
    p.add_argument('--settings',default=','.join(k for k in SETTINGS if not k.startswith('fixed_local_withheld')))
    p.add_argument('--num-threads',type=int,default=1)
    run(p.parse_args())


if __name__=='__main__':main()
