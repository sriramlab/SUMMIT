"""Validate exported per-probe delta uncertainty against independent reference draws."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.epistasis.validate import run_setting
from summit.epistasis.summary import fit_epistasis
from summit.prediction.spec import VariantAxis
from summit.epistasis.cli import _jsonable


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(9391);n,m=192,64
    raw=rng.binomial(2,.3,(n,m));axis=VariantAxis(tuple(f'v{i}' for i in range(m)),('1',)*m,tuple(range(1,m+1)),('A',)*m,('G',)*m)
    experiment=run_setting('additive_null_p30',raw,axis,rng,32,'numpy',return_experiment=True)
    study=experiment['study'];summary=experiment['summary'];rows=[];bounds=[]
    exact=np.array([fit_epistasis(summary,i)['coefficients'][1] for i in range(32)])
    for count in (64,256,1024):
        coefs=[];predicted=[];bias=[]
        for seed in range(24):
            reference=study.reference(nvecs=count,seed=293+seed)
            candidate=replace(summary,matrix=reference.matrix,probe_matrices=reference.probe_matrices,metadata=dict(reference.metadata))
            fits=[fit_epistasis(candidate,i) for i in range(32)]
            coefs.append([f['coefficients'][1] for f in fits])
            predicted.append(fits[0]['probe_uncertainty']['standard_errors'][1])
            bias.append(fits[0]['probe_uncertainty']['leading_order_bias'][1])
            bounds.append(dict(probes=count,seed=seed,**fits[0]['probe_uncertainty']['uniform_precision_bound']))
        coefs=np.array(coefs)
        rows.append(dict(probes=count,reference_draws=24,phenotypes=32,
            empirical_probe_sd_fixed_y=float(np.std(coefs[:,0],ddof=1)),mean_delta_probe_se=float(np.mean(predicted)),
            probe_bias_fixed_y=float(np.mean(coefs[:,0])-exact[0]),mean_leading_order_bias=float(np.mean(bias)),
            phenotype_sd_fixed_probes=float(np.std(coefs[0],ddof=1)),phenotype_sd_exact=float(np.std(exact,ddof=1))))
    pd.DataFrame(rows).to_csv(a.out/'summary.csv',index=False)
    (a.out/'bounds.json').write_text(json.dumps(_jsonable(bounds),indent=2,allow_nan=False)+'\n')
    (a.out/'design.json').write_text(json.dumps(dict(seed=9391,n=n,m=m,
        contract='Genotypes fixed; 32 Gaussian phenotypes regenerated once; 24 independent reference draws per count. Analytic phenotype moments fixed and exact. Delta SE not used to alter FAME P values; uniform bound is phenotype independent.'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
