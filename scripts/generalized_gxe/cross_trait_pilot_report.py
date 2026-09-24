"""Publish the requested pilot contrasts and age/BMI block with no refitting."""
from pathlib import Path
import argparse
import csv
import itertools
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
from summit.context.reference_zpass_cli import file_sha256
from cross_trait_pilot_fit import TRAITS
from cross_trait_refit import write_table


def report(fits,output):
    complete=json.loads((fits/'COMPLETE.json').read_text())
    for name,digest in complete['tables'].items():
        if file_sha256(fits/name)!=digest:raise ValueError('pilot table checksum differs')
    def read(name):
        with (fits/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
    rows=read('pilot_estimates.tsv');shifts=read('pilot_maximum_mode_shifts.tsv')
    key=lambda r:(r['trait_x'],r['trait_y'],r['quantity'],r['entry'])
    default={key(r):r for r in rows if r['mode']=='factorized'}
    shifts={key(r):r for r in shifts}
    comparisons=[];blocks=[]
    for x,y in itertools.combinations(TRAITS,2):
        category='cardiometabolic' if x in TRAITS[:6] and y in TRAITS[:6] else 'control_involving'
        baseline=default[x,y,'baseline_rg','0']
        cohort={name:baseline.get(name,'') for name in ('n_x','n_y','n_overlap')}
        row=dict(trait_x=x,trait_y=y,**cohort,pair_category=category,exploratory=True)
        for quantity in ('baseline_rg','orthogonal_rg','orthogonal_minus_baseline_rg'):
            k=(x,y,quantity,'0');entry=default[k]
            for field in ('estimate','jackknife_se','lower_95','upper_95'):
                row[quantity+'_'+field]=entry[field]
            row[quantity+'_maximum_mode_shift_se']=shifts[k]['maximum_mode_shift_se']
            row[quantity+'_undefined_shift_modes']=shifts[k]['undefined_shift_modes']
            if quantity!='orthogonal_minus_baseline_rg':
                value=float(entry['estimate'])
                row[quantity+'_admissible']=math.isfinite(value) and abs(value)<=1
        comparisons.append(row)
        selected=[r for r in default.values() if (r['trait_x'],r['trait_y'])==(x,y)
            and r['quantity']=='h_xy' and r['exposure_x'] in ('age','bmi_raw')
            and r['exposure_y'] in ('age','bmi_raw')]
        if len(selected)!=4:raise ValueError('four named age/BMI cross entries are required')
        for entry in selected:
            k=key(entry)
            blocks.append(dict(trait_x=x,trait_y=y,**cohort,pair_category=category,exploratory=True,
                **{f:entry[f] for f in ('exposure_x','exposure_y','estimate','jackknife_se','lower_95','upper_95')},
                maximum_mode_shift_se=shifts[k]['maximum_mode_shift_se'],
                undefined_shift_modes=shifts[k]['undefined_shift_modes']))
    write_table(output/'baseline_vs_orthogonal_response.tsv',comparisons)
    write_table(output/'age_bmi_cross_exposure_covariance.tsv',blocks)
    result=dict(pilot_completion_sha256=file_sha256(fits/'COMPLETE.json'),script_sha256=file_sha256(__file__),
        input_tables=complete['tables'],default_mode='factorized',pairs=len(comparisons),age_bmi_entries=len(blocks),
        undefined_orthogonal_correlations=sum(not math.isfinite(float(r['orthogonal_rg_estimate'])) for r in comparisons),
        interpretation='exploratory; a nonsignificant control does not establish absence',
        interval_convention='paired target-SNP deletion; contrast uses its own paired jackknife SE',
        tables={p.name:file_sha256(p) for p in output.glob('*.tsv')})
    with (output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False);report(args.fits,args.output)


if __name__=='__main__':main()
