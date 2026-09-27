#!/usr/bin/env python3
"""Summarize paired official-code checks without selecting a new SE rule."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

from report_pcgc import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    inputs={}
    def read(path):
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    rows=read(args.root/'paired_official/replicates.json')
    previous=read(args.previous/'confirmation/replicates.json')
    lookup={(r['scenario'],r['risk'],r['blocks'],r['seed']):r for r in previous}
    output=dict(inputs=inputs,paired=[],benchmarks=[],
                analytic=read(args.root/'null_identity.json'),
                report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for key in sorted({(r['scenario'],r['risk'],r['blocks']) for r in rows}):
        group=[r for r in rows if (r['scenario'],r['risk'],r['blocks'])==key]
        est=np.array([r['estimate'] for r in group]); truth=np.array([r['truth'] for r in group])
        old=[lookup[(*key,r['seed'])] for r in group]
        point_error=float(np.max(np.abs(est-[r['marginal_total'] for r in old])))
        se_error=float(np.max(np.abs(np.array([r['summit_se'] for r in group])-[r['marginal_total_standard_error'] for r in old])))
        if point_error>1e-7 or se_error>1e-7:
            raise RuntimeError('previous results did not reproduce')
        item=dict(zip(('scenario','risk','blocks'),key),replicates=len(group),
                  maximum_previous_point_difference=point_error,maximum_previous_se_difference=se_error)
        for method in ('summit','official'):
            se=np.array([r[method+'_se'] for r in group])
            item[method]=metrics(est,truth,se,len(group))
            failures=int(np.sum(np.abs(est-truth)>norm.isf(.025)*se))
            item[method]['undercoverage_binomial_p']=float(binomtest(failures,len(group),.05,alternative='greater').pvalue)
        item['median_official_over_summit_se']=float(np.median([r['official_se']/r['summit_se'] for r in group]))
        output['paired'].append(item)
        print(*key,'SUMMIT',item['summit']['rms_se_over_sd'],item['summit']['coverage'],
              'official',item['official']['rms_se_over_sd'],item['official']['coverage'],flush=True)
    for name in ('native_partition_benchmark','native_overlap_benchmark','real_bed_full_n'):
        data=read(args.root/(name+'.json'))
        medians={engine:float(np.median([r['seconds'] for r in data['repeats'] if r['engine']==engine]))
                 for engine in ('generalized','rank_one')}
        output['benchmarks'].append(dict(name=name,median_seconds=medians,
                                         speedup=medians['generalized']/medians['rank_one'],
                                         arguments=data['arguments'],
                                         maximum_relative_error=max(max(r['relative_errors']) for r in data['repeats']),
                                         peak_process_rss_bytes=data['process_peak_rss_bytes']))
    with args.out.open('x') as handle:
        json.dump(output,handle,indent=2,allow_nan=False); handle.write('\n')


if __name__=='__main__':
    main()
