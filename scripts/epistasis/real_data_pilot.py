"""Execute a prespecified public training/preparation/fit workflow once.

Use a frozen checkout inside an external resource supervisor. This driver has
no automatic retries and preserves incomplete stages for explicit recovery.
"""
import argparse
import json
from pathlib import Path
import time

from summit.epistasis.cli import main as cli
from summit.prediction.artifacts import file_digest
from scripts.epistasis.full_matched import write_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-threads',type=int,default=1)
    parser.add_argument('--block-size',type=int,default=4096)
    parser.add_argument('--memory-gib',type=float,default=32)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--trained',type=Path,help='reuse a completed learner; preparation authenticates the matched outcomes and recipe')
    args=parser.parse_args();start=time.perf_counter()
    spec=json.loads(args.manifest.read_text())
    training=(args.manifest.parent/spec['training']).resolve()
    args.out.mkdir(exist_ok=args.resume)
    # The supplied preparation must point at this run's matched direction.
    trained=(args.trained if args.trained is not None else args.out/'trained').resolve()
    if len(spec['directions'])!=1 or (args.manifest.parent/spec['directions'][0]['direction']).resolve()!=trained/'direction.json':
        raise ValueError('pilot requires one prespecified trait and its matched output direction')
    controls=['--num-threads',str(args.num_threads),'--block-size',str(args.block_size),
        '--memory-gib',str(args.memory_gib),*(['--resume'] if args.resume else [])]
    timings={}
    stages=[] if args.trained is not None else [('train',['train-direction',str(training),'--out',str(trained)])]
    stages.append(('prepare',['prepare',str(args.manifest),'--out',str(args.out/'prepared')]))
    for name,command in stages:
        begin=time.perf_counter();cli(command+controls);timings[name]=time.perf_counter()-begin
        print(json.dumps(dict(event='stage_complete',stage=name,seconds=timings[name])),flush=True)
    preparation=json.loads((args.out/'prepared/preparation.json').read_text())
    result=args.out/'fit.json'
    if result.exists():
        if not args.resume:raise ValueError('preserve existing fit output')
        from summit.epistasis.robust import load_robust_scores,robust_score_tests
        from summit.epistasis.cli import _jsonable
        summary=load_robust_scores(args.out/'prepared'/preparation['summaries'][0]['file'])
        expected=_jsonable(dict(kind='summit.epistasis.fit',schema_version=1,
            fits=[robust_score_tests(summary,trait=i,burden=summary.metadata.get('burden_weights'))
                for i in range(len(summary.trait_names))]))
        if json.loads(result.read_text())!=expected:
            raise ValueError('completed portable fit changed')

    else:
        cli(['fit',str(args.out/'prepared'/preparation['summaries'][0]['file']),'--out',str(result)])
    receipt=args.out/'COMPLETED.json'
    if not receipt.exists():
        write_json(receipt,dict(seconds=time.perf_counter()-start,stage_seconds=timings,
            trained=str(trained),training_reused=args.trained is not None,
            files={str(p.resolve()):file_digest(p) for p in (training,args.manifest,
                args.out/'prepared/preparation.json',result)},
            scope='prespecified real-trait pilot; model-based inference, no genome-wide calibration claim'))
    print(json.dumps(dict(event='pilot_complete',fit=str(result),seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()
