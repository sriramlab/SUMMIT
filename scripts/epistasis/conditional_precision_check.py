"""Measure production covariance precision on an authenticated completed fit.

The original frozen diagnostic driver authenticates its own saved contrasts.
This check refits training moments, verifies unchanged coefficients, and adds
aggregate precision diagnostics. It does not rerun or replace the association.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from summit.epistasis.conditional_reference import prepare_conditional_reference
from summit.epistasis.polygenic import (PolygenicKernels, estimate_components,
    conditional_variance_precision)
from summit.prediction.genotype import source_from_spec
from summit.prediction.cli import _rows, _variants
from summit.prediction.artifacts import file_digest
from scripts.epistasis.full_matched import write_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--completed',type=Path,required=True)
    parser.add_argument('--simulation',type=Path,required=True)
    parser.add_argument('--setting',required=True)
    parser.add_argument('--saved-driver',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-threads',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=24)
    parser.add_argument('--block-size',type=int,default=4096)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--directional',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('precision check needs a new output path')
    begin=time.perf_counter()
    spec=json.loads(args.manifest.read_text())
    # __file__ must retain the original frozen path because the saved identity
    # authenticates both path and contents. Never weaken it for a new checkout.
    imported=importlib.util.spec_from_file_location('frozen_conditional_diagnostic',args.saved_driver)
    driver=importlib.util.module_from_spec(imported);imported.loader.exec_module(driver)
    original=argparse.Namespace(manifest=args.manifest,out=args.completed,
        simulation=args.simulation,setting=args.setting,memory_gib=args.memory_gib)
    saved,public_seconds=driver.load_diagnostic_state(original,spec)
    if args.preflight:
        print(json.dumps(dict(authenticated=True,training_shape=saved['y0'].shape,
            confirmation_shape=saved['y1'].shape,original_public_seconds=public_seconds)))
        return
    previous=json.loads((args.completed/'prepared/preparation.json').read_text())
    training_path=(args.manifest.parent/spec['training']).resolve()
    train=json.loads(training_path.read_text());root=training_path.parent
    budget=int(args.memory_gib*2**30)
    with ExitStack() as stack:
        source=stack.enter_context(source_from_spec(train['genotypes'],root))
        source1=stack.enter_context(source_from_spec(train['genotypes'],root))
        rows0=np.sort(_rows(source,root/train['samples']))
        rows1=np.sort(_rows(source,args.manifest.parent/spec['samples']))
        variants=_variants(source,root/train['variants'])
        settings=spec['inference']
        reference=prepare_conditional_reference(source,rows0,rows1,variants,train,root,
            path=Path(previous['reference']),threads=args.num_threads,block_size=args.block_size,
            memory_bytes=budget,probes=settings.get('probes',128),seed=settings.get('seed',871631),
            moment_weighting=settings.get('moment_weighting','none'))
        mean=reference['mean'];training=reference['operator']
        complete=PolygenicKernels(source1,mean['rows'],variants,reference['scales'],
            mean['contexts'],mean['noise'],threads=args.num_threads,
            block_size=1024,memory_bytes=budget)
        retained=sum(v.nbytes for v in mean.values() if isinstance(v,np.ndarray))
        retained+=sum(v.nbytes for v in reference['geometry'].values() if isinstance(v,np.ndarray))
        retained+=sum(v.nbytes for v in saved.values() if isinstance(v,np.ndarray))
        retained+=sum(v.nbytes for v in saved['fit'].values() if isinstance(v,np.ndarray))
        reserve=retained+512*2**20
        training.memory_bytes=budget-complete.base_bytes-reserve
        complete.memory_bytes=budget-training.base_bytes-reserve
        theta,uncertainty=estimate_components(training,saved['y0'],reference['geometry'],
            return_uncertainty='directional' if args.directional else True)
        np.testing.assert_allclose(theta,saved['theta'],rtol=2e-9,atol=1e-11)
        precision=conditional_variance_precision(complete,saved['i0'],saved['i1'],
            saved['fit'],theta,uncertainty,training=training)
        record=dict(training_n=len(rows0),confirmation_n=len(rows1),markers=len(variants),
            kernel_names=reference['metadata']['kernel_names'],covariance_components=theta,
            covariance_precision=precision,seconds=time.perf_counter()-begin,
            original_public_seconds=public_seconds,reference_reused=reference['reused'],
            precision_representation='directional' if args.directional else 'full',
            phenotype_columns=spec['phenotypes']['columns'],
            identity_checks='original frozen diagnostic identity, array digests, genotype reference and refitted coefficients',
            genotype_traversals=dict(training=asdict(training.stream.ledger),confirmation=asdict(complete.stream.ledger)),
            inputs={str(p.resolve()):file_digest(p) for p in (args.saved_driver,args.manifest,
                args.completed/'diagnostic-inputs.npz',args.completed/'prepared/preparation.json')},
            scope='precision diagnostic on an existing development fit; no new calibration or association claim')
        write_json(args.out,record)
        print(json.dumps(dict(event='precision_complete',seconds=record['seconds'],precision=precision)),flush=True)


if __name__=='__main__':main()
