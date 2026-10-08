"""Genotype-only full-marker covariance reference for the research candidate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from summit.prediction.genotype import FileGenotypeSource
from summit.prediction.cli import _rows, _variants, _aligned_table
from summit.epistasis.nuisance import varying_main_effects
from scripts.epistasis.polygenic_operator import fit_kernel_scales, PolygenicKernels, he_geometry
from scripts.epistasis.full_matched import write_json
from scripts.epistasis.benchmark_robust_workflow import io


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.chmod(0o700)
    root = a.training.parent.resolve()
    spec = json.loads(a.training.read_text())
    start, cpu, before = time.perf_counter(), time.process_time(), io()
    with FileGenotypeSource(root/spec['genotypes']['geno'], genome_build=spec['genotypes'].get('genome_build')) as source:
        training = np.sort(_rows(source, root/spec['samples']))
        confirmation = np.sort(_rows(source, a.confirmation.resolve()))
        if np.intersect1d(training, confirmation).size:
            raise ValueError('training/confirmation participants overlap')
        complete = np.union1d(training, confirmation)
        i0, i1 = np.searchsorted(complete, training), np.searchsorted(complete, confirmation)
        variants = _variants(source, root/spec['variants'])
        scales = fit_kernel_scales(source, training, variants, threads=a.num_threads,
            block_size=128, memory_bytes=int(a.memory_gib*2**30))
        print('kernel scales',round(time.perf_counter()-start),flush=True)
        samples = [source.samples[i] for i in complete]
        cv = spec['covariates']
        cov = _aligned_table(root/cv['file'], samples)[cv['columns']].to_numpy(float)
        names = cv.get('varying_effects', [])
        if not names:
            raise ValueError('supply the prespecified structure covariates')
        z = cov[:, [cv['columns'].index(name) for name in names]]
        local_names = sorted(set(spec.get('local_variants', [])) | set(spec.get('dominance_variants', [])) | {spec['target']})
        local = np.array(sorted(source.variants.ids.index(v) for v in local_names))
        local_names = [source.variants.ids[i] for i in local]
        source.prepare(complete, max(128,len(local)), a.num_threads)
        raw = source.read(local)
        observed = raw != -127
        dosage_mean = np.where(observed[i0], raw[i0], 0).sum(0)/observed[i0].sum(0)
        h = raw == 1
        dominance_mean = h[i0].sum(0)/observed[i0].sum(0)
        additive = np.where(observed, raw, dosage_mean)
        dominance = np.where(observed, h, dominance_mean)
        target = local_names.index(spec['target'])
        if not np.all(observed[:, target]):
            raise ValueError('target must be observed in this supplied-target experiment')
        x = (additive[:, target]-dosage_mean[target])/np.sqrt(dosage_mean[target]*(1-dosage_mean[target]/2))
        fixed = np.column_stack([np.ones(len(complete)), cov, additive, dominance])
        fixed, definition = varying_main_effects(fixed, z, x[:,None],
            covariate_names=names, main_names=[spec['target']], memory_bytes=int(a.memory_gib*2**30))
        contexts = np.column_stack([np.ones(len(complete)), z])
        noise = np.column_stack([np.ones(len(complete)), x*x])
        operator = PolygenicKernels(source, training, variants, scales, contexts[i0], noise[i0],
            threads=a.num_threads, block_size=512, memory_bytes=int(a.memory_gib*2**30), storage='packed')
        geometry = he_geometry(operator, fixed[i0], probes=a.probes, seed=a.seed)
        archive = dict(rows=complete, training_index=i0, confirmation_index=i1, variants=variants,
            mean=scales['mean'], inverse_scale=scales['inverse_scale'], fixed=fixed, contexts=contexts,
            noise=noise, target=x, **{k:v for k,v in geometry.items() if isinstance(v,np.ndarray)})
        np.savez(a.out/'reference.npz', **archive)
        after = io()
        write_json(a.out/'reference.json',dict(training_manifest=str(a.training.resolve()),
            confirmation_samples=str(a.confirmation.resolve()), genotypes=spec['genotypes'], source=source.identity,
            scales={k:v for k,v in scales.items() if k not in ('mean','inverse_scale')},
            geometry={k:v for k,v in geometry.items() if not isinstance(v,np.ndarray)},
            kernel_names=['additive',*[f'additive_by_{name}' for name in names],'dominance','noise_iid','noise_target_squared'],
            covariance_condition=float(np.linalg.cond(geometry['h'])),
            kernel_trace_mc_se=(geometry['probe_gram'].std(0,ddof=1)/np.sqrt(a.probes)).tolist(),
            scientific_scope='research conditional Gaussian polygenic mean test with estimated covariance; not a FAME variance test',
            fixed_definition=definition, n_training=len(training),n_confirmation=len(confirmation),markers=len(variants),
            source_traversals=asdict(operator.stream.ledger), seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            io={k:after[k]-v for k,v in before.items()}))
    print('full covariance reference',round(time.perf_counter()-start),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probes',type=int,default=128)
    p.add_argument('--seed',type=int,default=871631)
    p.add_argument('--num-threads',type=int,default=1)
    p.add_argument('--memory-gib',type=float,default=16)
    run(p.parse_args())


if __name__=='__main__':
    main()
