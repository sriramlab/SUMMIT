"""Manifest entry point for scale-free focal-context x background epistasis.

Manifest (JSON, paths relative to the manifest):

    kind: "summit.epistasis.scale_free", schema_version: 1
    genotypes: SUMMIT genotype specification (background and focal variants)
    samples: FID/IID table of the pass rows (e.g. a held-out analysis split)
    phenotypes: {file, columns, positive?: [columns restricted to values > 0]}
    covariates?: {file, columns?, categorical?}
    strata?: {file, column}      crossed with context classes (e.g. sex)
    contexts: list of
        {name, variant, coding?: additive|carrier, min_minor_homozygotes?: 300}
        {name, file, column, values: {label: value}, low, high, exclude_chromosomes}
    analyses?: [[context, phenotype], ...]   default: all combinations
    background?: {variants?: ID file, annotations?: {file, columns}, jackknife_blocks?: 200}
    scales?: subset of raw/pooled_rank/within_rank (default pooled_rank, within_rank)
    pattern_scales?: scales that also get lo/hi pattern columns (default all)
    additive_scales?: scales that also get the additive column (default none)
    groups?: 200, seed?: 20261011, genotype_scale?: empirical|hwe, save_kernels?: false

A variant context codes minor-allele counts on the pass rows. With fewer than
min_minor_homozygotes, the minor homozygotes are merged with the
heterozygotes (carrier coding). Its own chromosome is excluded from the
background.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .design import Context, Outcome, build_plan
from .stats import summarize
from .stream import accumulate, background_axis, plan_block_size

KIND = 'summit.epistasis.scale_free'


def _numeric(table, columns):
    import pandas as pd
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    return table[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)


def covariate_matrix(table, columns=(), categorical=()):
    """Numeric columns plus full-rank dummies of categorical columns; NaN marks missing."""
    import pandas as pd
    parts = [_numeric(table, list(columns))] if columns else []
    for name in categorical:
        if name not in table.columns:
            raise ValueError(f'missing categorical covariate {name}')
        values = table[name].astype(str).replace({'': np.nan, 'NA': np.nan, 'nan': np.nan})
        dummies = pd.get_dummies(values, drop_first=True, dtype=np.float64).to_numpy()
        dummies[values.isna().to_numpy()] = np.nan
        parts.append(dummies)
    return np.column_stack(parts) if parts else np.zeros((len(table), 0))


def variant_contexts(source, rows, specs, *, threads=1):
    """Minor-allele-count contexts read through the shared genotype stream."""
    from summit.prediction.genotype import RawBlockStream
    if not specs:
        return []
    lookup = {v: i for i, v in enumerate(source.variants.ids)}
    missing = [s['variant'] for s in specs if s['variant'] not in lookup]
    if missing:
        raise ValueError(f'focal variants absent from the genotype source: {missing}')
    wanted = np.array(sorted({lookup[s['variant']] for s in specs}), dtype=np.int64)
    stream = RawBlockStream(source, rows, wanted, block_size=len(wanted), threads=threads)
    raw = np.concatenate([np.array(r, copy=True) for _, _, r in stream.blocks('scale_free_focal')], axis=1)
    column = {int(v): k for k, v in enumerate(wanted)}
    contexts = []
    for s in specs:
        j = lookup[s['variant']]
        calls = raw[:, column[j]].astype(np.int64)
        observed = calls != -127
        if np.any((calls[observed] < 0) | (calls[observed] > 2)):
            raise ValueError(f"{s['variant']}: variant contexts require hard calls")
        if calls[observed].mean() > 1:
            calls = np.where(observed, 2 - calls, calls)
        coding = s.get('coding', 'additive')
        if coding not in ('additive', 'carrier'):
            raise ValueError('variant coding must be additive or carrier')
        homozygotes = int(np.sum(calls == 2))
        if coding == 'carrier' or homozygotes < int(s.get('min_minor_homozygotes', 300)):
            labels, values, coding = np.where(observed, np.minimum(calls, 1), -1), {0: 0., 1: 1.}, 'carrier'
        else:
            labels, values = np.where(observed, calls, -1), {0: 0., 1: 1., 2: 2.}
        contexts.append((Context(name=s['name'], labels=labels, values=values, low=(0,), high=(1, 2),
                                 exclude_chromosomes=(source.variants.chromosome[j],)),
                         dict(variant=s['variant'], coding=coding, minor_homozygotes=homozygotes,
                              chromosome=source.variants.chromosome[j],
                              position=int(source.variants.position[j]))))
    return contexts


def run(manifest, out, *, threads=1, memory_bytes=16 * 2**30, block_size=1024, native=True, progress=None):
    from summit.prediction._validation import closed
    from summit.prediction.artifacts import read_json
    from summit.prediction.cli import _aligned_table, _rows, _variants
    from summit.prediction.genotype import estimate_scale, native_module, source_from_spec
    from summit.prediction.runtime import configure_prediction_threads
    from ..cli import _jsonable

    started = perf_counter()
    code = _code_identity()
    if progress is None:
        progress = _reporter()
    path, out = Path(manifest).resolve(), Path(out)
    spec = read_json(path)
    closed(spec, ('kind', 'schema_version', 'genotypes', 'samples', 'phenotypes', 'contexts'),
           ('covariates', 'strata', 'analyses', 'background', 'scales', 'pattern_scales', 'additive_scales',
            'groups', 'seed', 'genotype_scale', 'save_kernels'), name='scale-free manifest')
    if spec['kind'] != KIND or spec['schema_version'] != 1:
        raise ValueError('unsupported scale-free manifest')
    root = path.parent
    phen = spec['phenotypes']
    closed(phen, ('file', 'columns'), ('positive',), name='phenotypes')
    if not phen['columns'] or len(set(phen['columns'])) != len(phen['columns']):
        raise ValueError('phenotype columns must be nonempty and unique')
    names = [c['name'] for c in spec['contexts']]
    if not names or len(set(names)) != len(names):
        raise ValueError('context names must be nonempty and unique')
    for c in spec['contexts']:
        if 'variant' in c:
            closed(c, ('name', 'variant'), ('coding', 'min_minor_homozygotes'), name='variant context')
        else:
            closed(c, ('name', 'file', 'column', 'values', 'low', 'high', 'exclude_chromosomes'), (),
                   name='file context')
    if spec.get('genotype_scale', 'empirical') not in ('empirical', 'hwe'):
        raise ValueError('genotype_scale must be empirical or hwe')
    out.mkdir(parents=True, exist_ok=False)

    if native:
        configure_prediction_threads(native_module(), threads)
    source = source_from_spec(spec['genotypes'], root)
    try:
        rows = np.sort(_rows(source, root / spec['samples']))
        samples = [source.samples[i] for i in rows]
        table = _aligned_table(root / phen['file'], samples)
        values = _numeric(table, phen['columns'])
        positive = set(phen.get('positive', ()))
        outcomes = [Outcome(c, values[:, k], positive=c in positive) for k, c in enumerate(phen['columns'])]
        covariates = np.zeros((len(rows), 0))
        if 'covariates' in spec:
            cv = spec['covariates']
            closed(cv, ('file',), ('columns', 'categorical'), name='covariates')
            covariates = covariate_matrix(_aligned_table(root / cv['file'], samples), cv.get('columns', ()),
                                          cv.get('categorical', ()))
        strata = None
        if 'strata' in spec:
            closed(spec['strata'], ('file', 'column'), (), name='strata')
            column = _aligned_table(root / spec['strata']['file'], samples)[spec['strata']['column']]
            strata = column.astype(str).to_numpy()
        variant_specs = [c for c in spec['contexts'] if 'variant' in c]
        focal = variant_contexts(source, rows, variant_specs, threads=threads)
        contexts, context_info = [], {}
        for ctx, info in focal:
            contexts.append(ctx)
            context_info[ctx.name] = info
        for c in spec['contexts']:
            if 'variant' in c:
                continue
            labels = _numeric(_aligned_table(root / c['file'], samples), [c['column']])[:, 0]
            labels = np.where(np.isfinite(labels), labels, -1).astype(np.int64)
            contexts.append(Context(name=c['name'], labels=labels,
                                    values={int(k): float(v) for k, v in c['values'].items()},
                                    low=tuple(c['low']), high=tuple(c['high']),
                                    exclude_chromosomes=tuple(str(x) for x in c['exclude_chromosomes'])))
            context_info[c['name']] = dict(file=c['file'], column=c['column'])
        plan = build_plan(rows, contexts, outcomes, covariates, analyses=spec.get('analyses'), strata=strata,
                          scales=spec.get('scales', ('pooled_rank', 'within_rank')),
                          pattern_scales=spec.get('pattern_scales'), additive_scales=spec.get('additive_scales', ()),
                          n_groups=spec.get('groups', 200),
                          seed=spec.get('seed', 20261011))
        print(f'scale-free plan: {len(plan.analyses)} analyses, {plan.n_columns} columns, {len(rows)} rows, '
              f'{perf_counter() - started:.0f} s', file=sys.stderr, flush=True)
        bg = spec.get('background', {})
        closed(bg, (), ('variants', 'annotations', 'jackknife_blocks'), name='background')
        variants = _variants(source, root / bg['variants'] if 'variants' in bg else None)
        annotations, annotation_names = None, ()
        if 'annotations' in bg:
            closed(bg['annotations'], ('file', 'columns'), (), name='background annotations')
            ann = _table_by_variant(root / bg['annotations']['file'], bg['annotations']['columns'],
                                    [source.variants.ids[i] for i in variants])
            annotations, annotation_names = ann, tuple(bg['annotations']['columns'])
        background = background_axis(source, variants=variants, annotations=annotations, names=annotation_names,
                                     n_blocks=bg.get('jackknife_blocks', 200))
        scale = estimate_scale(source, rows, background.variants, ddof=1, block_size=256, threads=threads,
                               memory_bytes=memory_bytes)
        if spec.get('genotype_scale') == 'hwe':
            from summit.prediction.spec import GenotypeScale
            variance = scale.mean * (1 - scale.mean / 2)
            inverse = np.divide(1., np.sqrt(variance), out=np.ones_like(variance), where=variance > 0)
            scale = GenotypeScale(scale.mean, inverse, scale.variant_identity, scale.sample_identity,
                                  {'method': 'study_mean_imputed_hwe', 'source': source.identity}, ddof=0)
        print(f'scale-free genotype scale fitted, {perf_counter() - started:.0f} s', file=sys.stderr, flush=True)
        width = plan_block_size(plan, len(rows), memory_bytes=memory_bytes, block_size=block_size)
        result = accumulate(source, plan, scale, background, block_size=width, threads=threads, native=native,
                            progress=progress)
        records = summarize(plan, result)
        source_identity = source.identity
    finally:
        source.close()

    import pandas as pd
    pd.DataFrame(records).to_csv(out / 'results.tsv', sep='\t', index=False, na_rep='NA', float_format='%.6g')
    run_info = dict(kind=KIND + '.result', schema_version=1, manifest=spec, n_rows=int(len(rows)),
                    contexts=context_info, analyses=plan.analyses, annotations=list(background.names),
                    background=dict(variants=int(len(background.variants)), jackknife_blocks=background.n_blocks),
                    genotype_source=source_identity, genotype_scale=scale.identity,
                    groups=plan.n_groups, seed=spec.get('seed', 20261011), diagnostics=result.diagnostics,
                    seconds=perf_counter() - started)
    run_info['code_sha256'] = code
    if native:
        info = native_module().build_info()
        run_info['native_build'] = {k: info.get(k) for k in ('source_commit', 'backend_version', 'blas_vendor',
                                                              'gemm_integrity_enabled')}
    with (out / 'run.json').open('x') as handle:
        json.dump(_jsonable(run_info), handle, indent=1, allow_nan=False)
        handle.write('\n')
    if spec.get('save_kernels', False):
        arrays = {f'kernel_{p}': v for p, v in result.kernels.items()}
        arrays.update({f'between_{p}': v for p, v in result.between.items()})
        arrays.update({f'keep_{p}': v for p, v in result.keep.items()})
        np.savez(out / 'kernels.npz', block_weight=result.block_weight,
                 block_chromosomes=result.block_chromosomes.astype(str), **arrays)
    (out / 'COMPLETE').write_text('')
    return out


def _code_identity():
    """SHA-256 over this package's Python sources (sorted by name)."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def _reporter(step=.02):
    state = dict(next=0.)

    def report(done, total, seconds):
        if done / total >= state['next'] or done == total:
            state['next'] = done / total + step
            rate = done / max(seconds, 1e-9)
            print(f'scale-free pass {done}/{total} variants, {seconds:.0f} s, '
                  f'{(total - done) / max(rate, 1e-9):.0f} s left', file=sys.stderr, flush=True)
    return report


def _table_by_variant(path, columns, ids):
    import pandas as pd
    table = pd.read_csv(path, sep=r'\s+', dtype={'SNP': str})
    if 'SNP' not in table or table['SNP'].duplicated().any():
        raise ValueError('annotation table needs a unique SNP column')
    table = table.set_index('SNP')
    if not pd.Index(ids).isin(table.index).all():
        raise ValueError('annotation table is missing background variants')
    return _numeric(table.loc[ids], list(columns))
