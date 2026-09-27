#!/usr/bin/env python3
"""Bounded local hypertension sanity check; aggregate outputs, no participant rows.

The population is the phenotype-eligible unrelated EUR UKBB cohort. Its observed
case fraction is the declared prevalence for artificial case-status sampling.
This is not an estimate of UK population prevalence or a full-panel h2 analysis.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extension-dir', type=Path, required=True)
    parser.add_argument('--auxiliary-extension-dir', type=Path)
    parser.add_argument('--samples', type=int, default=16000)
    parser.add_argument('--variants', type=int, default=20000)
    parser.add_argument('--reference-samples', type=int, default=4000)
    parser.add_argument('--probes', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--case-fractions', type=float, nargs='+', default=[.5, .9])
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--exact-reference', action='store_true', help='Small-data exact population LD oracle; PCGC study reference still uses variant probes.')
    parser.add_argument('--reuse-pcgc', type=Path, help='Reuse authenticated PCGC fits while rescoring OLS baselines in one study pass.')
    args = parser.parse_args()
    # Load exactly this checkout and the selected native build, never an
    # installed editable finder pointing at an unrelated worktree.
    root = Path(__file__).resolve().parents[2]
    sys.meta_path = [f for f in sys.meta_path if type(f).__module__ != '_gwldcore_editable']
    sys.path.insert(0, str(root/'src'))
    import summit
    summit.__path__.append(str(args.extension_dir.resolve(strict=True)))
    if args.auxiliary_extension_dir:
        summit.__path__.append(str(args.auxiliary_extension_dir.resolve(strict=True)))
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from summit.context.spec import canonical_sha256, array_sha256
    from summit.prediction.genotype import FileGenotypeSource, native_module
    from summit.prediction.spec import GenotypeScale
    from summit.pcgc.genotype import scaled_file_operator
    from summit.pcgc.reference import generalized_reference, contract_reference, population_ld_reference, reference_provenance
    from summit.pcgc.moments import BinaryMoments, fit_moments, external_ld_moments
    from summit.sumstats.binary import fit_binary_risk
    from summit.inference.h2core import prepare_h2, fit_h2
    from summit.inference.jackknife import JackknifeDesign, JackknifeSpec

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError('process-local THP guard failed')
    initial_affinity = sorted(os.sched_getaffinity(0))
    if len(initial_affinity) != args.threads:
        raise ValueError('launch with one reserved physical CPU per requested thread')
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    data_root = Path('/home/bronsonj/UKBB/02_genetics_of_recognition/gwas/pcgc')
    geno = Path('/home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc_imp.bed')
    inputs = [data_root/'code_ht_fix.phe', data_root/'code_ht_fix.cov', data_root/'freq_all.afreq']
    sources = [Path(__file__).resolve(), root/'src/summit/sumstats/binary.py', *sorted((root/'src/summit/pcgc').glob('*.py'))]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    initial_affinity=initial_affinity, thp_disabled=True,
                    covariates=['age','sex','age_sex','age2','age2_sex'],
                    population='phenotype-eligible unrelated EUR UKBB cohort; empirical cohort prevalence',
                    limitations=['restricted common-SNP panel', 'risk model and prevalence treated as fixed in SNP jackknife',
                                 'no ancestry projection; residual structure may affect estimates',
                                 'HE/LDSC global liability conversion is an approximation with covariates'])
    parent_manifest, parent_results = None, None
    if args.reuse_pcgc:
        parent_manifest = json.loads((args.reuse_pcgc/'manifest.json').read_text())
        parent_results = json.loads((args.reuse_pcgc/'results.json').read_text())
        if parent_manifest['input_hashes'] != manifest['input_hashes'] or any(
            parent_manifest['arguments'][k] != getattr(args,k) for k in
            ('seed','samples','variants','reference_samples','probes','case_fractions')):
            raise ValueError('reuse inputs and sampling/probe settings must match exactly')
        for key in ('src/summit/sumstats/binary.py','src/summit/pcgc/moments.py',
                    'src/summit/pcgc/reference.py','src/summit/pcgc/genotype.py'):
            if parent_manifest['source_hashes'][key] != manifest['source_hashes'][key]:
                raise ValueError('PCGC implementation changed since parent fits: '+key)
        manifest['reused_pcgc'] = {str(args.reuse_pcgc/f): hashlib.sha256((args.reuse_pcgc/f).read_bytes()).hexdigest()
                                   for f in ('manifest.json','results.json')}
    pheno = pd.read_csv(inputs[0], sep='\t', dtype={'FID':str,'IID':str})
    cov = pd.read_csv(inputs[1], sep='\t', dtype={'FID':str,'IID':str}, usecols=['FID','IID']+manifest['covariates'])
    table = pheno.merge(cov, on=['FID','IID'], validate='one_to_one').dropna()
    if not set(table.code_ht_fix.unique()) == {0, 1}:
        raise ValueError('expected existing 0/1 phenotype')
    start = time.perf_counter()
    with FileGenotypeSource(geno, genome_build='GRCh37') as source:
        lookup = {s:i for i,s in enumerate(source.samples)}
        table['row'] = [lookup.get(s, -1) for s in zip(table.FID, table.IID)]
        table = table.loc[table.row >= 0].sort_values('row').reset_index(drop=True)
        K = float(table.code_ht_fix.mean())
        rng = np.random.default_rng(args.seed)
        # Hold out the population reference BEFORE case/control sampling.
        ref_indices = np.sort(rng.choice(len(table), args.reference_samples, replace=False))
        reference_rows = table.row.to_numpy()[ref_indices]
        available = table.drop(index=ref_indices)
        af = pd.read_csv(inputs[2], sep='\t').set_index('ID').loc[list(source.variants.ids)]
        if tuple(af.ALT) != source.variants.counted or tuple(af.REF) != source.variants.other:
            raise ValueError('frequency alleles do not match native BED counted A1')
        freq = af.ALT_FREQS.to_numpy(float)
        if not np.all((freq > 0) & (freq < 1)):
            raise ValueError('population scale contains monomorphic SNPs')
        eligible = np.flatnonzero((freq >= .05) & (freq <= .95))
        variants = np.sort(rng.choice(eligible, args.variants, replace=False))
        # Seeded common variants spread across all chromosomes, ordered by the
        # source axis. No selection based on phenotype scores or fit results.
        axis = source.variants.subset(variants)
        scale = GenotypeScale(2*freq, 1/np.sqrt(2*freq*(1-freq)), source.variants.identity,
                              'full_EUR_cohort_frequency_estimate', {'population_scale':True,
                              'source_sha256':manifest['input_hashes'][str(inputs[2])]}, ddof=0)
        manifest.update(eligible_samples=len(table), prevalence=K, source_identity=source.identity,
                        variant_axis_identity=axis.identity, scale_identity=scale.identity,
                        reference_sample_identity=canonical_sha256({'samples':[source.samples[i] for i in reference_rows]}),
                        chromosomes=sorted(set(axis.chromosome), key=int), genotype_format=source.input.format)
        if parent_manifest is not None:
            for key in ('source_identity','variant_axis_identity','scale_identity','reference_sample_identity'):
                if parent_manifest[key] != manifest[key]:
                    raise ValueError('reuse identity mismatch: '+key)
        options = dict(threads=args.threads, probes=args.probes, seed=args.seed,
                       block_size=256, memory_bytes=4*2**30, native=True)
        ref_operator = scaled_file_operator(source, scale, reference_rows, threads=args.threads, variant_indices=variants)
        if args.exact_reference:
            from summit.pcgc.research import exact_external_ld
            from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
            reference_x = np.empty((len(reference_rows), len(variants)), order='F')
            ref_operator.begin_pass(1)
            for offset in range(0, len(variants), 256):
                stop = min(offset+256, len(variants))
                reference_x[:,offset:stop] = ref_operator.read_block(offset,stop).values
            ref_operator.finish_pass()
            nn = ProtectedNNOperator(threads=args.threads, native_module=native_module())
            reference_ld = exact_external_ld(reference_x, np.ones((len(variants),1)),
                matmul=lambda left,right: nn.matmul(np.asfortranarray(left),np.asfortranarray(right)))
            del reference_x
        else:
            reference_ld, _ = population_ld_reference(ref_operator, np.ones((len(variants),1)), **options)
        manifest.update(reference_execution=reference_provenance(options), reference_genotype_passes=ref_operator.observed_passes,
                        population_reference_estimator='exact_small_data_oracle' if args.exact_reference else 'variant_probes',
                        reference_seconds=time.perf_counter()-start, reference_ld_sha256=array_sha256(reference_ld),
                        reference_ld_range=[float(reference_ld.min()),float(reference_ld.max())])
        np.save(args.out/'population_ld.npy',reference_ld,allow_pickle=False)
        print(json.dumps(dict(stage='population_reference', seconds=manifest['reference_seconds'])), flush=True)
        results = []
        for study_index, fraction in enumerate(args.case_fractions):
            stage = time.perf_counter()
            cases = int(round(args.samples*fraction))
            selected = np.sort(np.r_[rng.choice(available.loc[available.code_ht_fix == 1].row, cases, replace=False),
                                     rng.choice(available.loc[available.code_ht_fix == 0].row, args.samples-cases, replace=False)])
            study = available.set_index('row').loc[selected]
            y = study.code_ht_fix.to_numpy(float)
            c = study[manifest['covariates']].to_numpy(float)
            with threadpool_limits(limits=1):
                risk = fit_binary_risk(y, K, c)
                scalar = fit_binary_risk(y, K)
                C = np.column_stack([np.ones(len(y)), (c-c.mean(axis=0))/c.std(axis=0)])
                Q, _ = np.linalg.qr(C, mode='reduced')
                yr = y-Q @ (Q.T @ y)
                var_residual = float(yr @ yr/len(y))
                yr /= np.sqrt(var_residual)
                yg = (y-y.mean())/y.std()
            # Share score products, OLS normalization and exact PCGC diagonal
            # correction in the existing second native genotype pass.
            response = np.column_stack([risk.sensitivity*risk.z, yr, yg, Q])
            diagonal_response = np.column_stack([(risk.sensitivity*risk.z)**2, np.ones(len(y))])
            operator = scaled_file_operator(source, scale, selected, threads=args.threads, variant_indices=variants)
            annotation = np.ones((len(variants),1))
            sample_identity = canonical_sha256({'samples':[source.samples[i] for i in selected]})
            if parent_results is None:
                ref, scored, work = generalized_reference(operator, annotation, risk.sensitivity[:,None],
                    responses=response, diagonal_responses=diagonal_response, collect_diagonal_rows=True, **options)
                ld, sp = contract_reference(ref, np.ones(1))
                planned_bytes = work.peak_resident_bytes+scored.pcgc_score_buffer_bytes
            else:
                parent = parent_results[study_index]
                if parent['sample_identity'] != sample_identity or parent['risk'] != risk.diagnostics():
                    raise ValueError('reuse sample or fitted-risk identity mismatch')
                from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
                tn = ProtectedTNOperator(threads=args.threads, native_module=native_module())
                scored = SimpleNamespace(scores=np.empty((len(variants),response.shape[1])),
                                         diagonals=np.empty((len(variants),diagonal_response.shape[1])),observed_passes=1)
                operator.begin_pass(1)
                for offset in range(0,len(variants),256):
                    stop=min(offset+256,len(variants))
                    x=operator.read_block(offset,stop).values
                    scored.scores[offset:stop]=tn.matmul_tn(x,np.asfortranarray(response))
                    scored.diagonals[offset:stop]=tn.matmul_tn(np.asfortranarray(x*x),np.asfortranarray(diagonal_response))
                operator.finish_pass()
                planned_bytes = None
            rhs = scored.scores[:,0]**2-scored.diagonals[:,0]
            ids = np.arange(len(variants))*50//len(variants)
            if parent_results is None:
                moments = BinaryMoments(annotation, ld-scored.reference_diagonal_rows[:,0]/len(y)**2,
                                         sp, rhs, len(y), 'pcgc', risk.covariate_variance)
                pcgc = fit_moments(moments, block_ids=ids)
            else:
                pcgc = next(m for m in parent['methods'] if m['method']=='pcgc').copy()
            external = fit_moments(external_ld_moments(rhs, annotation, risk, reference_ld), block_ids=ids)
            for fit in (pcgc, external):
                fit.pop('jackknife_replicates',None)
            raw_norm = scored.diagonals[:,1]
            cov_norm = raw_norm-np.sum(scored.scores[:,3:]**2, axis=1)
            centered_norm = raw_norm-scored.scores[:,3]**2
            if np.any(cov_norm <= 0):
                raise ValueError('study SNP has zero residual genotype variance')
            trace = SimpleNamespace(nsnps=len(variants), nbins=1, snps=np.asarray(axis.ids),
                                    annot=annotation, ldscores=reference_ld, delta=None, annot_header=['all'])
            matched = SimpleNamespace(nsnps=len(variants), snps=trace.snps, nsamp=len(y), n_scale=len(y)-1,
                                      n=np.full(len(variants),len(y)))
            jack = JackknifeDesign.from_trace_view(trace, JackknifeSpec.parse(50))
            baselines = []
            summary_rows = {'SNP':np.asarray(axis.ids),'N':matched.n,'pcgc_rhs':rhs}
            for label, score, normalization, variance_fraction, rank in (
                ('age_sex_adjusted',scored.scores[:,1],cov_norm,var_residual/(y.mean()*(1-y.mean())),Q.shape[1]),
                ('intercept_only',scored.scores[:,2],centered_norm,1.,1)):
                # Exact OLS residual degrees of freedom, followed by the SAME
                # beta/SE score reconstruction used by the ordinary h2 CLI.
                matched.beta = score/normalization
                matched.se = np.sqrt((len(y)-score*score/normalization)/((len(y)-rank-1)*normalization))
                summary_rows[label+'_BETA'],summary_rows[label+'_SE']=matched.beta,matched.se
                prepared = prepare_h2(trace, matched, jack)
                for method in ('he','ldsc'):
                    fit = fit_h2(prepared, report_tau=False, weight_mode=method,
                                  ldsc_m_annot=np.array([len(variants)]), ldsc_overlap_matrix=np.array([[len(variants)]]),
                                  ldsc_source_nsnps=len(variants))
                    conversion = variance_fraction/scalar.sensitivity[0]**2
                    baselines.append(dict(method=method, covariates=label,
                        marginal_total=float(fit.h2_reps[-1,0]*conversion),
                        marginal_standard_error=float(fit.h2[-1,1]*conversion),
                        global_conversion=float(conversion)))
            result = dict(samples=len(y), cases=int(y.sum()), variants=len(variants), case_fraction=float(y.mean()),
                          risk=risk.diagnostics(), sample_identity=sample_identity,
                          methods=[pcgc,external], baselines=baselines, genotype_passes=scored.observed_passes,
                          peak_planned_bytes=planned_bytes, baseline_score_contract='SUMMIT_beta_se_exact_h2',
                          seconds=time.perf_counter()-stage)
            summary_path=args.out/f'study_{study_index}.sumstats.npz'
            np.savez_compressed(summary_path,**summary_rows)
            result['summary_rows_sha256']=hashlib.sha256(summary_path.read_bytes()).hexdigest()
            # Native placement is verified through SUMMIT's production guard;
            # retain its attestation, not just thread-count environment values.
            module = native_module()
            result['native_build'] = reference_provenance(options)['native_build']
            result['openmp_placement'] = module.configure_openmp_placement(initial_affinity, args.threads)
            build = module.build_info()
            result['blas_contract'] = {k:build.get(k) for k in ('blas_runtime_threads',
                'blas_runtime_worker_affinity_policy','blas_runtime_thread_strategy','blas_runtime_owner_thread_configured')}
            results.append(result)
            print(json.dumps(dict(stage='study', case_fraction=fraction, seconds=result['seconds'],
                                   pcgc=pcgc['marginal_total'], baselines=baselines)), flush=True)
        manifest['total_seconds'] = time.perf_counter()-start
        source.check()
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    (args.out/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
