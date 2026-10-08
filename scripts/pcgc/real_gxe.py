"""Paired real-array PCGC/HE research comparison; aggregate outputs only.

Target population: the phenotype/covariate-eligible direct-array EUR cohort.
Empirical cohort prevalence defines deliberate case/control sampling, not UK
population prevalence. Genotypes are projected on the ten ancestry PCs
BEFORE context multiplication, with no subsequent SNP renormalization.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from summit.context.fit import solve_context_normal_equations
from summit.context.fit_v1 import assemble_contextual_normal_equations_from_moments_v1
from summit.context.reference import ReferenceMoments
from summit.context.spec import array_sha256
from summit.ldscore.generalized_gxe_fit_v1 import _trait_moments_after_deleting_blocks
from summit.ldscore.generalized_gxe_pass1 import GeneralizedGxEPass1Executor, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import GeneralizedGxEPass2Executor, ProtectedTNOperator
from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
from summit.ldscore.generalized_gxe_trait_summary import (
    GeneralizedGxEPerVariantTraitStatistics, aggregate_generalized_gxe_trait_statistics,
    generalized_gxe_per_variant_trait_statistics)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs, GlobalVariantProbeSpec, plan_generalized_gxe_variant_work)
from summit.pcgc.genotype import scaled_file_operator
from summit.pcgc.gxe import context_pairs, fit_gxe, prepare_gxe_moments
from summit.prediction.genotype import FileGenotypeSource, native_module
from summit.prediction.spec import GenotypeScale
from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk


def basis(c):
    c = np.asarray(c, float)
    q, r = np.linalg.qr(c, mode='reduced')
    if np.min(np.abs(np.diag(r))) < 1e-10:
        raise ValueError('rank deficient fixed design')
    return np.asfortranarray(q)


class AncestryOperator:
    """Common pre-context genotype transformation, using protected GEMMs."""
    def __init__(self, operator, u, nn, tn):
        self.operator, self.u, self.nn, self.tn = operator, u, nn, tn
        self.genotype_scale_id = operator.genotype_scale_id + ':PC:' + array_sha256(u)
        self.last_progress = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def read_block(self, start, stop):
        b = self.operator.read_block(start, stop)
        cross = self.tn.matmul_tn(self.u, b.values)
        x = np.asfortranarray(b.values - self.nn.matmul(self.u, np.asfortranarray(cross)))
        if time.monotonic() - self.last_progress > 120:
            print(json.dumps(dict(stage='genotypes', traversal=self.observed_passes,
                                  variants_done=stop, variants_total=self.num_variants)), flush=True)
            self.last_progress = time.monotonic()
        return replace(b, values=x, genotype_scale_id=self.genotype_scale_id)


class HEScorer:
    """Collect exact HE sufficient statistics during the second reference pass."""
    def __init__(self, operator, phi, u, y, d, tn):
        self.operator, self.phi, self.u, self.d, self.tn = operator, phi, u, d, tn
        n, q = phi.shape
        h, r = d.shape[1], u.shape[1]
        self.pairs = context_pairs(q)
        # The existing small-data implementation supplies residual-only moments.
        self.template = generalized_gxe_per_variant_trait_statistics(
            genotype=np.zeros((n, 1)), basis=phi, fixed_basis=u,
            phenotypes=y, residual_basis=d)
        self.weights = np.asfortranarray(np.column_stack([
            phi * self.template.normalized_phenotypes,
            *[phi[:, j, None]*u for j in range(q)],
            *[(d[:, k]*phi[:, j])[:, None]*u for k in range(h) for j in range(q)]]))
        products = np.column_stack([phi[:, j]*phi[:, k] for j, k in self.pairs])
        self.square_weights = np.asfortranarray(np.column_stack(
            [products, *[products*d[:, k, None] for k in range(h)]]))
        self.compressed = [u.T @ (d[:, k, None]*u) for k in range(h)]
        m, p = operator.num_variants, len(self.pairs)
        self.scores = np.empty((m, q, 1))
        self.information = np.empty((m, p))
        self.residual_information = np.empty((m, p, h))

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def read_block(self, start, stop):
        b = self.operator.read_block(start, stop)
        if self.observed_passes != 2:
            return b
        n, q = self.phi.shape
        h, r, p = self.d.shape[1], self.u.shape[1], len(self.pairs)
        cross = self.tn.matmul_tn(b.values, self.weights)
        squares = self.tn.matmul_tn(np.asfortranarray(b.values*b.values), self.square_weights)
        self.scores[start:stop, :, 0] = cross[:, :q]
        fixed = cross[:, q:q+q*r].reshape(stop-start, q, r)
        weighted = cross[:, q+q*r:].reshape(stop-start, h, q, r)
        for t, (j, k) in enumerate(self.pairs):
            left, right = fixed[:, j], fixed[:, k]
            self.information[start:stop, t] = squares[:, t] - np.sum(left*right, axis=1)
            for ell in range(h):
                self.residual_information[start:stop, t, ell] = (
                    squares[:, p*(ell+1)+t] - np.sum(left*weighted[:, ell, k], axis=1)
                    - np.sum(weighted[:, ell, j]*right, axis=1)
                    + np.einsum('mi,ij,mj->m', left, self.compressed[ell], right, optimize=True))
        return b

    def statistics(self):
        return replace(self.template, scores=self.scores, information=self.information,
                       residual_information=self.residual_information)


def fit_he(operator, phi, u, y, ids, options, nn, tn):
    n, q = phi.shape
    m = operator.num_variants
    # Span of all context products; prune the redundant binary-sex square.
    candidates = [np.ones(n), phi[:, 1], phi[:, 2], phi[:, 1]**2, phi[:, 1]*phi[:, 2]]
    d = np.asfortranarray(np.column_stack(candidates))
    if np.linalg.matrix_rank(d) != d.shape[1]:
        raise ValueError('residual basis rank deficient')
    scored = HEScorer(operator, phi, u, y, d, tn)
    a = np.ones((m, 1))
    plan = plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
        num_samples=n, num_variants=m, num_basis=q, num_annotations=1,
        fixed_effect_rank=u.shape[1], num_probes=options['probes'],
        memory_limit_bytes=options['memory_bytes']-2**30, genotype_format=operator.genotype_format,
        threads=options['threads'], preferred_variant_block_width=options['block_size'],
        preferred_rhs_tile_columns=q*q*min(64, options['probes']), rhs_policy='tiled'))
    scored.configure_block_width(plan.tiling['variant_block_width'])
    common = dict(genotype_operator=scored, basis=phi, fixed_effect_basis=u,
                  annotations=a, annotation_names=('all',), work_plan=plan)
    spec = GlobalVariantProbeSpec(root_seed=options['seed'], probe_offset=0, probe_count=options['probes'])
    first = GeneralizedGxEPass1Executor(**common, annotation_masses=np.array([m]),
        probe_spec=spec, nn_operator=nn, native_probe_module=native_module(),
        probe_tile_width=min(64, options['probes'])).execute()
    ref = GeneralizedGxEPass2Executor(**common, pass1_result=first, tn_operator=tn,
        probe_tile_width=min(64, options['probes'])).execute()
    labels = tuple(map(str, np.unique(ids)))
    stats = scored.statistics()
    trait = aggregate_generalized_gxe_trait_statistics(stats, annotations=a, annotation_names=('all',),
        variant_group_ids=ids, group_labels=labels, trait_ids=('disease',),
        residual_names=('constant', 'age', 'sex', 'age2', 'age_sex'), n_samples=n)
    bd, bm, error = reduce_generalized_gxe_reference_for_inference(
        directional_ldscores=ref.directional_ldscores, annotations=a,
        variant_block_ids=ids, block_labels=labels, expected_directed_numerator=ref.directed_numerator)
    solutions, equations = [], []
    for j in range(-1, len(labels)):
        mass = np.array([m], float) if j < 0 else np.array([m], float)-bm[j]
        directed = ref.directed_numerator if j < 0 else ref.directed_numerator-bd[j]
        gram = ref.residual_rank**2*(directed+directed.T)/2/mass[0]**2
        rm = ReferenceMoments(mass, gram, ref.same_person)
        deleted = () if j < 0 else (labels[j],)
        eq = assemble_contextual_normal_equations_from_moments_v1(
            component_index=trait.component_index, reference_n=n, trait=trait,
            trait_index=0, deleted_groups=deleted, reference_moments=rm,
            trait_moments=_trait_moments_after_deleting_blocks(trait, deleted))
        solution = solve_context_normal_equations(eq, require_full_rank=True)
        solutions.append(solution.coefficients)
        if j < 0:
            equations.append(eq)
    solutions = np.asarray(solutions)
    c = len(context_pairs(q))
    yr = y-u@(u.T@y)
    return dict(components=solutions[0, :c], jackknife_replicates=solutions[1:, :c],
        normal_matrix=equations[0].matrix, normal_rhs=equations[0].rhs,
        residual_coefficients=solutions[0, c:], residual_observed_variance=float(yr@yr/ref.residual_rank),
        normal_condition=float(np.linalg.cond(equations[0].matrix)),
        reference_reconstruction_error=float(error), genotype_passes=operator.observed_passes,
        reference_plan=plan.to_dict())


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--phenotypes', type=Path, help='Optional relocated copy of the existing curated table')
    parser.add_argument('--traits', nargs='+', default=['hypertension', 'glaucoma', 'ulcerative_colitis', 'depressive_episode'])
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--variants', type=int, default=50000, help='0 selects all polymorphic array SNPs')
    parser.add_argument('--probes', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--probe-seed', type=int, default=71261)
    parser.add_argument('--sampling-partners', type=int, default=128)
    parser.add_argument('--architecture-probes', type=int, default=32)
    parser.add_argument('--sampling-seed', type=int, default=81261)
    parser.add_argument('--memory-gib', type=float, default=12)
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prefix = args.data_root/'geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc'
    paths = dict(pheno=args.data_root/'UKB_phen.txt',
        cov=args.data_root/'real_phens/covars/EUR_300k.insample10.covar.NA.txt',
        frequencies=Path(str(prefix)+'_imp.afreq'))
    if args.phenotypes:
        paths['pheno'] = args.phenotypes
    covars = ['age', 'sex', 'age*sex', 'age^2', 'age^2*sex']+[f'PC{i}' for i in range(1, 11)]
    phenotype = pd.read_csv(paths['pheno'], sep='\t', usecols=['FID', 'IID']+args.traits,
                            dtype={'FID':str, 'IID':str})
    cov = pd.read_csv(paths['cov'], sep=r'\s+', usecols=['FID', 'IID']+covars,
                      dtype={'FID':str, 'IID':str}, na_values=['NA', '-9'])
    table = phenotype.merge(cov, on=['FID', 'IID'], validate='one_to_one').dropna(subset=covars)
    module = native_module()
    nn, tn = ProtectedNNOperator(threads=args.threads, native_module=module), ProtectedTNOperator(threads=args.threads, native_module=module)
    options = dict(probes=args.probes, seed=args.probe_seed, threads=args.threads,
                   memory_bytes=int(args.memory_gib*2**30), block_size=512)
    with FileGenotypeSource(Path(str(prefix)+'.bed'), genome_build='GRCh37') as source:
        lookup = {s:i for i, s in enumerate(source.samples)}
        table['row'] = [lookup.get(s, -1) for s in zip(table.FID, table.IID)]
        table = table.loc[table.row >= 0].sort_values('row').reset_index(drop=True)
        af = pd.read_csv(paths['frequencies'], sep='\t').set_index('ID').loc[list(source.variants.ids)]
        forward = (af.ALT.to_numpy() == source.variants.counted) & (af.REF.to_numpy() == source.variants.other)
        reverse = (af.REF.to_numpy() == source.variants.counted) & (af.ALT.to_numpy() == source.variants.other)
        if not np.all(forward | reverse) or np.any(forward & reverse):
            raise ValueError('frequency and BED allele axes disagree')
        freq = np.where(forward, af.ALT_FREQS.to_numpy(float), 1-af.ALT_FREQS.to_numpy(float))
        eligible = np.flatnonzero((freq > 0) & (freq < 1))
        if len(eligible) != len(freq):
            raise ValueError('population scale includes monomorphic variants')
        rng = np.random.default_rng(args.seed)
        variants = eligible if not args.variants else np.sort(rng.choice(eligible, args.variants, replace=False))
        scale = GenotypeScale(2*freq, 1/np.sqrt(2*freq*(1-freq)), source.variants.identity,
            'full_EUR_cohort_frequency_estimate', {'population_scale':True,
            'source_sha256':hashlib.sha256(paths['frequencies'].read_bytes()).hexdigest()}, ddof=0)
        ids = np.arange(len(variants))*args.blocks//len(variants)
        manifest = dict(arguments=vars(args), source_identity=source.identity, scale_identity=scale.identity,
            variant_axis_identity=source.variants.subset(variants).identity, sample_count_parent=len(table),
            variant_count=len(variants), chromosome_counts=dict(zip(*np.unique(
                np.asarray(source.variants.chromosome)[variants], return_counts=True))),
            frequency_alleles_reversed=int(reverse.sum()),
            input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            genotype_transform='ten supplied ancestry PC directions removed before context multiplication; no intercept removal or renormalization',
            liability_scale='unit conditional total liability SD; marginal denominator estimated by inverse ascertainment',
            risk_estimation='same-study ascertainment-aware probit; uncertainty propagated',
            target_population='phenotype/covariate-eligible EUR direct-array UKBB cohort; empirical prevalence',
            context_names=['intercept', 'age_standardized_in_parent', 'sex_code_standardized_in_parent'],
            risk_covariates=covars, ascertainment='uniform sampling within case/control status, balanced case/control study',
            native_build=module.build_info())
        code_root = Path(__file__).resolve().parents[2]
        code_files = [Path(__file__), code_root/'scripts/pcgc/gxe_python.py',
                      code_root/'src/summit/sumstats/binary.py',
                      *sorted((code_root/'src/summit/pcgc').glob('*.py')),
                      *sorted((code_root/'src/summit/ldscore').glob('generalized_gxe*.py')),
                      *sorted((code_root/'src/summit/context').glob('fit*.py'))]
        manifest['source_hashes'] = {str(p.relative_to(code_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in code_files}
        manifest['runtime'] = dict(host=os.uname().nodename, job_id=os.environ.get('JOB_ID'),
            native_module_path=module.__file__, native_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (args.out/'manifest.json').write_text(json.dumps(manifest, default=json_value, indent=2)+'\n')
        for name in args.traits:
            stage = time.perf_counter()
            parent = table.dropna(subset=[name]).copy()
            if set(parent[name].unique()) != {0, 1}:
                raise ValueError('expected 0/1 curated disease phenotype')
            yparent = parent[name].to_numpy(float)
            K = float(yparent.mean())
            cparent = parent[covars].to_numpy(float)
            env = parent[['age', 'sex']].to_numpy(float)
            env_mean, env_sd = env.mean(axis=0), env.std(axis=0)
            phi_parent = np.column_stack([np.ones(len(env)), (env-env_mean)/env_sd])
            metric = phi_parent.T@phi_parent/len(env)
            sampling = np.random.default_rng(args.seed+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little'))
            nc = min(args.samples//2, int(yparent.sum()), int(len(parent)-yparent.sum()))
            selected = np.sort(np.r_[sampling.choice(np.flatnonzero(yparent == 1), nc, replace=False),
                                    sampling.choice(np.flatnonzero(yparent == 0), nc, replace=False)])
            study = parent.iloc[selected]
            y, phi = yparent[selected], np.asfortranarray(phi_parent[selected])
            scalar = prepare_binary_risk(y, K)
            c = cparent[selected]
            with threadpool_limits(limits=1):
                risk = fit_binary_risk(y, K, c)
            u = basis(np.column_stack([np.ones(len(y)), (c-c.mean(axis=0))/c.std(axis=0)]))
            # Do not remove an intercept here: that would replace the population
            # SNP centering by the ascertained study mean. Keep supplied PC
            # coordinates (no study recentering); only the HE response/features
            # use the full fixed-effect projection including the intercept.
            upc = basis(c[:, 5:])
            def operator():
                raw = scaled_file_operator(source, scale, study.row.to_numpy(), threads=args.threads,
                                           variant_indices=variants, block_size=options['block_size'])
                return AncestryOperator(raw, upc, nn, tn)
            nuisance_seconds = time.perf_counter()-stage
            reference_start = time.perf_counter()
            print(json.dumps(dict(stage='trait_start', trait=name, n=len(y), cases=nc, K=K,
                                  nuisance_seconds=nuisance_seconds)), flush=True)
            moments, diagnostics = prepare_gxe_moments(operator(), np.ones((len(variants), 1)), risk, phi,
                liability_sd=1., sampling_partners=args.sampling_partners,
                sampling_seed=args.sampling_seed, architecture_probes=args.architecture_probes,
                risk_covariates=c if args.sampling_partners else None, **options)
            pcgc = fit_gxe(moments, block_ids=ids)
            total = moments.population_liability_variance
            pcgc_metric = moments.population_kernel_second_moment[0].copy()
            del moments
            pcgc_seconds = time.perf_counter()-reference_start
            print(json.dumps(dict(stage='pcgc_complete', trait=name, seconds=pcgc_seconds)), flush=True)
            he_start = time.perf_counter()
            he = fit_he(operator(), phi, u, y, ids, options, nn, tn)
            he_seconds = time.perf_counter()-he_start
            # HE estimates variance of the normalized projected observed trait.
            # Global marginal-liability conversion is deliberately the comparator.
            factor = he['residual_observed_variance']/(y.mean()*(1-y.mean()))/scalar.sensitivity[0]**2
            he['global_liability_conversion'] = factor
            he['marginal_components'] = he['components']*factor
            he['marginal_jackknife_replicates'] = he['jackknife_replicates']*factor
            pc_theta, pc_loo = np.asarray(pcgc['components'])/total, np.asarray(pcgc['jackknife_replicates'])/total
            population_weights = np.array([(1 if j == k else 2)*metric[j, k] for j, k in context_pairs(3)])
            rows = []
            for label, vector in [('population_average_genetic_fraction', population_weights)]+[
                    (f'omega_{j}_{k}_marginal', np.eye(6)[t]) for t, (j, k) in enumerate(context_pairs(3))]:
                x, z = float(vector@pc_theta), float(vector@he['marginal_components'])
                p, h = pc_loo@vector, he['marginal_jackknife_replicates']@vector
                jkse = lambda v: float(np.sqrt((len(v)-1)*np.var(v, ddof=0)))
                row = dict(quantity=label, pcgc=x, pcgc_snp_block_se=jkse(p), he=z, he_snp_block_se=jkse(h))
                if label == 'population_average_genetic_fraction':
                    row.update(pcgc=pcgc['population_heritability'],
                        pcgc_se=pcgc['population_heritability_se'],
                        pcgc_score_set_95=pcgc.get('population_heritability_plugin_score_set_95'),
                        pcgc_snp_block_se=None)
                else:
                    # Marginal coefficients use an estimated denominator. The
                    # saved joint inference is reported for original-scale Ω;
                    # dividing an Ω SE alone would omit denominator uncertainty.
                    t = int(np.argmax(vector))
                    row.update(pcgc_conditional=float(pcgc['components'][t]),
                        pcgc_conditional_se=float(pcgc['standard_errors'][t]))
                row['difference_pcgc_minus_he'] = row['pcgc']-z
                rows.append(row)
            record = dict(trait=name, n=len(y), cases=nc, parent_n=len(parent), parent_cases=int(yparent.sum()),
                K=K, P=float(y.mean()), fitted_probit_variance=risk.covariate_variance,
                study_sensitivity_cv=float(risk.sensitivity.std()/risk.sensitivity.mean()),
                parent_context_metric=metric, pcgc_population_kernel_metric=pcgc_metric,
                context_mean=env_mean, context_sd=env_sd,
                sample_identity=array_sha256(study.row.to_numpy()), risk=risk.diagnostics(),
                pcgc=pcgc, pcgc_diagnostics=diagnostics, he=he, comparison=rows,
                phase_seconds=dict(nuisance=nuisance_seconds, pcgc=pcgc_seconds, he=he_seconds),
                elapsed_seconds=time.perf_counter()-stage, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            (args.out/(name+'.json')).write_text(json.dumps(record, default=json_value, indent=2)+'\n')
            print(json.dumps(dict(stage='trait_complete', trait=name, seconds=record['elapsed_seconds'], comparison=rows[0])), flush=True)
        (args.out/'complete.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
                                                             traits=args.traits))+'\n')


if __name__ == '__main__':
    main()
