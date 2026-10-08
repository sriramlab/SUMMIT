"""Off-diagonal liability GxE moments on SUMMIT's generalized context axes.

No projection or variance row is applied. See docs/pcgc_gxe.md for the scale,
ascertainment and external-reference contracts.
"""
from dataclasses import dataclass
import numpy as np

from summit.context.spec import ContextPairIndex, array_sha256
from summit.context.annotations import _jackknife_covariance
from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
from summit.sumstats.binary import finite_array, readonly
from .moments import annotations_array, solve, FROZEN_JACKKNIFE
from .reference import generalized_reference, reference_provenance, population_ld_reference

METHODS = ("pcgc", "pcgc-inverse", "pcgc-basis", "pcgc-ld")
EXTERNAL_CONTRACT = "risk_context_genotype_pair_factorization_v1"
SCALE_CONTRACT = "known_conditional_total_liability_sd_v1"


def context_pairs(q):
    return tuple((p.q, p.r) for p in ContextPairIndex(q).entries)


def pair_products(phi):
    return np.column_stack([(1 if q == r else 2)*phi[:, q]*phi[:, r]
                            for q, r in context_pairs(phi.shape[1])])


def liability_inputs(contexts, risk, liability_sd, method, *, basis=None, coefficients=None):
    phi = finite_array("contexts", contexts, 2)
    if phi.shape[0] != risk.n_samples or not 1 <= phi.shape[1] < len(phi):
        raise ValueError("contexts must have shape N by Q with 1 <= Q < N")
    norms = np.linalg.norm(phi, axis=0)
    if np.any(norms == 0) or np.linalg.matrix_rank(phi/norms) != phi.shape[1]:
        raise ValueError("context basis is rank deficient")
    if method not in METHODS:
        raise ValueError("contextual PCGC requires pcgc, pcgc-inverse, pcgc-basis or pcgc-ld")
    if liability_sd is None:
        raise ValueError("declare liability_sd explicitly; binary risks do not identify liability variance")
    sd = np.broadcast_to(finite_array("liability_sd", liability_sd), (len(phi),))
    if np.any(sd <= 0):
        raise ValueError("liability_sd must be positive on the sample axis")
    d = risk.sensitivity
    if method == "pcgc-basis":
        b = finite_array("risk basis", basis, 2)
        c = finite_array("risk basis coefficients", coefficients, 1)
        if b.shape != (len(phi), len(c)) or not len(c):
            raise ValueError("risk basis and coefficients must align to people")
        d_hat = b @ c
        # Check individual tails as well as the global norm; no approximation.
        if np.any(d_hat <= 0) or not np.allclose(d_hat, d, rtol=1e-10, atol=0) or np.linalg.norm(d_hat-d) > 1e-12*np.linalg.norm(d):
            raise ValueError("risk basis must span the sensitivity exactly")
        d = d_hat
    elif basis is not None or coefficients is not None:
        raise ValueError("risk basis arguments are only valid for pcgc-basis")
    w = d/sd
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        psi = phi if method == "pcgc-inverse" else w[:, None]*phi
        response = risk.z/w if method == "pcgc-inverse" else risk.z
        scores = psi*response[:, None]
        diagonal = pair_products(psi)*response[:, None]**2
        fourth = psi**4
    for name, value in (("context features", psi), ("score responses", scores),
                        ("diagonal responses", diagonal), ("fourth moments", fourth)):
        finite_array(name, value)
    return phi, sd, psi, scores, diagonal


def population_metric(contexts, risk, liability_sd, *, second_moment=None, total_variance=None):
    """IPW population metric; held fixed by SNP jackknife, not refitted."""
    from scipy.special import ndtri
    phi = np.asarray(contexts)
    sd = np.broadcast_to(np.asarray(liability_sd), (len(phi),))
    K, P = risk.population_prevalence, risk.sample_prevalence
    weights = np.where(risk.z > 0, K/P, (1-K)/(1-P))
    weights /= weights.sum()
    metric = phi.T @ (weights[:, None]*phi) if second_moment is None else second_moment
    # On the supplied liability scale, mu_i - threshold = sd_i Phi^-1(k_i).
    mu = sd*ndtri(risk.population_risk)
    variance = float(weights @ (sd**2+(mu-weights@mu)**2)) if total_variance is None else total_variance
    return metric, variance


def population_kernel_weights(contexts, risk):
    """IPW weights for E_population[t_annotation * phi phi.T]."""
    K, P = risk.population_prevalence, risk.sample_prevalence
    weights = np.where(risk.z > 0, K/P, (1-K)/(1-P))
    weights /= weights.sum()
    return np.asfortranarray(weights[:, None]*np.column_stack(
        [contexts[:, u]*contexts[:, v] for u, v in context_pairs(contexts.shape[1])]))


def _kernel_metric(summary, masses, q):
    return _omega((np.asarray(summary)/masses).T, q)


@dataclass(frozen=True)
class GxEMoments:
    annotations: np.ndarray
    ldscores: np.ndarray
    rhs_rows: np.ndarray
    num_contexts: int
    n_samples: int
    method: str
    population_second_moment: np.ndarray
    population_liability_variance: float
    population_kernel_second_moment: np.ndarray | None = None
    sampling_moments: object | None = None
    reference_probe_deviations: np.ndarray | None = None
    external_reference_covariance: np.ndarray | None = None
    external_context_gram: np.ndarray | None = None

    def __post_init__(self):
        if type(self.num_contexts) is not int or self.num_contexts < 1:
            raise ValueError("num_contexts must be a positive integer")
        if type(self.n_samples) is not int or self.n_samples <= self.num_contexts or self.method not in METHODS:
            raise ValueError("invalid contextual sample count or method")
        rhs = finite_array("context score rows", self.rhs_rows, 2)
        a = annotations_array(self.annotations, len(rhs))
        p = len(context_pairs(self.num_contexts))
        ld = finite_array("offdiagonal context LD rows", self.ldscores, 3)
        if rhs.shape != (len(a), p) or ld.shape != (len(a), p, a.shape[1]*p):
            raise ValueError("contextual PCGC moment axes disagree")
        metric = finite_array("population context second moment", self.population_second_moment, 2)
        if metric.shape != (self.num_contexts, self.num_contexts) or not np.allclose(metric, metric.T, rtol=1e-12, atol=1e-14):
            raise ValueError("population context metric must be symmetric Q by Q")
        if np.linalg.eigvalsh(metric).min() < -1e-12*max(1., np.linalg.norm(metric)):
            raise ValueError("population context metric must be positive semidefinite")
        variance = float(self.population_liability_variance)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("population liability variance must be finite and positive")
        for name, value in (("annotations", a), ("ldscores", ld), ("rhs_rows", rhs),
                            ("population_second_moment", metric)):
            object.__setattr__(self, name, readonly(value))
        object.__setattr__(self, "population_liability_variance", variance)
        if self.population_kernel_second_moment is not None:
            km = finite_array("population kernel second moment", self.population_kernel_second_moment, 3)
            if km.shape != (a.shape[1], self.num_contexts, self.num_contexts):
                raise ValueError("population kernel metric must have annotation by Q by Q shape")
            if not np.allclose(km, km.transpose(0,2,1), rtol=1e-12, atol=1e-14):
                raise ValueError("population kernel metric must be symmetric")
            if np.linalg.eigvalsh(km).min() < -1e-12*max(1.,np.linalg.norm(km)):
                raise ValueError("population kernel metric must be positive semidefinite")
            object.__setattr__(self, "population_kernel_second_moment", readonly(km))
        if self.sampling_moments is not None:
            from .sampling import SamplingMoments
            if not isinstance(self.sampling_moments,SamplingMoments) or self.sampling_moments.n_samples != self.n_samples or len(self.sampling_moments.linear) != self.component_count:
                raise ValueError("sampling covariance and genetic moment axes disagree")
        if self.reference_probe_deviations is not None:
            if self.sampling_moments is None:
                raise ValueError("reference-probe covariance requires sampling covariance moments")
            value = finite_array("reference probe deviations",self.reference_probe_deviations,3)
            if value.shape[0] < 2 or value.shape[1:] != (self.component_count,self.component_count):
                raise ValueError("reference probe covariance axes disagree")
            if np.linalg.norm(value.mean(0)) > 1e-11*max(1.,np.linalg.norm(value)):
                raise ValueError("reference probe deviations must be centered")
            object.__setattr__(self,"reference_probe_deviations",readonly(value))
        if self.external_reference_covariance is not None:
            if self.method != "pcgc-ld" or self.sampling_moments is None:
                raise ValueError("external reference covariance requires external-LD sampling moments")
            covariance = finite_array("external reference covariance",self.external_reference_covariance,2)
            gram = finite_array("external context Gram",self.external_context_gram,2)
            k = a.shape[1]
            if covariance.shape != (k*k,k*k) or gram.shape != (p,p):
                raise ValueError("external reference covariance axes disagree")
            if not np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-14) or not np.allclose(gram,gram.T,rtol=1e-12,atol=1e-14):
                raise ValueError("external reference covariance and context Gram must be symmetric")
            object.__setattr__(self,"external_reference_covariance",readonly(covariance))
            object.__setattr__(self,"external_context_gram",readonly(gram))
        elif self.external_context_gram is not None:
            raise ValueError("external context Gram requires reference covariance")

    @property
    def pairs(self):
        return context_pairs(self.num_contexts)

    @property
    def component_count(self):
        return self.annotations.shape[1]*len(self.pairs)

    def raw_equations(self, retained=None):
        if retained is None:
            retained = slice(None)
        else:
            retained = np.asarray(retained)
            if retained.dtype != bool or retained.shape != (len(self.annotations),):
                raise ValueError("retained must be a Boolean SNP mask")
        a = self.annotations[retained]
        mass = np.repeat(a.sum(axis=0), len(self.pairs))
        if np.any(mass <= 0):
            raise ValueError("variant deletion empties an annotation")
        ld = self.ldscores[retained]
        directed = (a.T @ ld.reshape(len(a), -1)).reshape(self.component_count, self.component_count)
        rhs = (a.T @ self.rhs_rows[retained]).reshape(-1)
        return directed, rhs, mass

    def equations(self, retained=None):
        directed, rhs, mass = self.raw_equations(retained)
        source_mass = np.repeat(self.annotations.sum(axis=0), len(self.pairs))
        return self.n_samples**2*directed/np.outer(mass, source_mass), rhs/mass


def _omega(theta, q):
    values = np.asarray(theta).reshape(-1, len(context_pairs(q)))
    out = np.zeros((len(values), q, q))
    for i, (u, v) in enumerate(context_pairs(q)):
        out[:, u, v] = out[:, v, u] = values[:, i]
    return out


def fit_gxe(moments, *, block_ids=None):
    """Estimate every annotation/context covariance; never PSD-clip estimates."""
    H, b = moments.equations()
    theta, diagnostics = solve(H, b)
    q = moments.num_contexts
    metric = moments.population_second_moment
    metrics = (np.tile(metric[None], (moments.annotations.shape[1],1,1))
               if moments.population_kernel_second_moment is None else moments.population_kernel_second_moment)
    contrast = np.array([(1 if u == v else 2)*item[u,v] for item in metrics for u,v in moments.pairs])
    denominator = moments.population_liability_variance
    omega = _omega(theta, q)
    result = dict(method=moments.method, components=theta.tolist(),
                  omega=omega.tolist(), omega_total=omega.sum(axis=0).tolist(),
                  pair_table=[list(p) for p in moments.pairs],
                  component_order="annotation_major_context_pair",
                  population_genetic_variance=float(contrast@theta),
                  population_heritability=float(contrast@theta/denominator),
                  population_liability_variance=denominator,
                  population_genetic_variance_metric=("unit_conditional_genotype_variance_assumption"
                    if moments.population_kernel_second_moment is None else "annotation_specific_population_kernel_diagonal"),
                  normal_condition=float(np.linalg.cond(H)),
                  normal_relative_asymmetry=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H), np.finfo(float).tiny)),
                  minimum_normal_eigenvalue=float(diagnostics.eigenvalues[0]),
                  omega_minimum_eigenvalues=np.linalg.eigvalsh(omega).min(axis=1).tolist(),
                  relative_solve_residual=float(np.linalg.norm(H@theta-b)/max(1., np.linalg.norm(b))),
                  estimates_constrained=False, uncertainty_status="not_requested")
    if block_ids is not None:
        ids = np.asarray(block_ids)
        if ids.dtype.kind not in "iu" or ids.shape != (len(moments.annotations),):
            raise ValueError("block IDs must be integers on the SNP axis")
        labels, ids, sizes = np.unique(ids, return_inverse=True, return_counts=True)
        if len(labels) < 2:
            raise ValueError("jackknife requires at least two nonempty blocks")
        directed, raw_rhs, mass = moments.raw_equations()
        bd, bm, reconstruction = reduce_generalized_gxe_reference_for_inference(
            directional_ldscores=moments.ldscores, annotations=moments.annotations,
            variant_block_ids=ids, block_labels=tuple(map(str, labels)),
            expected_directed_numerator=directed)
        br = np.empty((len(labels), moments.component_count))
        p = len(moments.pairs)
        for a in range(moments.annotations.shape[1]):
            for j in range(p):
                br[:, a*p+j] = np.bincount(ids, weights=moments.annotations[:, a]*moments.rhs_rows[:, j], minlength=len(labels))
        replicates = []
        for dm, dH, db in zip(bm, bd, br):
            retained_mass = mass-np.repeat(dm, p)
            if np.any(retained_mass <= 0):
                raise ValueError("jackknife deletion empties an annotation")
            matrix = moments.n_samples**2*(directed-dH)/np.outer(retained_mass, mass)
            replicates.append(solve(matrix, (raw_rhs-db)/retained_mass, deletion=True)[0])
        covariance = _jackknife_covariance(np.asarray(replicates))
        se = np.sqrt(np.maximum(0., covariance.diagonal()))
        variance_se = float(np.sqrt(max(0., contrast@covariance@contrast)))
        result.update(uncertainty_status="estimated", covariance=covariance.tolist(),
                      standard_errors=se.tolist(), omega_standard_errors=_omega(se, q).tolist(),
                      population_genetic_variance_se=variance_se,
                      population_heritability_se=variance_se/denominator,
                      jackknife_replicates=np.asarray(replicates).tolist(),
                      jackknife_blocks=len(labels), jackknife_block_sizes=sizes.tolist(),
                      jackknife_method=FROZEN_JACKKNIFE,
                      jackknife_reconstruction_error=float(reconstruction),
                      uncertainty_conditioning="fixed_risks_liability_sd_population_metric_genotype_scale_reference_probes",
                      external_reference_uncertainty_included=False)
    if moments.sampling_moments is not None:
        from .sampling import sampling_inference
        if "covariance" in result:
            result["snp_block_covariance"] = result["covariance"]
            result["snp_block_standard_errors"] = result["standard_errors"]
        result.update(sampling_inference(moments,theta,H,contrast))
        result["omega_standard_errors"] = _omega(result["standard_errors"],q).tolist()
    return result


def evaluate_contexts(result, contexts):
    """Genetic variance/covariance at supplied contexts, with linear JK SEs.

    These are genetic covariances on the declared liability scale; dividing by
    context-specific total liability variances is a separate interpretation.
    """
    phi = finite_array("evaluation contexts", contexts, 2)
    omega = finite_array("omega", result["omega"], 3)
    if phi.shape[1] != omega.shape[1]:
        raise ValueError("evaluation context dimension disagrees with fitted omega")
    values = phi @ omega.sum(axis=0) @ phi.T
    output = {"genetic_covariance": values.tolist()}
    if "covariance" in result:
        pairs = context_pairs(phi.shape[1])
        linear = np.stack([np.outer(phi[:, u], phi[:, v]) if u == v else
                           np.outer(phi[:, u], phi[:, v])+np.outer(phi[:, v], phi[:, u])
                           for u, v in pairs], axis=-1)
        linear = np.tile(linear, (1, 1, len(omega)))
        cov = np.asarray(result["covariance"])
        variances = np.einsum("ijc,cd,ijd->ij", linear, cov, linear, optimize=True)
        output["standard_errors"] = np.sqrt(np.maximum(0., variances)).tolist()
    return output


def _quartic_plan(q):
    pairs = context_pairs(q)
    monomials = sorted({tuple(sorted(a+b)) for a in pairs for b in pairs})
    indices = {term: j for j, term in enumerate(monomials)}
    return monomials, np.array([[indices[tuple(sorted(a+b))] for b in pairs] for a in pairs])



def _gxe_reserved_bytes(n, m, q, k):
    from math import comb
    p = q*(q+1)//2
    reserve = 8*(3*m*p*k*p + 3*m*p + 3*m*k + n*(4*q+2*p+comb(q+3,4)) + 4*(k*p)**2)
    return reserve + (reserve+6)//7 + 64*1024**2


def plan_gxe_reference(*, num_samples, num_variants, num_contexts, num_annotations,
                       probes=256, memory_bytes=2**30, block_size=256, threads=1,
                       genotype_format="bed", sampling_partners=0, risk_rank=0, architecture_probes=0):
    """Dimension-only workspace admission; no genotypes, risks, or arrays."""
    from math import comb
    from dataclasses import replace
    from summit.ldscore.generalized_gxe_variant import GeneralizedGxEPlanInputs, plan_generalized_gxe_variant_work
    n, m, q, k = num_samples, num_variants, num_contexts, num_annotations
    inputs = GeneralizedGxEPlanInputs(num_samples=n, num_variants=m, num_basis=q,
        num_annotations=k, num_probes=probes, memory_limit_bytes=memory_bytes,
        threads=threads, genotype_format=genotype_format,
        preferred_variant_block_width=block_size,
        preferred_rhs_tile_columns=q*q*min(probes,64), rhs_policy="tiled")
    p, t, b = q*(q+1)//2, comb(q+3,4), min(block_size,m)
    reserve = _gxe_reserved_bytes(n,m,q,k)
    if sampling_partners:
        from .sampling import sampling_workspace_bytes
        if type(sampling_partners) is not int or sampling_partners < 2:
            raise ValueError("sampling partners must be zero or an integer >=2")
        reserve += sampling_workspace_bytes(n,m,q,k,sampling_partners,block_size,risk_rank,architecture_probes)
        reserve += 8*(4*probes*(k*p)**2+2*b*min(probes,64)+k*min(probes,64))
    elif architecture_probes:
        raise ValueError("architecture inference requires individual-sampling partners")
    score = 8*((n+m)*(q+p)+n*b+n*k+m*t*k+n*t+(n+b)*(p+t*k)+p*(n+k))
    output = 8*m*p*k*p
    budget = memory_bytes-reserve-score-output
    if budget <= 0:
        raise MemoryError("contextual PCGC memory cannot accommodate output and score buffers")
    plan = plan_generalized_gxe_variant_work(replace(inputs, memory_limit_bytes=budget))
    return dict(reference_plan=plan.to_dict(), reserved_result_and_input_bytes=reserve,
                score_buffer_bytes=score, resident_directional_output_bytes=output,
                peak_planned_workspace_bytes=reserve+score+plan.peak_resident_bytes,
                requested_memory_bytes=memory_bytes)


def prepare_gxe_moments(operator, annotations, risk, contexts, method="pcgc", *,
                        liability_sd, basis=None, coefficients=None,
                        population_second_moment=None, population_liability_variance=None,
                        population_kernel_second_moment=None,
                        sampling_partners=0, sampling_seed=0, risk_covariates=None, architecture_probes=0,
                        memory_bytes=2**30, **options):
    """Two passes, variant-axis probes, exact same-person removal in every row."""
    if method == "pcgc-ld":
        raise ValueError("pcgc-ld requires prepare_gxe_external with an independent reference")
    if operator.num_samples != risk.n_samples:
        raise ValueError("risk and genotype sample axes disagree")
    phi, sd, psi, responses, diagonal = liability_inputs(contexts, risk, liability_sd, method,
                                                        basis=basis, coefficients=coefficients)
    a = annotations_array(annotations, operator.num_variants)
    n, m, q, k = len(phi), len(a), phi.shape[1], a.shape[1]
    p = q*(q+1)//2
    monomials, indices = _quartic_plan(q)
    # Reserve the resident engine output, ownership copies, and our inputs.
    # The generalized engine's planner budgets its source/scratch separately.
    reserve = _gxe_reserved_bytes(n, m, q, k)
    sampler = None
    probe_collector = None
    risk_rank = 0 if risk_covariates is None else finite_array("risk inference covariates",risk_covariates,2).shape[1]+1
    if sampling_partners:
        from .sampling import SamplingOperator,sampling_workspace_bytes
        if type(sampling_partners) is not int or sampling_partners < 2:
            raise ValueError("sampling partners must be zero or an integer >=2")
        reserve += sampling_workspace_bytes(n,m,q,k,sampling_partners,options.get("block_size",256),risk_rank,architecture_probes)
        probes = options.get("probes",256)
        reserve += 8*(4*probes*(k*p)**2+2*min(m,options.get("block_size",256))*min(probes,64)+k*min(probes,64))
    elif risk_covariates is not None:
        raise ValueError("risk inference covariates require individual-sampling inference")
    elif architecture_probes:
        raise ValueError("architecture inference requires individual-sampling partners")
    if not isinstance(memory_bytes, int) or memory_bytes <= reserve:
        raise MemoryError("contextual PCGC memory cannot accommodate output and score buffers")
    plan_gxe_reference(num_samples=n, num_variants=m, num_contexts=q, num_annotations=k,
        probes=options.get("probes",256), memory_bytes=memory_bytes,
        block_size=options.get("block_size",256), threads=options.get("threads",1),
        genotype_format=operator.genotype_format,sampling_partners=sampling_partners,risk_rank=risk_rank,architecture_probes=architecture_probes)
    if sampling_partners:
        from .reference import ProbeGramCollector
        probe_collector = ProbeGramCollector(a,q,options.get("probes",256),n)
        response = risk.z/(risk.sensitivity/sd) if method == "pcgc-inverse" else risk.z
        sampler = SamplingOperator(operator,a,phi,risk,sd,psi,response,method,
            partners=sampling_partners,seed=sampling_seed,risk_covariates=risk_covariates,
            threads=options.get("threads",1),native=options.get("native",True),
            architecture_probes=architecture_probes,
            estimate_population_metric=population_kernel_second_moment is None and population_second_moment is None,
            estimate_population_variance=population_liability_variance is None)
        operator = sampler
    fourth = np.column_stack([np.prod(psi[:, term], axis=1) for term in monomials])
    finite_array("context fourth products", fourth)
    ref, scored, plan = generalized_reference(operator, a, psi, responses=responses,
        diagonal_responses=diagonal, diagonal_weights=fourth, collect_diagonal_rows=True,
        kernel_summary_weights=population_kernel_weights(phi,risk),
        probe_product_sink=probe_collector,
        memory_bytes=memory_bytes-reserve, admit_resident_output=True, **options)
    ld = np.array(ref.directional_ldscores, copy=True)
    factors = np.array([1 if u == v else 2 for u, v in ref.pair_table])
    # The source same-person diagonal is accumulated in pass 1, then scored
    # in pass 2. A full same-person Gram subtraction alone is wrong for JK.
    for target in range(p):
        for source in range(p):
            correction = scored.reference_diagonal_rows[:, indices[target, source], :]
            ld[:, target, source::p] -= factors[target]*factors[source]*correction/n**2
    rhs = np.column_stack([factors[j]*scored.scores[:, u]*scored.scores[:, v]-scored.diagonals[:, j]
                           for j, (u, v) in enumerate(ref.pair_table)])
    metric, variance = population_metric(phi, risk, sd, second_moment=population_second_moment,
                                         total_variance=population_liability_variance)
    km = population_kernel_second_moment
    if km is None and population_second_moment is None:
        km = _kernel_metric(scored.kernel_summary, a.sum(0), q)
    sampling = sampler.finalize() if sampler is not None else None
    moments = GxEMoments(a, ld, rhs, q, n, method, metric, variance, km, sampling,
                         None if probe_collector is None else probe_collector.deviations())
    diagnostics = dict(genotype_passes=operator.observed_passes, reference_kind="study_specific_context_variant_probes",
                       liability_scale_contract=SCALE_CONTRACT, liability_sd_range=[float(sd.min()), float(sd.max())],
                       context_sha256=array_sha256(phi), liability_sd_sha256=array_sha256(sd),
                       feature_sha256=array_sha256(psi), reference_execution=reference_provenance(dict(options, memory_bytes=memory_bytes)),
                       maximum_absolute_score_response=float(np.max(np.abs(responses))),
                       risk_sensitivity_range=[float(risk.sensitivity.min()), float(risk.sensitivity.max())],
                       reference_plan=plan.to_dict(),
                       peak_planned_workspace_bytes=reserve+plan.peak_resident_bytes+scored.pcgc_score_buffer_bytes,
                       exact_same_person_per_target_snp=True,
                       population_metric_source="inverse_ascertainment" if population_second_moment is None else "supplied",
                       population_kernel_metric_source=("supplied" if population_kernel_second_moment is not None else
                           "inverse_ascertainment_adjusted_genotype_diagonals" if km is not None else "unit_conditional_genotype_variance_assumption"),
                       population_variance_source="inverse_ascertainment" if population_liability_variance is None else "supplied")
    if sampler is not None:
        diagnostics["sampling_inference"] = sampler.diagnostics
        diagnostics["sampling_inference"]["reference_probe_uncertainty_included"] = True
    return moments, diagnostics


def context_pair_gram(psi):
    """Sum over i != j of context kernel products, without an N by N matrix."""
    gram = psi.T@psi
    diagonal = pair_products(psi)
    pairs = context_pairs(psi.shape[1])
    H = np.empty((len(pairs), len(pairs)))
    for p, (u, v) in enumerate(pairs):
        left = ((u, v),) if u == v else ((u, v), (v, u))
        for s, (c, d) in enumerate(pairs):
            right = ((c, d),) if c == d else ((c, d), (d, c))
            H[p, s] = sum(gram[i, k]*gram[j, l] for i, j in left for k, l in right)
    return H-diagonal.T@diagonal


def prepare_gxe_external(operator, reference_operator, annotations, risk, contexts, *,
                         liability_sd, factorization_contract,
                         population_second_moment=None, population_liability_variance=None,
                         population_kernel_second_moment=None,
                         sampling_partners=0, sampling_seed=0, risk_covariates=None, architecture_probes=0,
                         memory_bytes=2**30, block_size=256, threads=1, native=True, **options):
    """External-LD approximation with two complete reference traversals.

    Study scoring uses one traversal, or two with sampling inference.
    Caller must align SNPs/alleles/scales and exclude overlapping people.
    prepare_gxe_from_source enforces these identities for file-backed inputs.
    """
    if set(options)-{"probes", "seed"}:
        raise ValueError("external contextual reference only accepts probes and seed as reference options")
    if factorization_contract != EXTERNAL_CONTRACT:
        raise ValueError("pcgc-ld requires an explicit risk/context/genotype pair factorization contract")
    if operator.num_samples != risk.n_samples or operator.num_variants != reference_operator.num_variants:
        raise ValueError("external reference/study dimensions disagree")
    if operator.genotype_scale_id != reference_operator.genotype_scale_id:
        raise ValueError("external reference/study population genotype scales disagree")
    phi, sd, psi, responses, diagonal = liability_inputs(contexts, risk, liability_sd, "pcgc-ld")
    a = annotations_array(annotations, operator.num_variants)
    n, m, q, k = len(phi), len(a), phi.shape[1], a.shape[1]
    p = q*(q+1)//2
    reserve = 8*(3*m*p*k*p + 3*m*p + 3*m*k + n*(3*q+3*p+4*min(block_size,m)) + 4*(k*p)**2 + p*k + min(block_size,m)*p)
    reserve += (reserve+6)//7 + 64*1024**2
    sampler = None
    probe_collector = None
    reference_sampler = None
    risk_rank = 0 if risk_covariates is None else finite_array("risk inference covariates",risk_covariates,2).shape[1]+1
    if sampling_partners:
        from .sampling import SamplingOperator,sampling_workspace_bytes
        if type(sampling_partners) is not int or sampling_partners < 2:
            raise ValueError("sampling partners must be zero or an integer >=2")
        reserve += sampling_workspace_bytes(n,m,q,k,sampling_partners,block_size,risk_rank,architecture_probes)
        probes = options.get("probes",256)
        reserve += 8*(4*probes*(k*p)**2+4*probes*k*k)
        from .reference_sampling import reference_sampling_workspace_bytes
        reserve += reference_sampling_workspace_bytes(reference_operator.num_samples,k,sampling_partners,block_size)
    elif risk_covariates is not None:
        raise ValueError("risk inference covariates require individual-sampling inference")
    elif architecture_probes:
        raise ValueError("architecture inference requires individual-sampling partners")
    if not isinstance(memory_bytes, int) or memory_bytes <= reserve:
        raise MemoryError("external contextual PCGC memory cannot accommodate output and score buffers")
    from .rank_one import plan_pcgc_reference
    # Admit the external pass and all retained study buffers before study I/O.
    plan_pcgc_reference(num_samples=reference_operator.num_samples, num_variants=m,
        num_annotations=k, probes=options.get("probes", 256), memory_bytes=memory_bytes-reserve,
        block_size=block_size, threads=threads, genotype_format=reference_operator.genotype_format)
    if sampling_partners:
        from .reference import ProbeGramCollector
        probe_collector = ProbeGramCollector(a,1,options.get("probes",256),reference_operator.num_samples)
        sampler = SamplingOperator(operator,a,phi,risk,sd,psi,risk.z,"pcgc-ld",
            partners=sampling_partners,seed=sampling_seed,risk_covariates=risk_covariates,
            threads=threads,native=native,
            architecture_probes=architecture_probes,
            estimate_population_metric=population_kernel_second_moment is None and population_second_moment is None,
            estimate_population_variance=population_liability_variance is None)
        operator = sampler
        from .reference_sampling import ReferenceSamplingOperator
        reference_sampler = ReferenceSamplingOperator(reference_operator,a,partners=sampling_partners,
            seed=sampling_seed,threads=threads,native=native)
        reference_operator = reference_sampler
    from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator, NumpyTNOperator
    if native:
        from summit.prediction.genotype import native_module
        from summit.prediction.runtime import configure_prediction_threads
        configure_prediction_threads(native_module(), threads)
    tn = ProtectedTNOperator(threads=threads) if native else NumpyTNOperator(threads=threads)
    rhs = np.empty((m, p))
    population_weights = population_kernel_weights(phi,risk)
    kernel_summary = np.zeros((p,k))
    operator.begin_pass(1)
    for start in range(0, m, block_size):
        stop = min(m, start+block_size)
        x = operator.read_block(start, stop).values
        scores = tn.matmul_tn(x, np.asfortranarray(responses))
        diag = tn.matmul_tn(np.asfortranarray(x*x), np.asfortranarray(diagonal))
        kernel_summary += tn.matmul_tn(np.asfortranarray(x*x),population_weights).T@a[start:stop]
        for j, (u, v) in enumerate(context_pairs(q)):
            rhs[start:stop, j] = (1 if u == v else 2)*scores[:, u]*scores[:, v]-diag[:, j]
    operator.finish_pass()
    pop, plan = population_ld_reference(reference_operator, a, memory_bytes=memory_bytes-reserve,
                                        block_size=block_size, threads=threads, native=native,
                                        probe_square_sink=None if probe_collector is None else probe_collector.add_squared,**options)
    factor = context_pair_gram(psi)
    ld = (pop[:, None, :, None]*factor[None, :, None, :]/n**2).reshape(m, p, k*p)
    sampling = None
    if sampler is not None:
        operator.begin_pass(2)
        for start in range(0,m,block_size):
            operator.read_block(start,min(m,start+block_size))
        operator.finish_pass()
        annotation_gram = (a.T@pop)/np.outer(a.sum(0),a.sum(0))
        sampling = sampler.finalize(reference_annotation_gram=annotation_gram)
    metric, variance = population_metric(phi, risk, sd, second_moment=population_second_moment,
                                         total_variance=population_liability_variance)
    km = population_kernel_second_moment
    if km is None and population_second_moment is None:
        km = _kernel_metric(kernel_summary, a.sum(0), q)
    deviations = None
    if probe_collector is not None:
        nr = reference_operator.num_samples
        annotation_deviations = probe_collector.deviations()/(nr*(nr-1))
        deviations = np.einsum('lab,ps->lapbs',annotation_deviations,factor).reshape(-1,k*p,k*p)
    moments = GxEMoments(a, ld, rhs, q, n, "pcgc-ld", metric, variance, km, sampling,deviations,
        None if reference_sampler is None else reference_sampler.covariance(),
        None if reference_sampler is None else factor)
    diagnostics = dict(reference_kind="external_population_risk_context_factorization_approximation",
                       factorization_contract=EXTERNAL_CONTRACT, study_genotype_passes=operator.observed_passes,
                       reference_genotype_passes=reference_operator.observed_passes,
                       reference_samples=reference_operator.num_samples, liability_scale_contract=SCALE_CONTRACT,
                       context_sha256=array_sha256(phi), liability_sd_sha256=array_sha256(sd),
                       liability_sd_range=[float(sd.min()), float(sd.max())],
                       reference_execution=reference_provenance(dict(options, threads=threads, native=native, memory_bytes=memory_bytes)),
                       peak_planned_workspace_bytes=reserve+plan.peak_resident_bytes,
                       reference_plan=plan.to_dict(), exact_same_person_per_target_snp=True)
    diagnostics["population_kernel_metric_source"] = ("supplied" if population_kernel_second_moment is not None else
        "inverse_ascertainment_adjusted_genotype_diagonals" if km is not None else "unit_conditional_genotype_variance_assumption")
    if sampler is not None:
        diagnostics["sampling_inference"] = sampler.diagnostics
        diagnostics["sampling_inference"]["reference_probe_uncertainty_included"] = True
        diagnostics["sampling_inference"]["external_reference_uncertainty_included"] = True
    return moments, diagnostics
