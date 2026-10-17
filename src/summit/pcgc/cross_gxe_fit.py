"""Directed joint fits and conditional SNP-block/reference-draw uncertainty."""
from time import perf_counter
import numpy as np
from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes
from summit.context.annotations import _jackknife_covariance
from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
from .moments import solve

CALIBRATION_SCHEMA = "summit.pcgc.cross_interval_calibration.v2"
# Rare outcomes or h N / M at or above one: the regimes in which the
# independent review and the original calibration found within-trait binary
# SNP-block variance deficits (h is the sample liability genetic variance).
LARGE_SIGNAL_RATIO = 1.
CROSS_TRAIT_BINARY_NOTE = ("threshold heteroscedasticity affects cross-trait components only mildly "
                           "(independent review: variance ratio 0.87-0.95)")


def omega(theta, pair, qa, qb):
    result = np.zeros((len(theta)//pair.p, qa, qb))
    for p, orientations in enumerate(pair.layout):
        for u,v in orientations:
            result[:,u,v] = np.asarray(theta).reshape(-1,pair.p)[:,p]
    return result


def equations(moments, pair, retained=None):
    selected = slice(None) if retained is None else retained
    a = moments.annotations[selected]
    mass = np.repeat(a.sum(0),pair.p)
    source_mass = np.repeat(moments.annotations.sum(0),pair.p)
    if np.any(mass <= 0):
        raise ValueError("target deletion empties an annotation")
    directed = (a.T@pair.ldscores[selected].reshape(len(a),-1)).reshape(len(mass),len(mass))
    rhs = (a.T@pair.rhs_rows[selected]).reshape(-1)
    return pair.sample_product*directed/np.outer(mass,source_mass),rhs/mass


def plan_cross_fit(moments, blocks, memory_bytes=None):
    """Admit resident inputs, block reductions and joint uncertainty outputs."""
    sizes=[p.p*moments.annotations.shape[1] for p in moments.pairs]
    dimension=sum(sizes)
    probes=max(len(p.probe_grams) for p in moments.pairs)
    largest=max(sizes)
    baseline=_peak_rss_bytes()
    # The reducer can copy target rows. Python/JSON versions of covariance and
    # delete-block arrays coexist with NumPy storage at return/serialization.
    row_copy=max(p.ldscores.nbytes for p in moments.pairs)
    workspace=row_copy+8*(4*blocks*largest**2+16*dimension**2+
                           12*blocks*dimension+4*probes*dimension+4*probes*largest**2+8*len(moments.annotations))
    peak=baseline+workspace
    if memory_bytes is not None:
        if type(memory_bytes) is not int or memory_bytes<1:
            raise ValueError("fit memory budget must be a positive integer")
        if peak>memory_bytes:
            raise MemoryError("cross-trait fit rows and joint uncertainty exceed memory budget")
    return dict(baseline_rss_bytes=baseline,fit_workspace_bytes=workspace,
                peak_planned_bytes=peak,memory_budget_bytes=memory_bytes)


def _reference_inverse_diagnostics(h, deviations, theta):
    """Plug-in Neumann expansion; diagnostics only, no bias correction.

    R_b = H^-1 (H_b - mean(H_b)). With B independent common probe draws,
    sum ||R_b||_F^2 / [B(B-1)] estimates squared relative error of the mean.
    The next inverse term is sum R_b R_b theta / [B(B-1)]. Neither expression
    guarantees accuracy when the inverse perturbation is large.
    """
    b,c,_=deviations.shape
    relative=np.linalg.solve(h,deviations.transpose(1,0,2).reshape(c,-1)).reshape(c,b,c).transpose(1,0,2)
    denominator=b*(b-1)
    rms=np.sqrt(np.einsum("bij,bij->",relative,relative)/denominator)
    impact=np.einsum("bij,j->bi",relative,theta)
    bias=np.einsum("bij,bj->i",relative,impact)/denominator
    return dict(reference_inverse_rms_perturbation=float(rms),
                reference_inverse_second_order_bias=bias.tolist())


def calibration_status(moments, pairs, *, modeled_pairs=()):
    """Interval calibration flags; diagnostics only, no variance correction.

    For a binary trait, Var(z | g) depends on the genetic liability (threshold
    heteroscedasticity, delta ~ (eta^2-1)/2 at eta = Phi^-1(K)). This
    correlates the signal-by-noise parts of the estimating function across
    SNP blocks, Cov(u_b, u_b') ~ 2 delta h_b h_b'. The delete-block jackknife
    assumes independent blocks and omits that covariance, which grows with N / M.
    Simulations at array scale (N/M <= 1, including real UK Biobank genotypes,
    K = 0.02) found the jackknife calibrated, so a rare outcome alone is not
    flagged (schema v2). Within-trait binary components are marked uncalibrated
    when h N / M >= 1, and when supplied risks were kept under own-case-status
    sampling. Contrasts that use them inherit the flag.
    modeled_pairs lists within-trait pairs whose omitted covariance a supplied
    participant-level term models (see fit_cross_gxe).
    """
    variants = len(moments.annotations)
    size = max(pairs[f"{p.left}:{p.right}"]["coefficient_slice"][1] for p in moments.pairs)
    components = ["not_flagged"]*size
    by_pair = {}
    for pair in moments.pairs:
        key = f"{pair.left}:{pair.right}"
        kinds = [moments.traits[t]["metadata"]["kind"] for t in (pair.left, pair.right)]
        entry = dict(status="not_flagged", reasons=[], notes=[])
        if pair.left == pair.right and kinds[0] == "binary":
            meta = moments.traits[pair.left]["metadata"]
            prevalence = meta.get("reference_prevalence")
            second = meta.get("context_second_moment")
            h = None if second is None else max(0., float(np.sum(
                np.asarray(pairs[key]["omega_total"])*np.asarray(second, dtype=float))))
            ratio = None if h is None else h*meta["sample_size"]/variants
            triggers = []
            if ratio is None or ratio >= LARGE_SIGNAL_RATIO:
                triggers.append("large_h_N_over_M")
            entry.update(reference_prevalence=prevalence, sample_liability_genetic_variance=h,
                         sample_size=meta["sample_size"], variants=variants, h_N_over_M=ratio,
                         heteroscedasticity_triggers=triggers)
            if triggers and key not in modeled_pairs:
                entry["reasons"].append("SNP_block_variance_omits_threshold_heteroscedasticity_covariance")
            if meta.get("selection") == "own_case_status" and meta.get("risk_source") == "supplied":
                entry["reasons"].append("supplied_risks_under_case_status_sampling")
            if key in modeled_pairs:
                entry["notes"].append("participant_covariance_term_supplied")
        elif pair.left != pair.right and "binary" in kinds:
            entry["notes"].append(CROSS_TRAIT_BINARY_NOTE)
        if entry["reasons"]:
            entry["status"] = "uncalibrated"
            begin, end = pairs[key]["coefficient_slice"]
            components[begin:end] = ["uncalibrated"]*(end-begin)
        by_pair[key] = entry
    return dict(schema=CALIBRATION_SCHEMA, components=components, pairs=by_pair,
                large_h_N_over_M_threshold=LARGE_SIGNAL_RATIO,
                scope=("flags known interval deficits only; not_flagged does not establish calibration "
                       "outside the validated settings"))


def _participant_term(term, moments, pairs, size):
    """Validate a supplied participant-level covariance for within-trait binary pairs."""
    if not isinstance(term, dict) or set(term) != {"method", "pairs", "covariance"}:
        raise ValueError("participant covariance needs method, pairs and covariance")
    method, keys = term["method"], tuple(term["pairs"])
    if not isinstance(method, str) or not method:
        raise ValueError("participant covariance needs a method name")
    eligible = {f"{p.left}:{p.right}" for p in moments.pairs
                if p.left == p.right and moments.traits[p.left]["metadata"]["kind"] == "binary"}
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= eligible:
        raise ValueError("participant covariance applies to declared within-trait binary pairs only")
    value = np.asarray(term["covariance"], dtype=float)
    if value.shape != (size, size) or not np.isfinite(value).all() or not np.allclose(value, value.T, rtol=1e-12, atol=0):
        raise ValueError("participant covariance must be a finite symmetric matrix on the component axis")
    mask = np.zeros(size, dtype=bool)
    for key in keys:
        begin, end = pairs[key]["coefficient_slice"]
        mask[begin:end] = True
    if np.any(value[~mask]) or np.any(value[:, ~mask]):
        raise ValueError("participant covariance must be zero outside its declared pairs")
    if np.linalg.eigvalsh(value).min() < -1e-12*max(1., float(np.abs(value).max())):
        raise ValueError("participant covariance must be positive semidefinite")
    return value, keys, method


def fit_cross_gxe(moments, *, block_ids=None, memory_bytes=None, confidence=.95,
                  participant_covariance=None):
    """No genotype access; reference draws and deletion IDs are shared.

    participant_covariance is an extension point for a calibrated
    participant-level sampling covariance of within-trait binary components:
    the between-block covariance the SNP-block jackknife omits (see
    calibration_status). It is a mapping with "method" (name), "pairs"
    (within-trait binary pair keys such as "0:0") and "covariance" (a
    component-axis matrix that is zero outside those pairs). It is added to
    the joint covariance as a known term and clears the heteroscedasticity
    flag for those pairs. The supplier must exclude variation the jackknife
    already captures. No such covariance is computed here.
    """
    start = perf_counter()
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly inside (0,1)")
    moments.validate()
    a = moments.annotations
    labels = ids = None
    if block_ids is not None:
        ids = np.asarray(block_ids)
        if ids.dtype.kind not in "iu" or ids.shape != (len(a),):
            raise ValueError("SNP block IDs must be integers on the common variant axis")
        labels,ids = np.unique(ids,return_inverse=True)
        if len(labels) < 2:
            raise ValueError("at least two nonempty SNP blocks are required")
    plan=plan_cross_fit(moments,0 if labels is None else len(labels),memory_bytes)
    results, point, deleted, mc_actions = {}, [], [], []
    offset = 0
    for pair in moments.pairs:
        H,b = equations(moments,pair)
        theta,diag = solve(H,b)
        count=len(theta)
        qa,qb=[len(moments.traits[i]["context_names"]) for i in (pair.left,pair.right)]
        matrix=omega(theta,pair,qa,qb)
        key=f"{pair.left}:{pair.right}"
        condition=float(np.linalg.cond(H))
        results[key]=dict(left=pair.left,right=pair.right,omega=matrix.tolist(),
            omega_total=matrix.sum(0).tolist(),coefficient_slice=[offset,offset+count],
            pair_table=[list(p[0]) for p in pair.layout],overlap_count=pair.overlap,
            normal_condition=condition,weak_normal_equations=condition>=1e7,
            minimum_symmetric_normal_eigenvalue=float(diag.eigenvalues[0]),
            normal_relative_asymmetry=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H),np.finfo(float).tiny)),
            solve_residual=float(np.linalg.norm(H@theta-b)/max(1.,np.linalg.norm(b))))
        if pair.left == pair.right:
            results[key]["omega_minimum_eigenvalues"]=np.linalg.eigvalsh(matrix).min(1).tolist()
        dev=pair.probe_grams-pair.probe_grams.mean(0)
        mc_actions.append(-np.linalg.solve(H,np.einsum("bij,j->bi",dev,theta).T).T)
        results[key].update(_reference_inverse_diagnostics(H,dev,theta))
        results[key]["relative_reference_mc_frobenius_se"]=float(
            np.sqrt(np.sum(dev**2)/(len(dev)*(len(dev)-1)))/np.linalg.norm(H))
        if labels is not None:
            raw=(a.T@pair.ldscores.reshape(len(a),-1)).reshape(count,count)
            raw_rhs=(a.T@pair.rhs_rows).reshape(-1)
            bd,bm,_=reduce_generalized_gxe_reference_for_inference(
                directional_ldscores=pair.ldscores,annotations=a,variant_block_ids=ids,
                block_labels=tuple(map(str,labels)),expected_directed_numerator=raw)
            mass=np.repeat(a.sum(0),pair.p)
            br=np.empty((len(labels),count))
            for k in range(a.shape[1]):
                for p in range(pair.p):
                    br[:,k*pair.p+p]=np.bincount(ids,weights=a[:,k]*pair.rhs_rows[:,p],minlength=len(labels))
            loo=[]
            for dm,dH,db in zip(bm,bd,br):
                retained=mass-np.repeat(dm,pair.p)
                if np.any(retained<=0):
                    raise ValueError("target deletion empties an annotation")
                h=pair.sample_product*(raw-dH)/np.outer(retained,mass)
                loo.append(solve(h,(raw_rhs-db)/retained,deletion=True)[0])
            deleted.append(np.asarray(loo))
        point.append(theta)
        offset+=count
    point=np.concatenate(point)
    actions=np.concatenate(mc_actions,axis=1)
    mc=actions.T@actions/(len(actions)*(len(actions)-1))
    covariance=mc.copy()
    result=dict(kind="summit.pcgc.cross_context_fit",schema_version=1,fit_memory_plan=plan,
        traits=list(moments.traits),pairs=results,components=point.tolist(),
        reference_mc_covariance=mc.tolist(),estimates_constrained=False,
        reference_probe_count=len(actions),jackknife_block_count=0 if labels is None else len(labels),
        confidence_level=float(confidence),
        uncertainty_status="reference_mc_only",
        uncertainty_limitations=[
            "first_order_liability_moments",
            "SNP_blocks_require_weak_between_block_dependence_and_no_dominant_block",
            "no_separate_model_based_response_or_SNP_effect_covariance",
            "no_fitted_risk_or_estimated_population_metric_influence",
            "external_prevalence_genotype_scale_and_liability_SD_fixed",
            "finite_probe_inverse_bias_not_removed",
            "within_trait_binary_SNP_block_variance_omits_threshold_heteroscedasticity_see_calibration",
            "reference_noise_partly_counted_in_both_SNP_block_and_reference_terms_conservative",
            "selection_law_declared_not_verified",
            "validated_only_up_to_M_8192_and_N_32768_without_long_range_LD",
            "interval_calibration_not_established"])
    if labels is not None:
        block_mass=np.column_stack([np.bincount(ids,weights=column,minlength=len(labels)) for column in a.T])
        result["block_diagnostics"]=dict(
            count=len(labels),few_blocks=len(labels)<30,
            annotation_occupied_blocks=np.sum(block_mass>0,axis=0).tolist(),
            annotation_maximum_mass_fraction=np.max(block_mass/a.sum(0),axis=0).tolist(),
            maximum_variant_fraction=float(np.bincount(ids).max()/len(ids)))
        loo=np.concatenate(deleted,axis=1)
        snp=_jackknife_covariance(loo)
        covariance+=snp
        result.update(snp_block_covariance=snp.tolist(),jackknife_replicates=loo.tolist(),
            jackknife_labels=labels.tolist(),
            jackknife_method="frozen_full_genome_variant_ldscore_delete_block_v1",
            uncertainty_status="experimental_joint_SNP_blocks_plus_reference_MC")
    modeled=()
    if participant_covariance is not None:
        extra,modeled,method=_participant_term(participant_covariance,moments,results,len(point))
        covariance+=extra
        result.update(participant_covariance=extra.tolist(),participant_covariance_method=method,
                      participant_covariance_pairs=list(modeled),
                      uncertainty_status=result["uncertainty_status"]+"_plus_participant_term")
    calibration=calibration_status(moments,results,modeled_pairs=modeled)
    result.update(calibration=calibration,component_calibration=calibration["components"])
    result.update(covariance=covariance.tolist(),standard_errors=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),
                  fit_uncertainty_seconds=perf_counter()-start)
    for pair in moments.pairs:
        if pair.left == pair.right:
            continue
        key=f"{pair.left}:{pair.right}"
        aa=np.array(results[f"{pair.left}:{pair.left}"]["omega_total"])
        bb=np.array(results[f"{pair.right}:{pair.right}"]["omega_total"])
        ab=np.array(results[key]["omega_total"])
        result["pairs"][key]["joint_minimum_eigenvalue"]=float(np.linalg.eigvalsh(np.block([[aa,ab],[ab.T,bb]])).min())
    result["component_intervals"]=[linear_interval(result,direction) for direction in np.eye(len(point))]
    return result


def contrast_calibration(result, contrast):
    """Calibration flag of a contrast: uncalibrated if it uses a flagged pair."""
    calibration = result.get("calibration")
    if calibration is None:
        return dict(status="unknown", uncalibrated_pairs=[])
    c = np.asarray(contrast, dtype=float)
    flagged = []
    for key, entry in calibration["pairs"].items():
        begin, end = result["pairs"][key]["coefficient_slice"]
        if entry["status"] == "uncalibrated" and np.any(c[begin:end] != 0):
            flagged.append(key)
    return dict(status="uncalibrated" if flagged else "not_flagged", uncalibrated_pairs=flagged)


def linear_interval(result, contrast, *, value=None, confidence=None):
    """Working finite-block interval for a linear contrast or delta gradient.

    The Welch-Satterthwaite degrees of freedom assume independent variance
    estimates from approximately independent, balanced SNP blocks and probes.
    A supplied participant covariance term is treated as known. This is not
    an exact t pivot for arbitrary LD or fitted nuisance models. The returned
    calibration flag marks contrasts that use components with known deficits.
    """
    from scipy.stats import norm, t
    c=np.asarray(contrast,dtype=float)
    point=np.asarray(result["components"])
    if c.shape!=point.shape or not np.isfinite(c).all():
        raise ValueError("contrast must match the finite coefficient axis")
    level=result.get("confidence_level",.95) if confidence is None else confidence
    if not np.isfinite(level) or not 0<level<1:
        raise ValueError("confidence must lie strictly inside (0,1)")
    estimate=float(c@point) if value is None else float(value)
    if not np.isfinite(estimate):
        raise ValueError("interval estimate must be finite")
    out=dict(estimate=estimate,confidence=float(level),interval=None,
             method="undefined_without_trait_uncertainty",degrees_of_freedom=None,
             calibration=contrast_calibration(result,c))
    blocks=result.get("jackknife_block_count",0)
    if blocks<2:
        return out
    vs=max(0.,float(c@np.asarray(result["snp_block_covariance"])@c))
    vr=max(0.,float(c@np.asarray(result["reference_mc_covariance"])@c))
    vp=(max(0.,float(c@np.asarray(result["participant_covariance"])@c))
        if "participant_covariance" in result else 0.)
    variance=vs+vr+vp
    if variance==0:
        out.update(interval=[estimate,estimate],standard_error=0.,method="degenerate_zero_variance")
        return out
    probes=result["reference_probe_count"]
    denominator=vs*vs/(blocks-1)+vr*vr/(probes-1)
    df=variance*variance/denominator if denominator else np.inf
    critical=float(t.ppf((1+level)/2,df)) if np.isfinite(df) else float(norm.ppf((1+level)/2))
    half=critical*np.sqrt(variance)
    out.update(interval=[float(estimate-half),float(estimate+half)],standard_error=float(np.sqrt(variance)),
               method="approximate_SNP_block_reference_Satterthwaite",
               degrees_of_freedom=float(df) if np.isfinite(df) else None,
               reference_variance_fraction=vr/variance)
    if "participant_covariance" in result:
        out["participant_variance_fraction"]=vp/variance
    return out


def _contrast(result, left, right, metric):
    key=f"{min(left,right)}:{max(left,right)}"
    pair=result["pairs"][key]
    if left > right:
        metric=np.asarray(metric).transpose(0,2,1)
    matrix=np.asarray(pair["omega"])
    metric=np.asarray(metric,dtype=float)
    if metric.shape != matrix.shape or not np.isfinite(metric).all():
        raise ValueError("population/context metric must have annotation by left-context by right-context shape")
    contrast=np.zeros(len(result["components"]))
    values=np.column_stack([metric[:,u,v]*(2 if left==right and u!=v else 1) for u,v in pair["pair_table"]])
    start,end=pair["coefficient_slice"]
    contrast[start:end]=values.ravel()
    return contrast


def derived_covariance(result, left, right, metric_aa, metric_bb, metric_ab, *,
                       total_variances=None):
    """Fixed centered population feature metrics, or context outer products.

    Population metrics are annotation-weighted Cov_pop(X_j phi_A, X_j phi_B),
    including removal of each feature mean. Context outer products describe
    effect covariance surfaces; conditional genotype variance one is assumed
    when those surfaces are interpreted as genetic variances.
    """
    aa,bb,ab=[np.asarray(value,dtype=float) for value in (metric_aa,metric_bb,metric_ab)]
    if any(value.ndim!=3 or not np.isfinite(value).all() for value in (aa,bb,ab)):
        raise ValueError("population kernel moments must be finite annotation by context by context arrays")
    if aa.shape[0]!=bb.shape[0] or aa.shape[0]!=ab.shape[0] or aa.shape[1]!=aa.shape[2] or bb.shape[1]!=bb.shape[2] or ab.shape[1:]!=(aa.shape[1],bb.shape[1]):
        raise ValueError("population kernel moment axes disagree")
    for a,b,c in zip(aa,bb,ab):
        joint=np.block([[a,c],[c.T,b]])
        if not np.allclose(joint,joint.T,rtol=1e-12,atol=1e-12) or np.linalg.eigvalsh(joint).min() < -1e-10*max(1.,np.linalg.norm(joint)):
            raise ValueError("joint population kernel moments must be positive semidefinite")
    contrasts=np.stack([_contrast(result,left,left,metric_aa),
                        _contrast(result,right,right,metric_bb),
                        _contrast(result,left,right,metric_ab)])
    point=contrasts@np.asarray(result["components"])
    cov=contrasts@np.asarray(result["covariance"])@contrasts.T
    va,vb,c=point
    se=np.sqrt(np.maximum(0,np.diag(cov)))
    out=dict(genetic_variances=point[:2].tolist(),genetic_covariance=float(c),
             joint_variance_covariance=cov.tolist(),genetic_covariance_se=float(se[2]),
             genetic_correlation=None,genetic_correlation_se=None,
             correlation_status="undefined_nonpositive_denominator",
             uncertainty_status=result["uncertainty_status"])
    out["variance_intervals"]=[linear_interval(result,direction) for direction in contrasts[:2]]
    out["genetic_covariance_interval"]=linear_interval(result,contrasts[2])
    # Variances, heritabilities and correlations use the within-trait
    # components; they inherit those components' interval calibration flags.
    variance_flags=[item["calibration"]["status"] for item in out["variance_intervals"]]
    out["calibration"]=dict(genetic_variances=variance_flags,heritabilities=variance_flags,
        genetic_covariance=out["genetic_covariance_interval"]["calibration"]["status"],
        genetic_correlation=contrast_calibration(result,contrasts.sum(0))["status"])
    if va>0 and vb>0:
        denom=np.sqrt(va*vb);r=c/denom
        jac=np.array([-r/(2*va),-r/(2*vb),1/denom])
        correlation_interval=linear_interval(result,jac@contrasts,value=r)
        out["genetic_correlation_interval"]=correlation_interval
        variance_lower=[item["interval"][0] if item["interval"] is not None else -np.inf
                        for item in out["variance_intervals"]]
        out.update(genetic_correlation=float(r),
                   genetic_correlation_se=float(np.sqrt(max(0,jac@cov@jac))),
                   correlation_status=("unreliable_weak_denominator" if min(variance_lower)<=0
                       else "experimental"),
                   correlation_outside_unit_interval=bool(abs(r)>1))
        if result.get("jackknife_block_count",0)<2:
            out["correlation_status"]="unreliable_no_trait_uncertainty"
        if any(result["pairs"][key]["weak_normal_equations"] for key in
               (f"{left}:{left}",f"{right}:{right}",f"{min(left,right)}:{max(left,right)}")):
            out["correlation_status"]="unreliable_weak_normal_equations"
    if total_variances is not None:
        total=np.asarray(total_variances,dtype=float)
        if total.shape!=(2,) or not np.isfinite(total).all() or np.any(total<=0):
            raise ValueError("heritability requires two positive fixed total population variances")
        out.update(heritabilities=(point[:2]/total).tolist(),
                   standardized_genetic_covariance=float(c/np.sqrt(total.prod())))
    return out


def evaluate_cross_contexts(result,left,right,phi_a,phi_b):
    """Ordered effect covariance surface, with matching variance denominators."""
    a=np.asarray(phi_a,dtype=float);b=np.asarray(phi_b,dtype=float)
    k=len(result["pairs"][f"{min(left,right)}:{max(left,right)}"]["omega"])
    return derived_covariance(result,left,right,
        np.tile(np.outer(a,a),(k,1,1)),np.tile(np.outer(b,b),(k,1,1)),np.tile(np.outer(a,b),(k,1,1)))
