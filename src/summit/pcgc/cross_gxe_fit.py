"""Directed joint fits and conditional SNP-block/reference-draw uncertainty."""
from time import perf_counter
import numpy as np
from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes
from summit.context.annotations import _jackknife_covariance
from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
from .moments import solve


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
                           12*blocks*dimension+4*probes*dimension+8*len(moments.annotations))
    peak=baseline+workspace
    if memory_bytes is not None:
        if type(memory_bytes) is not int or memory_bytes<1:
            raise ValueError("fit memory budget must be a positive integer")
        if peak>memory_bytes:
            raise MemoryError("cross-trait fit rows and joint uncertainty exceed memory budget")
    return dict(baseline_rss_bytes=baseline,fit_workspace_bytes=workspace,
                peak_planned_bytes=peak,memory_budget_bytes=memory_bytes)


def fit_cross_gxe(moments, *, block_ids=None, memory_bytes=None, confidence=.95):
    """No genotype access; reference draws and deletion IDs are shared."""
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


def linear_interval(result, contrast, *, value=None, confidence=None):
    """Working finite-block interval for a linear contrast or delta gradient.

    The Welch-Satterthwaite degrees of freedom assume independent variance
    estimates from approximately independent, balanced SNP blocks and probes.
    This is not an exact t pivot for arbitrary LD or fitted nuisance models.
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
             method="undefined_without_trait_uncertainty",degrees_of_freedom=None)
    blocks=result.get("jackknife_block_count",0)
    if blocks<2:
        return out
    vs=max(0.,float(c@np.asarray(result["snp_block_covariance"])@c))
    vr=max(0.,float(c@np.asarray(result["reference_mc_covariance"])@c))
    variance=vs+vr
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
