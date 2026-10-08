"""Portable finite-feature summaries for conditional polygenic mean inference.

The covariance estimator and cohort-side preparation remain under validation.
This adapter uses the existing score serialization and inference machinery; it
does not estimate a covariance, reuse one across phenotypes, or establish its
statistical validity.
"""
import numpy as np

from summit.context.spec import canonical_sha256, array_sha256
from .robust import RobustScoreSummary


def frozen_local_mean(source, training_rows, confirmation_rows, specification, root,
                      *, threads=1, memory_bytes=4*2**30):
    """Build one public finite mean recipe in frozen training genotype units.

    Only local calls are retained. Additive and heterozygote missing values use
    training constants in both cohorts; score constituents are added later.
    Returned training hashes can be compared directly with saved public learners.
    """
    from pathlib import Path
    from summit.prediction.genotype import ArrayGenotypeSource,estimate_scale,native_module
    from summit.prediction.spec import GenotypeScale
    from summit.prediction.cli import _aligned_table
    from summit.prediction._validation import array_digest,digest
    from .models import target_design
    from .nuisance import select_structure_covariates,varying_main_effects
    train,test=map(np.asarray,(training_rows,confirmation_rows))
    if (not source.hard_calls or any(v.ndim!=1 or len(v)<2 or not np.issubdtype(v.dtype,np.integer)
            or np.any(np.diff(v)<=0) or v[0]<0 or v[-1]>=len(source.samples) for v in (train,test))
            or np.intersect1d(train,test).size):
        raise ValueError('disjoint ordered hard-call training and confirmation rows required')
    rows=np.union1d(train,test)
    i0,i1=np.searchsorted(rows,train),np.searchsorted(rows,test)
    spec=specification
    names=set(spec.get('local_variants',[]))|set(spec.get('dominance_variants',[]))|{spec['target']}
    lookup={v:j for j,v in enumerate(source.variants.ids)}
    if names-set(lookup) or len(names)>4096:
        raise ValueError('declare at most 4096 known local mean variants')
    selected=np.array(sorted(lookup[v] for v in names))
    if 64*len(rows)*len(selected)+256*2**20>memory_bytes:
        raise MemoryError('local finite-mean preparation exceeds memory budget')
    source.prepare(rows,max(128,len(selected)),threads)
    raw=source.read(selected)
    samples=[source.samples[i] for i in rows]
    local=ArrayGenotypeSource(raw,samples,source.variants.subset(selected),hard_calls=True)
    empirical=estimate_scale(local,i0,np.arange(len(selected)),threads=threads)
    variance=empirical.mean*(1-empirical.mean/2)
    observed=raw!=-127
    if np.any(variance<=0) or np.any(observed[i0].sum(0)==0):
        raise ValueError('local variants lack training genotype support')
    target=list(local.variants.ids).index(spec['target'])
    if not np.all(observed[:,target]):
        raise ValueError('conditional target genotypes must be observed')
    scale=GenotypeScale(empirical.mean,1/np.sqrt(variance),empirical.variant_identity,
        empirical.sample_identity,dict(empirical.provenance),ddof=0)
    heterozygosity=(raw[i0]==1).sum(0)/observed[i0].sum(0)
    frozen_h={v:float(heterozygosity[list(local.variants.ids).index(v)])
        for v in spec.get('dominance_variants',[])}
    cv=spec['covariates']
    cov=_aligned_table(Path(root)/cv['file'],samples)[cv['columns']].to_numpy(float)
    prepared=target_design(local,np.arange(len(rows)),scale,
        components=[dict(name='interaction',target=spec['target'],background='all')],
        annotations={'all':np.ones(len(selected))},additive_annotations=['all'],covariates=cov,
        local_variants=spec.get('local_variants',[]),dominance_variants=spec.get('dominance_variants',[]),
        dominance_imputation=frozen_h,threads=threads,native=native_module(),memory_bytes=memory_bytes)
    structure,z=select_structure_covariates(cv,cov)
    fixed=prepared['fixed_effects']
    if structure:
        fixed,definition=varying_main_effects(fixed,z,prepared['modifiers'][:,1:],
            covariate_names=structure,main_names=[spec['target']],memory_bytes=memory_bytes)
    else:
        definition=dict(method='declared_finite_mean_without_varying_main_effects')
    x=prepared['modifiers'][:,1]
    return dict(rows=rows,training_index=i0,confirmation_index=i1,fixed=fixed,target=x,
        contexts=np.column_stack([np.ones(len(rows)),z]),noise=np.column_stack([np.ones(len(rows)),x*x]),
        metadata=dict(training_samples=digest([source.samples[i] for i in train]),
            confirmation_samples=digest([source.samples[i] for i in test]),source=source.identity,
            target=spec['target'],target_mean=float(empirical.mean[target]),
            target_inverse_scale=float(scale.inverse_scale[target]),
            training_fixed_hash=array_digest(fixed[i0]),
            training_context_hash=array_digest(prepared['modifiers'][i0]),
            fixed_definition=definition,dominance_imputation=frozen_h,
            mean_definition='public local A/H and covariate mean, frozen training imputation and units'))


def conditional_mean_summary(beta, covariance, information, *, feature_names,
                             trait_name, identities, diagnostics=None, trait_unit):
    """Encode one phenotype's response-normalized estimate and joint covariance.

    Information is the Gram matrix of the actual residualized interaction
    response F1-L F0. Covariance refers to the coefficient vector, including all
    cross-direction terms. Each separately estimated null needs its own artifact.
    No participant arrays or IDs are admitted to the portable metadata.
    """
    beta, covariance, information = map(lambda x: np.asarray(x, float),
        (beta, covariance, information))
    p = len(feature_names)
    if (p==0 or beta.shape != (p,) or covariance.shape != (p,p) or information.shape != (p,p)
            or not all(np.all(np.isfinite(x)) for x in (beta,covariance,information))
            or not np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-12)
            or not np.allclose(information,information.T,rtol=1e-10,atol=1e-12)):
        raise ValueError('finite aligned coefficient, covariance and response-information axes required')
    # A coefficient-space covariance must be PSD before multiplication by H;
    # H could otherwise hide negative variance in a nonidentifiable direction.
    for a in (covariance,information):
        diagonal=np.diag(a)
        if np.any(diagonal<0):
            raise ValueError('negative conditional variance or information')
        scale=np.sqrt(diagonal); scale[scale==0]=1.
        eigen=np.linalg.eigvalsh(a/scale[:,None]/scale[None,:])
        if eigen[0] < -64*p*np.finfo(float).eps*max(1.,eigen[-1]):
            raise ValueError('indefinite conditional variance or information')
    required={'genotype_reference','direction','training_outcomes','confirmation_outcomes','null_fit'}
    if (set(identities)!=required or any(not isinstance(v,str) or len(v)!=64
            or any(c not in '0123456789abcdef' for c in v) for v in identities.values())):
        raise ValueError('authenticate genotype, direction, both phenotypes and the fitted null')
    if not isinstance(trait_unit,str) or not trait_unit:
        raise ValueError('a scientific phenotype unit is required')
    diagnostics=dict(diagnostics or {})
    allowed={'nuisance_training_n','confirmation_n','fixed_rank','feature_rank','max_leverage',
        'minimum_feature_effective_support','outside_confirmation_design','information_condition',
        'covariance_precision'}
    if set(diagnostics)-allowed:
        raise ValueError('conditional portable diagnostics may contain only aggregate support quantities')
    if 'covariance_precision' in diagnostics:
        precision=diagnostics['covariance_precision']
        numeric={'sampling_relative_sd','trace_relative_sd','fixed_contrast_variance_relative_sd'}
        fields=numeric|{'fixed_contrast_effective_df','nnls_boundary_components','method','scope'}
        if (not isinstance(precision,dict) or set(precision)!=fields
                or any(not isinstance(precision[k],(int,float,np.number))
                    or not np.isfinite(precision[k]) or precision[k]<0 for k in numeric)
                or (precision['fixed_contrast_effective_df'] is not None
                    and (not isinstance(precision['fixed_contrast_effective_df'],(int,float,np.number))
                        or not np.isfinite(precision['fixed_contrast_effective_df'])
                        or precision['fixed_contrast_effective_df']<=0))
                or not isinstance(precision['nnls_boundary_components'],list)
                or any(not isinstance(k,int) or k<0 for k in precision['nnls_boundary_components'])
                or any(not isinstance(precision[k],str) for k in ('method','scope'))):
            raise ValueError('conditional covariance precision must contain aggregate diagnostics only')
    metadata=dict(diagnostics,
        method='conditional_polygenic_mean_tangent_v1',
        inference='estimated Gaussian polygenic covariance; training-conditioned mean tangent adjustment; validation required',
        covariance_assumption='independent Gaussian additive, dominance and declared covariate-dependent genetic effects with declared diagonal noise surfaces',
        estimand='response-normalized finite interaction coefficient after conditional additive-null prediction',
        biological_null='no generating cross-locus interaction under the declared mean and genetic covariance model',
        kernel_target='finite mean features, not marginal interaction variance',
        trait_unit=trait_unit,phenotype_units='raw',
        preparation_identity=canonical_sha256(dict(identities=identities,features=list(feature_names),
            trait=trait_name,unit=trait_unit,method='conditional_polygenic_mean_tangent_v1',
            estimate=array_sha256(beta),covariance=array_sha256(covariance),
            response_information=array_sha256(information))),
        identities=dict(identities),status='research; full-marker calibration and power not yet qualified')
    return RobustScoreSummary((information@beta)[:,None],information,
        (information@covariance@information)[None,:,:],tuple(feature_names),(trait_name,),metadata)
