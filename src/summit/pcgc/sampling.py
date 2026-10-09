"""Sampling covariance of the binary GxE estimating equations.

The sampling target holds the genomic architecture and population prevalence
fixed. Optional Gaussian effect, reference-probe and independent-reference
terms extend that target. Case and control counts are conditioned on. This
module is being qualified; it preserves the existing SNP-block jackknife.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, ndtri

from summit.sumstats.binary import finite_array, readonly


def center_strata(values, cases):
    out = np.array(values, copy=True)
    for case in (False, True):
        mask = cases == case
        if mask.sum() < 4:
            raise ValueError("individual-sampling inference needs at least four cases and four controls")
        out[mask] -= out[mask].mean(axis=0)
    return out


def nuisance_derivatives(risk, covariates, method, sd):
    """Observed probit information and full response/weight derivatives.

    The orthonormal parameterization changes no score-space derivatives.
    Returned IF rows sum to the coefficient perturbation for case-stratified
    empirical reweighting, before stratum centering.
    """
    n = risk.n_samples
    if covariates is None:
        if risk.source == "ascertained_probit":
            raise ValueError("fitted-risk sampling inference requires the original risk covariates")
        if risk.source not in ("supplied", "intercept_only"):
            raise ValueError("unsupported risk source for individual-sampling inference")
        empty = np.empty((n,0))
        return empty, empty, empty, empty
    if risk.source != "ascertained_probit":
        raise ValueError("risk covariates for inference require a same-study ascertained probit fit")
    cov = finite_array("risk covariates for inference",covariates,2)
    if len(cov) != n:
        raise ValueError("risk covariates must share the sample axis")
    raw = np.column_stack((np.ones(n),cov))
    if raw.shape[1] != len(risk.coefficients) or not np.allclose(ndtr(raw@risk.coefficients),risk.population_risk,rtol=1e-10,atol=1e-14):
        raise ValueError("risk covariates disagree with the fitted risk model")
    raw /= np.linalg.norm(raw,axis=0)
    if np.linalg.matrix_rank(raw) != raw.shape[1]:
        raise ValueError("risk inference design is rank deficient")
    design = np.linalg.qr(raw,mode="reduced")[0]
    p,k,z,d = risk.sample_risk,risk.population_risk,risk.z,risk.sensitivity
    eta = ndtri(k)
    normal_density = np.exp(-.5*eta**2)/np.sqrt(2*np.pi)
    skew = (1-2*p)/np.sqrt(p*(1-p))
    log_d_slope = -eta+.5*skew*d-normal_density*(1-2*k)/(k*(1-k))
    dz = (-d-.5*z*skew*d)[:,None]*design
    log_weight = log_d_slope[:,None]*design
    information = design.T@((d*d-d*z*(log_d_slope-.5*skew*d))[:,None]*design)
    if np.linalg.eigvalsh(information).min() <= 0 or np.linalg.cond(information) >= 1e10:
        raise ValueError("observed risk information is nonpositive or ill-conditioned")
    influence = np.linalg.solve(information,(d*z)[:,None].T*design.T).T
    if method == "pcgc-inverse":
        derivative = (dz-z[:,None]*log_weight)/(d/sd)[:,None]
        log_weight = np.zeros_like(log_weight)
    else:
        derivative = dz
    return design,influence,log_weight,derivative


@dataclass(frozen=True)
class SamplingMoments:
    """Covariance polynomial for [pair equations, population metric, V_L]."""
    constant: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    n_samples: int
    architecture_left: np.ndarray | None = None
    architecture_right: np.ndarray | None = None
    architecture_probes: int = 0
    pair_constant: np.ndarray | None = None
    pair_linear: np.ndarray | None = None
    pair_quadratic: np.ndarray | None = None

    def __post_init__(self):
        const = finite_array("sampling covariance constant",self.constant,2)
        linear = finite_array("sampling covariance linear",self.linear,3)
        quadratic = finite_array("sampling covariance quadratic",self.quadratic,4)
        c = len(linear)
        j = 2*c+1
        if c < 1 or const.shape != (j,j) or linear.shape != (c,j,j) or quadratic.shape != (c,c,j,j):
            raise ValueError("sampling covariance polynomial axes disagree")
        if type(self.n_samples) is not int or self.n_samples < 8:
            raise ValueError("invalid sampling covariance sample count")
        for name,value in (("constant",const),("linear",linear),("quadratic",quadratic)):
            object.__setattr__(self,name,readonly(value))
        if self.architecture_probes:
            if type(self.architecture_probes) is not int or self.architecture_probes < 2:
                raise ValueError("architecture probes must be an integer >=2")
            left = finite_array("architecture left sketch",self.architecture_left,3)
            right = finite_array("architecture right sketch",self.architecture_right,3)
            if left.shape[0] != c or left.shape[1] != left.shape[2] or right.shape != left.shape:
                raise ValueError("architecture sketch axes disagree")
            object.__setattr__(self,"architecture_left",readonly(left))
            object.__setattr__(self,"architecture_right",readonly(right))
        elif self.architecture_left is not None or self.architecture_right is not None:
            raise ValueError("architecture sketches require a probe count")
        if self.pair_constant is not None:
            for name,rank,shape in (("pair_constant",2,(c,c)),("pair_linear",3,(c,c,c)),("pair_quadratic",4,(c,c,c,c))):
                value = finite_array(name,getattr(self,name),rank)
                if value.shape != shape: raise ValueError("Hoeffding pair covariance axes disagree")
                object.__setattr__(self,name,readonly(value))
        elif self.pair_linear is not None or self.pair_quadratic is not None:
            raise ValueError("incomplete Hoeffding pair covariance")

    def covariance(self,theta):
        theta = finite_array("sampling covariance coefficients",theta,1)
        if theta.shape != (len(self.linear),):
            raise ValueError("sampling covariance coefficient count disagrees")
        value = self.constant+np.einsum('d,dij->ij',theta,self.linear)+np.einsum('d,e,deij->ij',theta,theta,self.quadratic)
        return (value+value.T)/2

    def pair_covariance(self,theta):
        if self.pair_constant is None: return None
        value = self.pair_constant+np.einsum('d,dij->ij',theta,self.pair_linear)+np.einsum('d,e,deij->ij',theta,theta,self.pair_quadratic)
        return (value+value.T)/2

    def component_direction_terms(self,theta,rows,products):
        """Linear/quadratic variance terms along each coefficient axis."""
        c = len(theta)
        if np.asarray(rows).shape != (c,2*c+1):
            raise ValueError('sampling confidence rows disagree')
        if products.native:
            return np.asarray(products.module.pcgc_sampling_direction_terms(
                np.asarray(theta),np.ascontiguousarray(rows),
                np.ascontiguousarray(self.linear.reshape(c,-1)),
                np.ascontiguousarray(self.quadratic.reshape(c*c,-1)),products.threads))
        result = []
        for d,row in enumerate(rows):
            first = self.linear[d]+np.einsum('e,eij->ij',theta,self.quadratic[d]+self.quadratic[:,d])
            result.append([row@first@row,row@self.quadratic[d,d]@row])
        return np.asarray(result)


def sampling_tile_people(n,partners):
    return min(n,max(1,32768//partners))


def sampling_panel_bytes(people,partners,c,p,s):
    # Six published panels plus the native feature/weight workspace. Protected
    # GEMM input snapshots are reserved separately in sampling_workspace_bytes.
    return 8*(people*partners*(3*s+3*c+p+2)+people*s)


def sampling_workspace_bytes(n,m,q,k,partners,block_size,risk_rank=0,architecture_probes=0):
    c = k*q*(q+1)//2
    j = 2*c+1
    b = min(m,block_size)
    packed = c*(c+1)//2
    # Pair identities/relatedness, score/actions/diagonals, nuisance arrays,
    # covariance tensors, assembly copies, and bounded product scratch.
    # Four copies of the packed influence panel cover the resident values,
    # protected input snapshots, and nuisance/centering scratch.
    result = 8*(n*partners*(k+1)+m*q+n*(6*c+3*k+8*risk_rank+4*packed)+n*b
              +6*(1+c+c*c)*j*j+6*c**4)+64*1024**2
    result += 8*packed*packed
    result += 3*sampling_panel_bytes(sampling_tile_people(n,partners),partners,c,q*(q+1)//2,packed)
    if architecture_probes:
        from .architecture import architecture_workspace_bytes
        result += architecture_workspace_bytes(n,q,k,architecture_probes,b)
    return result


def partner_proposal(features,response):
    """Basis-invariant proposal for both kernel and squared-score products."""
    basis = np.linalg.qr(features,mode="reduced")[0]
    leverage = np.sum(basis*basis,axis=1)
    squared = response*response
    weights = leverage*(1+squared/max(float(squared.mean()),np.finfo(float).tiny))
    return .25/len(response)+.75*weights/weights.sum()


def sample_partners(probabilities,count,seed):
    """Independent categorical draws conditional on excluding the same person."""
    probabilities = np.asarray(probabilities)
    n = len(probabilities)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.
    before = cumulative-probabilities
    rng = np.random.default_rng(np.random.SeedSequence([seed,0x50534758]))
    draws = rng.random((n,count))*(1-probabilities[:,None])
    draws += (draws>=before[:,None])*probabilities[:,None]
    result = np.searchsorted(cumulative,draws,side="right")
    if np.any(result>=n) or np.any(result==np.arange(n)[:,None]):
        raise RuntimeError("partner proposal failed to exclude the same person")
    return result


class SamplingOperator:
    """Reuse the two reference traversals for individual-sampling inference."""
    def __init__(self,operator,annotations,contexts,risk,sd,features,response,method,*,
                 partners,seed,risk_covariates=None,threads=1,native=True,
                 estimate_population_metric=True,estimate_population_variance=True,architecture_probes=0):
        from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator,ProtectedNNOperator
        from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator,ProtectedTNOperator
        from .gxe import context_pairs
        if type(partners) is not int or partners < 2:
            raise ValueError("individual-sampling inference requires at least two partners per person")
        if type(seed) is not int or seed < 0:
            raise ValueError("individual partner seed must be a nonnegative integer")
        if native:
            from summit.prediction.genotype import native_module
            module = native_module()
            required = ("pcgc_accumulate_relatedness","pcgc_pair_panels","pcgc_center_strata",
                        "pcgc_sampling_direction_terms","pcgc_architecture_features","pcgc_architecture_polynomials")
            if any(not callable(getattr(module,name,None)) for name in required):
                raise RuntimeError("native extension lacks PCGC moment kernels; rebuild SUMMIT")
        self.operator = operator
        self.annotations = annotations
        self.contexts,self.risk,self.sd = contexts,risk,sd
        self.features,self.response,self.method = features,response,method
        self.risk_covariates = risk_covariates
        self.estimate_population_metric = estimate_population_metric
        self.estimate_population_variance = estimate_population_variance
        # Validate nuisance ownership before any genotype traversal.
        nuisance_derivatives(risk,risk_covariates,method,sd)
        center_strata(np.zeros((len(contexts),1)),risk.z>0)
        n,m = operator.num_samples,operator.num_variants
        q,k = contexts.shape[1],annotations.shape[1]
        self.pairs = context_pairs(q)
        self.mass = annotations.sum(0)
        self.native,self.threads = native,threads
        self.partner_probabilities = partner_proposal(features,response)
        self.partners = sample_partners(self.partner_probabilities,partners,seed)
        self.relatedness = np.zeros((n,partners,k))
        self.scores = np.empty((m,q))
        self.actions = np.zeros((n,k*len(self.pairs)))
        self.diagonal = np.zeros((n,k))
        if native:
            from summit.prediction.genotype import native_module
            from summit.prediction.runtime import configure_prediction_threads
            module = native_module()
            configure_prediction_threads(module,threads)
            self.nn = ProtectedNNOperator(threads=threads,native_module=module)
            self.tn = ProtectedTNOperator(threads=threads,native_module=module)
        else:
            self.nn = NumpyNNOperator(threads=threads)
            self.tn = NumpyTNOperator(threads=threads)
        self.diagnostics = dict(method="individual_sampling_fixed_genome_v1",partners_per_person=partners,
            partner_seed=seed,partner_seed_domain=0x50534758,sampling="fixed_case_control_counts",
            partner_proposal="quarter_uniform_context_leverage_response_squared_v1",
            nuisance="same_study_probit_observed_information" if risk_covariates is not None else "fixed_supplied_risks",
            partner_monte_carlo_bias_corrected=True,reference_probe_uncertainty_included=False,
            genomic_architecture_uncertainty_included=False)
        self.architecture = None
        if architecture_probes:
            from .architecture import ArchitectureSketch
            self.architecture = ArchitectureSketch(features,annotations,risk.z>0,probes=architecture_probes,
                seed=seed,nn=self.nn,tn=self.tn,native=native,threads=threads)
            self.diagnostics.update(method="individual_and_gaussian_architecture_v1",
                genomic_architecture_uncertainty_included=True,architecture_probes=architecture_probes)

    def __getattr__(self,name):
        return getattr(self.operator,name)

    def read_block(self,start,stop):
        block = self.operator.read_block(start,stop)
        x = block.values
        a = np.asfortranarray(self.annotations[start:stop]/self.mass)
        n,l,k = self.relatedness.shape
        if self.observed_passes == 1:
            self.scores[start:stop] = self.tn.matmul_tn(x,np.asfortranarray(self.features*self.response[:,None]))
            self.diagonal += self.nn.matmul(np.asfortranarray(x*x),a)
            from .genotype import accumulate_relatedness
            from time import perf_counter
            started = perf_counter()
            accumulate_relatedness(x,a,self.partners,self.relatedness,
                nn=self.nn,native=self.native,threads=self.threads)
            self.diagnostics['sampled_pair_seconds'] = self.diagnostics.get('sampled_pair_seconds',0.)+perf_counter()-started
            self.diagnostics['sampled_pair_variant_products'] = self.diagnostics.get('sampled_pair_variant_products',0)+n*l*(stop-start)
        elif self.observed_passes == 2:
            for annotation in range(k):
                values = self.nn.matmul(x,np.asfortranarray(a[:,annotation,None]*self.scores[start:stop]))
                for p,(u,v) in enumerate(self.pairs):
                    self.actions[:,annotation*len(self.pairs)+p] += self.features[:,u]*values[:,v]
                    if u != v:
                        self.actions[:,annotation*len(self.pairs)+p] += self.features[:,v]*values[:,u]
        if self.architecture is not None:
            self.architecture.read_block(self.observed_passes,start,stop,x)
        return block

    def finalize(self,*,reference_annotation_gram=None):
        from .gxe import pair_products
        if self.observed_passes != 2:
            raise ValueError("sampling inference requires both study genotype traversals")
        diagonal = (self.diagonal[:,:,None]*pair_products(self.features)[:,None,:]).reshape(self.actions.shape)
        actions = self.actions-diagonal*self.response[:,None]
        result = build_sampling_moments(pair_kernels=self.relatedness,partners=self.partners,
            partner_probabilities=self.partner_probabilities,
            kernel_actions=actions,genotype_diagonal=self.diagonal,contexts=self.contexts,
            features=self.features,response=self.response,risk=self.risk,sd=self.sd,method=self.method,
            risk_covariates=self.risk_covariates,estimate_population_metric=self.estimate_population_metric,
            estimate_population_variance=self.estimate_population_variance,
            reference_annotation_gram=reference_annotation_gram,native=self.native,threads=self.threads,
            execution=self.diagnostics.setdefault("execution",{}))
        if self.architecture is not None:
            from dataclasses import replace
            result = replace(result,architecture_left=self.architecture.left,architecture_right=self.architecture.right,
                             architecture_probes=self.architecture.probes)
        return result


def build_sampling_moments(*, pair_kernels, partners, kernel_actions, genotype_diagonal,
                           contexts, features, response, risk, sd, method, risk_covariates=None,
                           estimate_population_metric=True, estimate_population_variance=True,
                           sampled=True, reference_annotation_gram=None,partner_probabilities=None,
                           native=False,threads=1,execution=None):
    """Form an unbiased pair-sampling approximation to a U-statistic sandwich.

    Each row samples L independent partners from all other people. Optional
    categorical probabilities are conditioned on excluding that row; inverse
    probabilities preserve the exact sums.
    Kernels and actions have already removed their original same-person terms.
    The correction for partner Monte Carlo noise is algebraic, not an SE
    inflation. Genomic random-effect or reference-probe uncertainty is separate.
    """
    from .gxe import context_pairs, pair_products
    partner = np.asarray(partners)
    n,l = partner.shape
    if n != risk.n_samples or l < 2 or partner.dtype.kind not in 'iu' or np.any(partner<0) or np.any(partner>=n):
        raise ValueError("invalid individual partner indices")
    if np.any(partner == np.arange(n)[:,None]):
        raise ValueError("same-person pairs are forbidden")
    if not sampled and (l != n-1 or any(len(np.unique(row)) != n-1 for row in partner)):
        raise ValueError("exact pair covariance requires every distinct partner once")
    probabilities = None
    if partner_probabilities is not None:
        probabilities = finite_array("partner probabilities",partner_probabilities,1)
        if not sampled or probabilities.shape != (n,) or np.any(probabilities<=0) or np.any(probabilities>=1) or not np.isclose(probabilities.sum(),1.,rtol=1e-12,atol=1e-14):
            raise ValueError("partner probabilities must be a positive categorical distribution")
    base = finite_array("sampled genetic relatedness",pair_kernels,3)
    if base.shape[:2] != (n,l):
        raise ValueError("sampled kernel and partner axes disagree")
    k = base.shape[2]
    q = contexts.shape[1]
    pairs = context_pairs(q)
    p,c = len(pairs),k*len(pairs)
    actions = finite_array("offdiagonal kernel response actions",kernel_actions,2)
    if actions.shape != (n,c):
        raise ValueError("kernel action axes disagree")
    cases = risk.z > 0
    design,risk_if,log_weight,dresponse = nuisance_derivatives(risk,risk_covariates,method,sd)
    # F = P_strata (I + 2 IF log_weight.T) propagates the fitted weights
    # through H. Its column squared norms weight the partner-noise correction.
    group_size = np.where(cases,cases.sum(),(~cases).sum())
    leverage = 1-1/group_size
    if risk_if.shape[1]:
        low_rank = center_strata(2*risk_if,cases)
        leverage += 2*np.sum(low_rank*log_weight,axis=1)
        leverage += np.einsum('ir,rs,is->i',log_weight,low_rank.T@low_rank,log_weight)
    from summit.ldscore.matrix_products import MatrixProducts
    products_backend = MatrixProducts(native=native,threads=threads)
    context_only = reference_annotation_gram is not None
    dimension = p if context_only else c
    upper = np.column_stack(np.triu_indices(dimension)).astype(np.int64)
    pair_index = np.empty((dimension,dimension),dtype=np.int64)
    pair_index[upper[:,0],upper[:,1]] = np.arange(len(upper))
    pair_index[upper[:,1],upper[:,0]] = np.arange(len(upper))
    if context_only:
        gram = finite_array("external annotation LD Gram",reference_annotation_gram,2)
        if gram.shape != (k,k):
            raise ValueError("external annotation LD Gram axes disagree")
        # Directional annotation Grams need not be symmetric. Only the context
        # product is shared; keep the ordered annotation multiplier intact.
        index = pair_index[np.arange(c)[:,None]%p,np.arange(c)[None,:]%p]
        multiplier = gram[np.arange(c)[:,None]//p,np.arange(c)[None,:]//p]
    else:
        index = pair_index
        multiplier = np.ones((c,c))
    s = len(upper)
    hrow = np.empty((n,s),order="F")
    t0 = np.zeros((c,c))
    t1 = np.zeros((c,s))
    t2 = np.zeros((s,s))
    mc = np.zeros((s,s))
    width = sampling_tile_people(n,l)
    panel_bytes = sampling_panel_bytes(width,l,c,p,s)
    context_pairs_array = np.asarray(pairs,dtype=np.int64)
    if native:
        panel_function = getattr(products_backend.module,"pcgc_pair_panels",None)
        if not callable(panel_function):
            raise RuntimeError("native extension lacks PCGC moment kernels; rebuild SUMMIT")
        base = np.ascontiguousarray(base)
        partner = np.ascontiguousarray(partner,dtype=np.int64)
        features = np.ascontiguousarray(features)
        response = np.ascontiguousarray(response)
        probability_input = np.empty(0) if probabilities is None else np.ascontiguousarray(probabilities)
    from time import perf_counter
    started = perf_counter()
    for start in range(0,n,width):
        stop = min(n,start+width)
        if native:
            panels = panel_function(base.reshape(n*l,k),partner,features,response,probability_input,
                np.ascontiguousarray(leverage),context_pairs_array,upper,start,stop,context_only,
                panel_bytes,threads)
            score,weighted_score,square,weighted_square,noise_square,mean = map(np.asarray,panels)
        else:
            other = partner[start:stop]
            f = features[start:stop]
            context = np.stack([f[:,u,None]*features[other,v] if u == v else
                f[:,u,None]*features[other,v]+f[:,v,None]*features[other,u] for u,v in pairs],axis=-1)
            kernels = (base[start:stop,:,:,None]*context[:,:,None,:]).reshape(-1,c)
            values = context.reshape(-1,p) if context_only else kernels
            square = values[:,upper[:,0]]*values[:,upper[:,1]]
            inverse = (np.full((stop-start,l),n-1.) if probabilities is None else
                (1-probabilities[start:stop,None])/probabilities[other]).reshape(-1,1)
            score = kernels*(response[start:stop,None]*response[other]).reshape(-1,1)
            weighted_score = score*inverse
            weighted_square = square*inverse
            noise_square = weighted_square*inverse*np.repeat(leverage[start:stop],l)[:,None]
            mean = weighted_square.reshape(stop-start,l,s).mean(1)
        hrow[start:stop] = mean
        t0 += products_backend.tn(score,weighted_score)/l
        t1 += products_backend.tn(score,weighted_square)/l
        t2 += products_backend.tn(square,weighted_square)/l
        if sampled:
            mc += (products_backend.tn(square,noise_square)/l -
                products_backend.tn(mean,leverage[start:stop,None]*mean))/(l-1)
        products_backend.drain()
        del score,weighted_score,square,weighted_square,noise_square,mean
        if native:
            del panels
    pair_seconds = perf_counter()-started
    j = 2*c+1
    a = np.zeros((n,j),order="F")
    a[:,:c] = 2*response[:,None]*actions
    if risk_if.shape[1]:
        db = 2*products_backend.tn(actions,dresponse+response[:,None]*log_weight)
        a[:,:c] += products_backend.nn(risk_if,db.T)
        dh = 4*products_backend.tn(log_weight,hrow)
    # One coefficient per unique pair product. All population-summary columns
    # of the old N x C x (2C+1) influence array were identically zero.
    b = hrow
    b *= -2
    if risk_if.shape[1]:
        products_backend.subtract_rank(b,risk_if,dh)
    count = float(n*(n-1))
    a[:,:c] /= count
    b /= count
    K,P = risk.population_prevalence,risk.sample_prevalence
    weights = np.where(cases,K/P,(1-K)/(1-P))/n
    if estimate_population_metric:
        values = (genotype_diagonal[:,:,None]*pair_products(contexts)[:,None,:]).reshape(n,c)
        a[:,c:2*c] = weights[:,None]*values
    if estimate_population_variance:
        mu = sd*ndtri(risk.population_risk)
        centered_mu = mu-weights@mu
        a[:,-1] = weights*(sd**2+centered_mu**2)
        if risk_if.shape[1]:
            derivative = 2*(weights*centered_mu*sd)@design
            a[:,-1] += risk_if@derivative
    products_backend.center_strata(a,cases)
    products_backend.center_strata(b,cases)
    const = products_backend.tn(a,a)
    cross = products_backend.tn(a,b)
    gram = products_backend.tn(b,b)
    # index is (equation, coefficient); the saved polynomial is
    # (coefficient[, coefficient], equation, equation).
    ix,scale = index.T,multiplier.T
    linear = np.zeros((c,j,j))
    linear[:,:,:c] = cross[:,ix].transpose(1,0,2)*scale[:,None,:]
    linear += linear.transpose(0,2,1).copy()
    pair_linear = -2*(t1[:,ix].transpose(1,0,2)*scale[:,None,:])/count**2
    pair_linear += pair_linear.transpose(0,2,1).copy()
    left,right = ix[:,None,:,None],ix[None,:,None,:]
    factor = scale[:,None,:,None]*scale[None,:,None,:]
    quadratic = np.zeros((c,c,j,j))
    quadratic[:,:,:c,:c] = (gram-(2*t2+4*mc)/count**2)[left,right]*factor
    pair_quadratic = (2*t2/count**2)[left,right]*factor
    pair_constant = 2*t0/count**2
    const[:c,:c] -= pair_constant
    linear[:,:c,:c] -= pair_linear
    if execution is not None:
        execution.update(products_backend.drain(),pair_feature_columns=s,
            expanded_pair_columns=c*c,pair_accumulation_seconds=pair_seconds,
            finalization_seconds=perf_counter()-started)
    else:
        products_backend.drain()
    return SamplingMoments(const,linear,quadratic,n,
        pair_constant=pair_constant,pair_linear=pair_linear,pair_quadratic=pair_quadratic)


def hoeffding_working_covariance(covariance,pair_covariance):
    """Nonnegative first Hoeffding component in its covariance metric.

    The Hoeffding variance has two PSD population components. Use the raw
    pair second moment for its leading second-order term; centering this
    kernel adds lower-order corrections away from degeneracy. Subtracting
    this pair term from the estimated variance can give an indefinite first
    component. Retain the pair term and project the first onto the PSD cone in a
    congruence-invariant metric. This is a working variance regularization;
    it is not an unbiased finite-sample covariance estimator.
    """
    c = len(pair_covariance)
    pair = np.zeros_like(covariance); pair[:c,:c] = pair_covariance
    metric = pair.copy(); metric[c:,c:] = covariance[c:,c:]
    diagonal = np.diag(metric)
    if np.any(diagonal<0):
        raise ValueError("pair or population-summary covariance is nonpositive")
    scales = np.sqrt(diagonal)
    scales[scales==0] = 1.
    # Genetic equations and population summaries have different units and
    # different N/M scaling. Decide numerical rank in correlation units.
    standardized = metric/np.outer(scales,scales)
    e,u = np.linalg.eigh((standardized+standardized.T)/2)
    cutoff = max(float(e[-1]),np.finfo(float).tiny)*1e-12
    if e[0] < -cutoff:
        raise ValueError("pair or population-summary covariance is nonpositive")
    active = e>cutoff
    if not np.any(active): raise ValueError("working covariance has no positive directions")
    root = scales[:,None]*u[:,active]*np.sqrt(e[active])
    inverse = (u[:,active]/np.sqrt(e[active])).T/scales[None,:]
    first = covariance-pair
    white = inverse@first@inverse.T
    values,vectors = np.linalg.eigh((white+white.T)/2)
    adjustment = root@((vectors*np.maximum(-values,0))@vectors.T)@root.T
    # Null directions of the population-summary metric must carry no signal.
    residual = (first-root@(inverse@first@inverse.T)@root.T)/np.outer(scales,scales)
    if np.linalg.norm(residual)>1e-8*max(np.linalg.norm(covariance/np.outer(scales,scales)),np.finfo(float).tiny):
        raise ValueError("working covariance metric does not span the sampling moments")
    return adjustment,dict(method="positive_hoeffding_components_v1",adjusted_directions=int(np.sum(values<0)),
        minimum_first_component_eigenvalue=float(values.min()),
        relative_adjustment=float(np.linalg.norm(adjustment)/max(np.linalg.norm(covariance),np.finfo(float).tiny)),
        equation_relative_adjustment=float(np.linalg.norm(adjustment[:c,:c])/max(np.linalg.norm(covariance[:c,:c]),np.finfo(float).tiny)))


def sampling_inference(moments,theta,H,population_weights,*,native=True,threads=None):
    """Propagate component, population-kernel and liability-variance sampling."""
    from summit.ldscore.matrix_products import MatrixProducts
    products = MatrixProducts(native=native,threads=threads)
    c = len(theta)
    count = moments.n_samples*(moments.n_samples-1)
    transform = np.eye(2*c+1)
    transform[:c,:c] = np.linalg.inv(H/count)
    architecture_diagnostics = None
    working_adjustment = np.zeros((2*c+1,2*c+1))
    working_diagnostics = None
    pair_covariance = moments.sampling_moments.pair_covariance(theta)
    if pair_covariance is not None:
        working_adjustment,working_diagnostics = hoeffding_working_covariance(
            moments.sampling_moments.covariance(theta),pair_covariance)
    if moments.sampling_moments.architecture_probes:
        from .architecture import gaussian_architecture_covariance,ArchitectureCovariance
        model = moments.sampling_moments
        evaluator = ArchitectureCovariance(model.architecture_left,model.architecture_right,
            moments.num_contexts,model.architecture_probes,products)
        architecture,architecture_diagnostics = gaussian_architecture_covariance(theta,moments.num_contexts,
            moments.population_second_moment,model.architecture_left,model.architecture_right,model.architecture_probes,
            evaluator=evaluator)
    def equation_covariance(value,*,include_sampling=True,include_architecture=True):
        result = working_adjustment.copy()
        if include_sampling:
            result += moments.sampling_moments.covariance(value)
        if include_architecture and architecture_diagnostics is not None:
            from .gxe import _omega
            working = np.asarray(architecture_diagnostics['working_omega'])+_omega(value-theta,moments.num_contexts)
            result[:c,:c] += architecture if np.array_equal(value,theta) else evaluator.covariance(working)
        if moments.reference_probe_deviations is not None:
            deviations = np.einsum('bij,j->bi',moments.reference_probe_deviations,value)/count
            probes = len(deviations)
            result[:c,:c] += deviations.T@deviations/(probes*(probes-1))
        if moments.external_reference_covariance is not None:
            k,p = moments.annotations.shape[1],len(moments.pairs)
            derivative = np.zeros((k,p,k,k))
            for annotation in range(k):
                derivative[annotation,:,annotation,:] = moments.external_context_gram@value.reshape(k,p).T
            derivative = derivative.reshape(c,k*k)/count
            result[:c,:c] += derivative@moments.external_reference_covariance@derivative.T
        return result
    joint = transform@equation_covariance(theta)@transform.T
    cov = (joint[:c,:c]+joint[:c,:c].T)/2
    diagonal = np.diag(cov)
    if np.any(diagonal <= 0):
        raise ValueError("combined PCGC covariance is nonpositive; assess model identification and partner/reference/architecture sketch precision")
    correlation = cov/np.sqrt(np.outer(diagonal,diagonal))
    if np.linalg.eigvalsh(correlation).min() < -1e-8:
        raise ValueError("combined PCGC covariance is indefinite; assess model identification and partner/reference/architecture sketch precision")
    genetic = float(population_weights@theta)
    V = moments.population_liability_variance
    vg_gradient = np.r_[population_weights,theta,0.]
    h2_gradient = vg_gradient/V
    h2_gradient[-1] = -genetic/V**2
    vgvar,h2var = float(vg_gradient@joint@vg_gradient),float(h2_gradient@joint@h2_gradient)
    if min(vgvar,h2var) < 0:
        raise ValueError("individual-sampling total-variance covariance is nonpositive")
    se = np.sqrt(diagonal)
    output = dict(covariance=cov.tolist(),standard_errors=se.tolist(),
        population_genetic_variance_se=float(np.sqrt(vgvar)),population_heritability_se=float(np.sqrt(h2var)),
        uncertainty_status="estimated",uncertainty_method="individual_sampling_fixed_genome_v1",
        uncertainty_calibration_status="experimental_coverage_not_established",
        uncertainty_conditioning="fixed_population_prevalence_liability_sd_genomic_architecture_genotype_adjustment_reference_probes",
        population_summary_covariance="propagated_from_saved_sampling_moments",external_reference_uncertainty_included=False,
        component_normal_intervals_95=np.column_stack((theta-1.959963984540054*se,theta+1.959963984540054*se)).tolist())
    if architecture_diagnostics is not None:
        output.update(uncertainty_method="individual_and_gaussian_architecture_v1",architecture_covariance_model=architecture_diagnostics,
            genomic_architecture_uncertainty_included=True,
            uncertainty_conditioning="fixed_population_prevalence_liability_sd_genotype_adjustment_reference_probes")
    output["reference_probe_uncertainty_included"] = moments.reference_probe_deviations is not None
    output["external_reference_uncertainty_included"] = moments.external_reference_covariance is not None
    if working_diagnostics is not None:
        output['sampling_covariance_regularization'] = working_diagnostics
    if moments.reference_probe_deviations is not None:
        output["uncertainty_conditioning"] = output["uncertainty_conditioning"].removesuffix("_reference_probes")
    # Candidate-parameter variance is a quadratic. Retain complete confidence
    # sets, including an unbounded set if its quadratic requires one.
    sets = []
    variance_polynomials = []
    sampling_terms = moments.sampling_moments.component_direction_terms(theta,transform[:c],products)
    architecture_terms = np.zeros((c,3))
    if architecture_diagnostics is not None:
        from .gxe import _omega
        architecture_terms = evaluator.scalar_polynomials(
            np.asarray(architecture_diagnostics['working_omega']),transform[:c,:c],
            np.asarray([_omega(direction,moments.num_contexts) for direction in np.eye(c)]))
    extra0 = equation_covariance(theta,include_sampling=False,include_architecture=False)
    for target in range(c):
        direction = np.eye(c)[target]
        row = transform[target]
        v0 = float(cov[target,target])
        base = float(row@extra0@row)
        vp = float(row@equation_covariance(theta+direction,include_sampling=False,include_architecture=False)@row)
        vm = float(row@equation_covariance(theta-direction,include_sampling=False,include_architecture=False)@row)
        v1 = (vp-vm)/2+sampling_terms[target,0]+architecture_terms[target,1]
        v2 = (vp+vm)/2-base+sampling_terms[target,1]+architecture_terms[target,2]
        variance_polynomials.append([v0,v1,v2])
        z2 = 1.959963984540054**2
        from .score_sets import polynomial_nonpositive_set
        scale = se[target]
        sets.append(polynomial_nonpositive_set([-z2*v0,-z2*v1*scale,(1-z2*v2)*scale**2],
            center=theta[target],scale=scale))
    output["component_plugin_score_sets_95"] = sets
    output["component_score_variance_polynomial"] = variance_polynomials
    output["score_set_nuisance"] = "other_components_held_at_unrestricted_estimates"
    output["score_set_architecture"] = "candidate_coefficient_in_fitted_working_gaussian_model"
    # Linear population variance and the ratio Vg/V_L also propagate uncertainty
    # in the metric and denominator. Profile the coefficient nuisance along
    # the estimated covariance direction for the requested linear functional.
    direction_variance = float(population_weights@cov@population_weights)
    if direction_variance > 0:
        coefficient_direction = cov@population_weights/direction_variance
        for label,point,scale,denominator in (("population_genetic_variance",genetic,np.sqrt(vgvar),1.),
                                             ("population_heritability",genetic/V,np.sqrt(h2var),V)):
            if scale == 0:
                output[label+"_plugin_score_set_95"] = [[point,point]]
                continue
            step = coefficient_direction*(scale*denominator)
            C0 = joint
            Cp = transform@equation_covariance(theta+step)@transform.T
            Cm = transform@equation_covariance(theta-step)@transform.T
            C1,C2 = (Cp-Cm)/2,(Cp+Cm)/2-C0
            g0 = np.r_[population_weights,theta,-point if label=="population_heritability" else 0.]
            g1 = np.r_[np.zeros(c),step,-scale if label=="population_heritability" else 0.]
            variance = np.array([g0@C0@g0,2*g1@C0@g0+g0@C1@g0,
                g1@C0@g1+2*g1@C1@g0+g0@C2@g0,g1@C1@g1+2*g1@C2@g0,g1@C2@g1])
            criterion = -1.959963984540054**2*variance
            criterion[2] += (scale*denominator)**2
            from .score_sets import polynomial_nonpositive_set
            output[label+"_plugin_score_set_95"] = polynomial_nonpositive_set(criterion,center=point,scale=scale)
    output["population_score_set_nuisance"] = "coefficient_covariance_direction_with_estimated_metric_and_liability_variance"
    products.drain()
    return output
