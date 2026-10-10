"""Continuous scale inference for genotype-defined coefficients with genomic covariance.

Contrasts are fixed conditional on genotypes, never outcome-trained directions.
Known covariance gives Gaussian marginal Wald inference. Estimated components
give model-based asymptotic inference, including when the two cohorts are
genetically correlated. This is not the conditional learned-score estimator.
"""
from math import factorial
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2, norm

from .polygenic import he_geometry, components_from_normalized_moments
from .robust import prepare_robust_geometry, _inverse
from .scale import boxcox_derivatives


def prepare_polygenic_scale(training, confirmation, training_features, confirmation_features,
                            training_fixed, confirmation_fixed, *, nn, tn,
                            probes=128, seed=871631, exact=False):
    """Reuse genotype geometry for all traits/scales with the same frozen design.

    The caller must freeze features, contexts, variance surfaces and row split
    without these outcomes. Include all tested features in the training mean,
    even for a null that tests only a subset of coefficients.
    """
    if training.stream.source is confirmation.stream.source:
        raise ValueError('each cohort operator requires its own genotype source owner')
    if (training.count != confirmation.count
            or training.stream.source.identity != confirmation.stream.source.identity
            or not np.array_equal(training.variants, confirmation.variants)
            or not np.array_equal(training.mean, confirmation.mean)
            or not np.array_equal(training.inverse, confirmation.inverse)
            or training.contexts.shape[1] != confirmation.contexts.shape[1]
            or np.intersect1d(training.rows, confirmation.rows).size):
        raise ValueError('disjoint rows and common frozen kernel definitions required')
    fa, fb, ca, cb = map(lambda x: np.asarray(x, float),
                        (training_features, confirmation_features, training_fixed, confirmation_fixed))
    if (any(x.ndim != 2 or not np.all(np.isfinite(x)) for x in (fa,fb,ca,cb))
            or len(fa) != len(training.rows) or len(ca) != len(fa)
            or len(fb) != len(confirmation.rows) or len(cb) != len(fb)
            or fa.shape[1] != fb.shape[1] or not fa.shape[1]):
        raise ValueError('finite aligned feature and nuisance matrices required')
    # The original coefficients stay jointly adjusted for every supplied term.
    geometry = prepare_robust_geometry(fb, cb, nn=nn, tn=tn)
    if not np.all(geometry.estimable):
        raise ValueError('identifiable genotype-defined coefficients required')
    one = np.ones((len(fb),1))
    if np.linalg.norm(one-nn(geometry.u,tn(geometry.u,one))) > 1e-9*np.sqrt(len(fb)):
        raise ValueError('confirmation nuisance span must include an intercept')
    weights = nn(geometry.r,geometry.inverse.T)
    weights -= nn(geometry.u,tn(geometry.u,weights))
    contracted = confirmation.component_grams(weights, phase='scale_confirmation_grams')
    diagnostics = dict(training_n=len(fa), confirmation_n=len(fb), feature_count=fb.shape[1],
        confirmation_fixed_rank=geometry.u.shape[1], information_condition=geometry.condition,
        maximum_nuisance_and_feature_leverage=float(geometry.leverage.max()))
    del geometry
    he = he_geometry(training,np.column_stack([ca,fa]),probes=probes,seed=seed,
                     exact=exact,keep_products=False)
    one = np.ones((len(fa),1))
    if np.linalg.norm(one-nn(he['basis'],tn(he['basis'],one))) > 1e-9*np.sqrt(len(fa)):
        raise ValueError('training mean span must include an intercept')
    eig = np.linalg.eigvalsh(he['h'])
    if eig[0] <= 0:
        raise ValueError('positive definite covariance moment geometry required')
    diagnostics.update(training_mean_rank=he['basis'].shape[1], he_condition=float(eig[-1]/eig[0]),
                       covariance_probes=he['probes'], exact_covariance_geometry=he['exact'])
    return SimpleNamespace(training=training, he=he, weights=weights, nn=nn, tn=tn,
        grams=contracted['gram'], weight_gram=tn(weights,weights), he_min_eigenvalue=float(eig[0]),
        covariance_radius_matrix=np.einsum('kij,k->ij',contracted['gram'],1/he['norms']),
        confirmation_identity=confirmation.identity, diagnostics=diagnostics)


class PolygenicScaleEvaluator:
    """Batched pointwise fits and conservative intervals, also used by tests."""
    def __init__(self, geometry, training_phenotype, confirmation_phenotype, *,
                 groups=None, order=3, contrast_mode='omnibus'):
        self.g = geometry
        a,b = map(lambda x: np.asarray(x,float),(training_phenotype,confirmation_phenotype))
        if (a.shape != (len(geometry.training.rows),) or b.shape != (len(geometry.weights),)
                or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
                or np.any(a <= 0) or np.any(b <= 0)):
            raise ValueError('positive finite aligned training and confirmation outcomes required')
        if type(order) is not int or not 1 <= order <= 4:
            raise ValueError('derivative order 1..4 required')
        if contrast_mode not in ('omnibus','omnibus_sparse'):
            raise ValueError('unknown genotype-defined contrast mode')
        self.order, self.contrast_mode = order, contrast_mode
        self.log_reference = float(np.log(a).mean())
        # The SAME outcome units must be used for the covariance and coefficients.
        self.ta, self.tb = np.log(a)-self.log_reference, np.log(b)-self.log_reference
        p = geometry.weights.shape[1]
        groups = {'joint':list(range(p))} if groups is None else dict(groups)
        if not groups:
            raise ValueError('nonempty coefficient groups required')
        self.groups = {}
        for name, selected in groups.items():
            index = np.asarray(selected)
            if (not isinstance(name,str) or not name or index.ndim != 1 or not len(index)
                    or not np.issubdtype(index.dtype,np.integer) or len(np.unique(index)) != len(index)
                    or np.any(index < 0) or np.any(index >= p)):
                raise ValueError('distinct valid integer coefficient indices required per group')
            self.groups[name] = index
        self.cache = {}

    def evaluate(self, powers):
        powers = sorted(set(map(float,powers))-self.cache.keys())
        if not powers:
            return
        g, d = self.g, self.order+1
        va = np.column_stack([boxcox_derivatives(self.ta,v,order=self.order) for v in powers])
        va -= g.nn(g.he['basis'],g.tn(g.he['basis'],va))
        contracted = g.training.component_grams(va,phase='scale_training_moments')
        va = None
        vb = np.column_stack([boxcox_derivatives(self.tb,v,order=self.order) for v in powers])
        beta = g.tn(g.weights,vb)
        vb = None
        moments = np.diagonal(contracted['gram'],axis1=1,axis2=2)[:,::d]
        theta = components_from_normalized_moments(moments/g.he['norms'][:,None],g.he)
        for j,power in enumerate(powers):
            sl = slice(j*d,(j+1)*d)
            b = beta[:,sl].copy()
            covariance = np.einsum('k,kij->ij',theta[:,j],g.grams)
            records = {}
            for name,index in self.groups.items():
                v = covariance[np.ix_(index,index)]
                record = dict(p=1.,omnibus_p=1.,sparse_p=1.,resolved=False)
                try:
                    inverse,_ = _inverse(v)
                    statistic = float(b[index,0]@inverse@b[index,0])
                    if not np.isfinite(statistic) or statistic < 0:
                        raise ValueError('unresolved Wald statistic')
                    full = float(chi2.sf(statistic,len(index)))
                    sparse = min(1.,len(index)*float(np.min(2*norm.sf(abs(b[index,0])/np.sqrt(np.diag(v))))))
                    hybrid = self.contrast_mode == 'omnibus_sparse' and len(index) > 1
                    record.update(p=min(1.,2*min(full,sparse)) if hybrid else full,
                        omnibus_p=full,sparse_p=sparse,resolved=True,statistic=statistic,
                        direction=inverse@b[index,0]/np.sqrt(statistic) if statistic>0 else np.zeros(len(index)))
                except (ValueError,np.linalg.LinAlgError):
                    pass  # A singular covariance cannot generate a rejection.
                records[name] = record
            self.cache[power] = dict(beta=b,covariance=covariance,theta=theta[:,j].copy(),
                moment_grams=contracted['gram'][:,sl,sl].copy(),trace=contracted['trace'].copy(),records=records)

    def interval(self, lower, upper):
        center,radius = (lower+upper)/2,(upper-lower)/2
        value,g = self.cache[center],self.g
        def remainder(t):
            return float(np.linalg.norm(radius**(self.order+1)/factorial(self.order+1)/(self.order+2)
                *abs(t)**(self.order+2)*np.exp(np.maximum(0.,center*t+radius*abs(t)))))
        ra,rb = remainder(self.ta),remainder(self.tb)
        diagonals = np.maximum(0.,np.diagonal(value['moment_grams'],axis1=1,axis2=2))
        delta = sum(radius**k/factorial(k)*np.sqrt(diagonals[:,k]) for k in range(1,self.order+1))
        delta += np.sqrt(value['trace'])*ra
        moment_change = 2*np.sqrt(diagonals[:,0])*delta+delta*delta
        # Strong convexity makes the constrained HE solution Lipschitz, even
        # when the active NNLS components change inside this interval.
        error = (1+1e-8)*float(np.linalg.norm(moment_change/g.he['norms']))/g.he_min_eigenvalue
        limits = {}
        for name,index in self.groups.items():
            record = value['records'][name]
            limits[name] = 1.
            if not record['resolved']:
                continue
            v = value['covariance'][np.ix_(index,index)]
            s = g.covariance_radius_matrix[np.ix_(index,index)]
            # Require every covariance in the interval to be nonsingular.
            try:
                np.linalg.cholesky(v-error*s-1e-8*np.diag(np.diag(v)))
            except np.linalg.LinAlgError:
                continue
            def lower_statistic(direction):
                signal = abs(float(direction@value['beta'][index,0]))
                change = sum(radius**k/factorial(k)*abs(float(direction@value['beta'][index,k]))
                             for k in range(1,self.order+1))
                change += rb*np.sqrt(max(0.,float(direction@g.weight_gram[np.ix_(index,index)]@direction)))
                variance = float(direction@v@direction)+error*float(direction@s@direction)
                slack = 1e-8*max(1.,signal,change)
                return max(0.,signal-change-slack)**2/(variance*(1+1e-8)) if variance>0 else 0.
            full = float(chi2.sf(lower_statistic(record['direction']),len(index)))
            if self.contrast_mode == 'omnibus_sparse' and len(index)>1:
                sparse = 1.
                for j in range(len(index)):
                    direction = np.zeros(len(index)); direction[j] = 1/np.sqrt(v[j,j])
                    sparse = min(sparse,len(index)*float(2*norm.sf(np.sqrt(lower_statistic(direction)))))
                full = min(1.,2*min(full,sparse))
            limits[name] = full
        return dict(lower=lower,upper=upper,center=center,p_upper=limits)


def boxcox_polygenic_scale_test(geometry, training_phenotype, confirmation_phenotype, *,
                                groups=None, bounds=(-2.,2.), alpha=.05, max_evaluations=129,
                                batch_size=4, order=3, contrast_mode='omnibus'):
    """Upper-bound the continuous supremum of genomic-covariance Wald p-values.

    No selection of a favorable transform, no assumption of independent pilot
    outcomes, and no frozen raw-scale covariance. The original genotype-defined
    coefficient null is tested; outcome-trained scores are not accepted by this
    scientific contract. Statistical calibration inherits the covariance model
    and plug-in consistency. The envelope only controls numerical scale search.
    """
    lower,upper = map(float,bounds)
    if (not np.isfinite(lower+upper) or lower >= upper or not np.isfinite(alpha) or not 0 < alpha < 1
            or type(max_evaluations) is not int or max_evaluations < 3
            or type(batch_size) is not int or batch_size < 2):
        raise ValueError('ordered finite bounds, threshold and valid search budget required')
    evaluator = PolygenicScaleEvaluator(geometry,training_phenotype,confirmation_phenotype,
        groups=groups,order=order,contrast_mode=contrast_mode)
    if max(abs(lower),abs(upper))*max(np.max(abs(evaluator.ta)),np.max(abs(evaluator.tb))) > 500:
        raise ValueError('Box-Cox interval exceeds supported numerical dynamic range')
    evaluator.evaluate([lower,(lower+upper)/2,upper])
    leaves = [evaluator.interval(lower,upper)]
    names = list(evaluator.groups)
    while True:
        sampled = {name:max(v['records'][name]['p'] for v in evaluator.cache.values()) for name in names}
        envelope = {name:max(v['p_upper'][name] for v in leaves) for name in names}
        unresolved = [name for name in names if sampled[name] < alpha <= envelope[name]]
        remaining = (max_evaluations-len(evaluator.cache))//2
        if not unresolved or remaining<1:
            break
        candidates = [j for j,leaf in enumerate(leaves) if any(leaf['p_upper'][name]>=alpha for name in unresolved)]
        candidates.sort(key=lambda j:max(leaves[j]['p_upper'][name] for name in unresolved),reverse=True)
        selected = candidates[:min(remaining,batch_size//2)]
        children = [(a,b) for j in selected for a,b in
                    ((leaves[j]['lower'],leaves[j]['center']),(leaves[j]['center'],leaves[j]['upper']))]
        centers = [(a+b)/2 for a,b in children]
        if any(c in evaluator.cache for c in centers):
            break
        evaluator.evaluate(centers)
        leaves = [leaf for j,leaf in enumerate(leaves) if j not in selected]
        leaves.extend(evaluator.interval(a,b) for a,b in children)
    tests = {}
    for name in names:
        best = max(evaluator.cache,key=lambda x:evaluator.cache[x]['records'][name]['p'])
        low = evaluator.cache[best]['records'][name]['p']
        high = max(low,max(leaf['p_upper'][name] for leaf in leaves))
        resolved = all(v['records'][name]['resolved'] for v in evaluator.cache.values())
        status = ('unresolved_covariance' if not resolved else 'rejected_specified_scale_family' if high<alpha
                  else 'compatible_scale_found' if low>=alpha else 'unresolved_search_bound')
        tests[name] = dict(p_sup_lower=low,p_upper=high,status=status,sampled_maximizer=best,
                          df=len(evaluator.groups[name]),threshold=alpha)
    return dict(method='continuous_boxcox_genotype_coefficients_polygenic_HE_v1',contrast_mode=contrast_mode,
        bounds=[lower,upper],log_reference=evaluator.log_reference,tests=tests,
        evaluations=len(evaluator.cache),max_evaluations=max_evaluations,derivative_order=order,
        diagnostics=geometry.diagnostics,intervals=sorted(leaves,key=lambda v:v['lower']),
        points=[dict(power=power,theta=v['theta'].tolist(),
                     p={name:r['p'] for name,r in v['records'].items()},
                     components={name:{k:r[k] for k in ('omnibus_p','sparse_p','resolved')} for name,r in v['records'].items()})
                for power,v in sorted(evaluator.cache.items())],
        null='Some common Box-Cox power in the specified interval makes the tested finite mean coefficients zero.',
        inference='Model-based marginal Wald inference with training-estimated covariance; asymptotic plug-in calibration.',
        scope='Genotype-defined fixed contrasts only. Requires a correct mean and covariance model at the null power, '
              'consistent variance estimates and Gaussian/CLT coefficient inference. Allows genetic dependence between '
              'cohorts; does not validate outcome-trained directions. Continuous bounds include NNLS boundary changes '
              'with float64 slack, not interval arithmetic. Neither arbitrary monotone invariance nor biological causality.')
