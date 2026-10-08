"""Matched, bounded diagnosis of conditional epistasis bias.

One real-genotype panel, nested training samples, and an unchanged confirmation
sample. The same outcomes are assessed under random-effect conditioning and
conditioning additionally on the realized genetic coefficients. Generating
truth enters diagnostics and labeled oracle controls only. No production
estimator or empirical standard-error correction is changed here.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space, orth
from scipy.optimize import nnls
from scipy.stats import norm

from summit.prediction.genotype import FileGenotypeSource, StandardizedBlock, estimate_scale, native_module
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.cli import _aligned_table
from summit.prediction.artifacts import file_digest
from summit.context.spec import array_sha256
from scripts.epistasis.cpu_policy import require_tree_affinity
from scripts.epistasis.full_matched import write_json
from scripts.epistasis.conditional_polygenic_reference import covariance_geometry, fit_prepared_covariance


def normalized_basis(c):
    norms = np.linalg.norm(c, axis=0)
    return orth(c[:, norms > 0] / norms[norms > 0], rcond=1e-11)


def restricted_geometry(kernels, fixed, n0, probes, seed):
    u = normalized_basis(fixed[:n0])
    q = null_space(u.T, rcond=1e-11)
    k00 = np.stack([q.T @ k[:n0, :n0] @ q for k in kernels])
    k10 = np.stack([k[n0:, :n0] @ q for k in kernels])
    exact = np.einsum('aij,bij->ab', k00, k00)
    # Same projected Rademacher Gram estimator as production he_geometry.
    z = q.T @ np.random.default_rng(seed).choice([-1., 1.], (n0, probes))
    products = k00 @ z
    sketch = np.einsum('anb,cnb->ac', products, products) / probes
    return dict(q=q, k00=k00, k10=k10, exact=exact, sketch=sketch)


def estimate(moments, gram):
    norms = np.sqrt(np.diag(gram))
    h = gram / norms[:, None] / norms[None, :]
    if np.linalg.cond(h) > 1e8:
        raise ValueError('covariance moment system unidentified')
    values, vectors = np.linalg.eigh(h)
    root = np.sqrt(values)[:, None] * vectors.T
    rhs = (vectors.T @ (moments / norms)) / np.sqrt(values)
    return nnls(root, rhs)[0] / norms


def transfer(geometry, theta):
    k00, k10 = geometry['k00'], geometry['k10']
    v00 = np.einsum('k,kij->ij', theta, k00)
    v10 = np.einsum('k,kij->ij', theta, k10)
    factor = cho_factor(v00, lower=True)
    m = cho_solve(factor, v10.T).T
    return m, factor, v00, v10


def conditional_bias_components(a, genetic0, genetic1, noise0, y0,
                                true_transfer, fitted_transfer):
    """Exact identity, holding the fitted contrast a and its units fixed."""
    architecture = float(a @ (genetic1 - true_transfer @ genetic0))
    training_noise = float(-a @ (true_transfer @ noise0))
    covariance = float(a @ ((true_transfer - fitted_transfer) @ y0))
    total = float(a @ (genetic1 - fitted_transfer @ y0))
    return dict(architecture=architecture, training_noise=training_noise,
                covariance=covariance, oracle_remaining=architecture+training_noise,
                total=total, identity_error=total-architecture-training_noise-covariance)


def exact_tangent_remainder(a, y0, mhat, fitted_factor, fitted_v00,
                            fitted_v10, true_v00, true_v10):
    """Exact resolvent identity; no small-error assumption in this equality."""
    da = true_v00 - fitted_v00
    db = true_v10 - fitted_v10
    left = a @ (db - mhat @ da)
    alpha = cho_solve(fitted_factor, y0)
    first = float(left @ alpha)
    true_alpha = cho_solve(cho_factor(true_v00, lower=True), y0)
    remainder = float(-left @ cho_solve(fitted_factor, da @ true_alpha))
    return first, remainder


def tail(mean, actual_variance, reported_variance, alpha):
    threshold = norm.isf(alpha/2) * np.sqrt(reported_variance)
    sd = np.sqrt(actual_variance)
    return float(norm.cdf((-threshold-mean)/sd) + norm.sf((threshold-mean)/sd))


def make_design(reference, sizes, n1, markers, seed):
    meta = json.loads((reference/'reference.json').read_text())
    with np.load(reference/'reference.npz', allow_pickle=False) as saved:
        available = saved['rows']
    n0max = max(sizes)
    n = n0max+n1
    if n > 2048 or min(sizes) < 96:
        raise ValueError('bounded design requires total N<=2048 and training N>=96')
    rng = np.random.default_rng(seed)
    native = native_module()
    configure_prediction_threads(native, 1)
    with FileGenotypeSource(meta['genotypes'], genome_build='GRCh37') as source:
        if source.identity != meta['source_identity']:
            raise ValueError('reference genotype identity changed')
        rows = np.sort(rng.choice(available, n, replace=False))
        order = rng.permutation(n)
        j = source.variants.ids.index(meta['target'])
        local = np.array([source.variants.ids.index(v) for v in meta['local_variants']])
        marker = np.sort(rng.choice(np.delete(np.arange(len(source.variants.ids)), j), markers, replace=False))
        selected = np.unique(np.r_[marker, local, j])
        # Freeze genotype-only scales across the sample-size experiment.
        scale = estimate_scale(source, rows[order[:n0max]], selected, threads=1, block_size=128)
        source.prepare(rows, len(selected), 1)
        raw = source.read(selected).copy(order='F')
        g = StandardizedBlock(native, 1).prepare(raw, np.arange(n), np.arange(len(selected)),
                scale.mean, scale.inverse_scale).copy()
        h = (raw == 1).astype(float)
        obs = raw != -127
        take = order[:n0max]
        mu = np.sum(h[take]*obs[take], axis=0)/obs[take].sum(0)
        h = np.where(obs, h-mu, 0.)
        spread = np.sqrt(np.sum(h[take]**2, axis=0)/obs[take].sum(0))
        spread[spread == 0] = 1.
        h /= spread
        samples = [source.samples[i] for i in rows]
        cv = _aligned_table(reference/'covariates.tsv', samples)[meta['covariates']].to_numpy(float)
        g, h, cv = g[order], h[order], cv[order]
        z = cv[:, meta['covariates'].index('PC1')]
        z = z / np.std(z[:n0max])
        x = g[:, np.searchsorted(selected, j)]
        gm, hm = g[:, np.searchsorted(selected, marker)], h[:, np.searchsorted(selected, marker)]
        c = np.column_stack([np.ones(n), cv, z*z,
                g[:, np.searchsorted(selected, local)], h[:, np.searchsorted(selected, local)]])
        bg = np.array([i for i, v in enumerate(marker) if source.variants.chromosome[v] != source.variants.chromosome[j]])
        gb = gm[:, bg]
    designs = [gm, hm, gm*z[:, None]]
    kernels = np.stack([*(d@d.T/markers for d in designs), np.eye(n), np.diag(x*x)])
    kbg = gb@gb.T/len(bg)
    theta = np.array([.8, .5, .5, .4, .6])
    noise_var = .4+.6*x*x
    receipt = dict(source_identity=meta['source_identity'], target=meta['target'],
        sample_rows=rows[order].tolist(), markers=marker.tolist(),
        kernel_sha256=array_sha256(kernels), fixed_sha256=array_sha256(c),
        theta=theta.tolist(), training_sizes=sizes, confirmation_samples=n1,
        genotype_scaling='fixed maximum-training genotype-only scales',
        covariance_basis=['additive','dominance','PC1-dependent additive','iid','target-squared noise'],
        fixed_mean='supplied covariates, PC1 squared, all local additive and heterozygote columns',
        scope='bounded adaptive reference; fixed genotypes, shared architectures/noise across N; not a full-marker production qualification')
    return kernels, c, designs, x, kbg, theta, noise_var, receipt


def fit_contrast(geometry, theta, y0, f0, f1, nuisance, *, tangent):
    m, factor, v00, v10 = transfer(geometry, theta)
    if tangent:
        alpha = cho_solve(factor, y0)
        tangents = np.column_stack([(b-m@k)@alpha for b,k in zip(geometry['k10'],geometry['k00'])])
        nuisance = np.column_stack([nuisance, tangents])
    u = normalized_basis(nuisance)
    d = f1-m@f0
    d -= u@(u.T@d)
    mass = float(d@d)
    if mass < 1e-20:
        raise ValueError('absorbed interaction response')
    return d/mass, m, factor, v00, v10, mass, u.shape[1]


def summarize(records, bootstrap_seed):
    out = []
    for n0 in sorted({r['n0'] for r in records}):
        for method in sorted({r['method'] for r in records}):
            group = [r for r in records if r['n0']==n0 and r['method']==method]
            a = lambda key: np.array([r[key] for r in group])
            # Independent architecture is the resampling unit. Noise repeats
            # remain together; do not count paired methods/N as replications.
            ids = sorted({r['architecture_id'] for r in group})
            draw = np.random.default_rng(bootstrap_seed).integers(len(ids),size=(2000,len(ids)))
            result = dict(n0=n0,method=method,learners=len(group),architectures=len(ids))
            for key in ('random_tail_05','random_tail_005','fixed_tail_05','fixed_tail_005',
                        'random_zero_bias_tail_005','fixed_oracle_mean_tail_005'):
                by_arch = np.array([np.mean([r[key] for r in group if r['architecture_id']==i]) for i in ids])
                result[key] = float(by_arch.mean())
                result[key+'_ci95'] = np.quantile(by_arch[draw].mean(1),[.025,.975]).tolist()
            for key in ('fixed_bias_z','covariance_bias_z','architecture_bias_z','training_noise_bias_z','oracle_remaining_z',
                        'random_sd_ratio','fixed_sd_ratio','relative_theta_error','trace_bias_z','phenotype_noise_bias_z',
                        'fixed_moment_bias_z'):
                result[key+'_rms'] = float(np.sqrt(np.mean(a(key)**2)))
                result[key+'_mean'] = float(np.mean(a(key)))
            result['bias_squared_cross_term'] = float(2*np.mean(a('covariance_bias_z')*a('oracle_remaining_z')))
            result['oracle_transfer_bias_mse_reduction'] = float(1-np.mean(a('oracle_remaining_z')**2)/np.mean(a('fixed_bias_z')**2))
            result['boundary_fraction'] = np.mean(np.array([r['theta'] for r in group])<1e-8,axis=0).tolist()
            result['max_decomposition_error'] = float(max(r['decomposition_error'] for r in group))
            out.append(result)
    return out


def run(args):
    begin = time.perf_counter()
    allocation=set(os.sched_getaffinity(0))
    require_tree_affinity(os.getpid(),allocation)
    args.out.mkdir(parents=True, exist_ok=False)
    sizes = [int(v) for v in args.training_sizes.split(',')]
    methods=args.methods.split(',')
    allowed={'known_tangent','he','he_tangent','probe_he_tangent','weighted_he_tangent','weighted_probe_he_tangent'}
    if not methods or set(methods)-allowed:
        raise ValueError('unknown comparison method')
    kernels,c,designs,x,kbg,truth,noisevar,receipt = make_design(
        args.reference,sizes,args.confirmation_samples,args.markers,args.design_seed)
    receipt.update(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        script_sha256=file_digest(__file__),
        estimand='null learned-interaction coefficient; two explicitly different conditioning laws',
        decomposition='fixed fitted contrast: fixed bias = oracle architecture remainder + oracle training-noise transfer + fitted covariance displacement',
        caveat='noncentral expected HE moments define an architecture-specific noiseless fitting criterion, not E[constrained HE]; oracle values are diagnostics only')
    write_json(args.out/'design.json',receipt)
    nmax=max(sizes)
    n=nmax+args.confirmation_samples
    # The same independent architectures and environmental draws at every N.
    outcomes=[]
    for architecture_id in range(args.architectures):
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,architecture_id,0]))
        genetic=sum(np.sqrt(t)*(d@rng.normal(size=args.markers))/np.sqrt(args.markers)
                    for t,d in zip(truth,designs))
        for noise_id in range(args.noise_repeats):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,architecture_id,noise_id,1]))
            noise=np.sqrt(noisevar)*rng.normal(size=n)
            outcomes.append((architecture_id,noise_id,genetic,noise))
    records=[]
    geometry_records=[]
    for n0 in sizes:
        selected=np.r_[np.arange(n0),np.arange(nmax,n)]
        k=kernels[:,selected][:,:,selected]
        cc=c[selected]
        geom=restricted_geometry(k,cc,n0,args.probes,args.probe_seed)
        weighted=(covariance_geometry(cc[:n0],k[:,:n0,:n0],diagonal_preconditioning=True)
                  if any(m.startswith('weighted_') for m in methods) else None)
        if weighted is not None:
            wz=weighted['q'].T@np.random.default_rng(args.probe_seed).choice([-1.,1.],(n0,args.probes))
            wp=weighted['projected']@wz
            weighted_sketch=np.einsum('anb,cnb->ac',wp,wp)/args.probes
        q=geom['q'];k00=geom['k00'];k10=geom['k10']
        learner=q.T@(.5*k[0,:n0,:n0]+.05*(x[:n0,None]*kbg[:n0,:n0]*x[None,:n0])+np.eye(n0))@q
        learner_factor=cho_factor(learner,lower=True)
        mt,ft,v00t,v10t=transfer(geom,truth)
        v11t=np.einsum('k,kij->ij',truth,k[:,n0:,n0:])
        st=v11t-mt@v10t.T
        r0=q.T@(noisevar[:n0,None]*q)
        r1=noisevar[nmax:]
        expected_noise=np.einsum('aij,ji->a',k00,r0)
        true_moments=geom['exact']@truth
        pseudo_probe=estimate(true_moments,geom['sketch'])
        kv=k00@v00t
        moment_covariance=2*np.einsum('aij,bji->ab',kv,kv)
        inverse_gram=np.linalg.inv(geom['exact'])
        component_covariance=inverse_gram@moment_covariance@inverse_gram
        geometry_records.append(dict(n0=n0,restricted_df=q.shape[1],
            exact_condition=float(np.linalg.cond(geom['exact']/np.sqrt(np.outer(np.diag(geom['exact']),np.diag(geom['exact']))))),
            relative_probe_gram_error=float(np.linalg.norm(geom['sketch']-geom['exact'])/np.linalg.norm(geom['exact'])),
            probe_population_theta=pseudo_probe.tolist(),
            unconstrained_HE_component_sd=np.sqrt(np.diag(component_covariance)).tolist(),
            unconstrained_HE_relative_rms_error=float(np.sqrt(np.trace(component_covariance))/np.linalg.norm(truth))))
        architecture_transfers={}
        for architecture_id,noise_id,genetic,noise in outcomes:
            require_tree_affinity(os.getpid(),allocation)
            y=genetic[selected]+noise[selected]
            y0=q.T@y[:n0]
            g0=q.T@genetic[:n0];g1=genetic[nmax:];eps0=q.T@noise[:n0]
            if architecture_id not in architecture_transfers:
                moments_fixed=np.einsum('i,aij,j->a',g0,k00,g0)+expected_noise
                theta_fixed=estimate(moments_fixed,geom['exact'])
                architecture_transfers[architecture_id]=transfer(geom,theta_fixed)[0]
            mfixed=architecture_transfers[architecture_id]
            moments=np.einsum('i,aij,j->a',y0,k00,y0)
            he=estimate(moments,geom['exact']);probe=estimate(moments,geom['sketch'])
            mhe=transfer(geom,he)[0]
            estimates={'known_tangent':truth,'he':he,'he_tangent':he,'probe_he_tangent':probe}
            if weighted is not None:
                estimates['weighted_he_tangent']=fit_prepared_covariance(y[:n0],weighted)[0]
                wy=weighted['q'].T@(y[:n0]*weighted['multiplier'])
                wm=np.einsum('i,aij,j->a',wy,weighted['projected'],wy)
                estimates['weighted_probe_he_tangent']=estimate(wm,weighted_sketch)
                mweighted=transfer(geom,estimates['weighted_he_tangent'])[0]
            dual=q@cho_solve(learner_factor,y0)
            e=.05*kbg[selected,:n0]@(x[:n0]*dual)
            f=x[selected]*e
            pgs=.5*k[0,:,:n0]@dual
            nuisance=np.column_stack([cc[n0:],e[n0:],pgs[n0:]])
            f0=q.T@f[:n0];f1=f[n0:]
            for method in methods:
                theta=estimates[method];tangent=method!='he'
                a,m,fac,v00,v10,mass,rank=fit_contrast(geom,theta,y0,f0,f1,nuisance,tangent=tangent)
                v11=np.einsum('k,kij->ij',theta,k[:,n0:,n0:])
                variance=float(a@v11@a-(a@m)@v10.T@a)
                if variance<=0:
                    raise ArithmeticError('nonpositive estimated variance')
                se=np.sqrt(variance)
                random_var=float(a@st@a);fixed_var=float((a*a)@r1)
                parts=conditional_bias_components(a,g0,g1,eps0,y0,mt,m)
                first,second=exact_tangent_remainder(a,y0,m,fac,v00,v10,v00t,v10t)
                delta=parts['covariance']
                np.testing.assert_allclose(delta,first+second,atol=2e-9,rtol=2e-8)
                np.testing.assert_allclose(parts['identity_error'],0.,atol=2e-9)
                if tangent:
                    np.testing.assert_allclose(first,0.,atol=2e-8*se)
                # Further exact split: genotype-probe error, training phenotype
                # noise relative to the fixed-architecture population criterion,
                # and its discrepancy from the random-effect covariance truth.
                trace_bias=float(a@(mhe-m)@y0) if method=='probe_he_tangent' else 0.
                # For weighted HE, this records the change of estimating
                # equation separately from the exact unweighted HE transfer.
                weighting_bias=float(a@(mhe-m)@y0) if method=='weighted_he_tangent' else 0.
                if method=='weighted_probe_he_tangent':
                    trace_bias=float(a@(mweighted-m)@y0)
                    weighting_bias=float(a@(mhe-mweighted)@y0)
                phenotype_bias=float(a@(mfixed-mhe)@y0) if method!='known_tangent' else 0.
                moment_bias=float(a@(mt-mfixed)@y0) if method!='known_tangent' else 0.
                if method!='known_tangent':
                    np.testing.assert_allclose(delta,trace_bias+weighting_bias+phenotype_bias+moment_bias,atol=2e-9,rtol=2e-8)
                row=dict(n0=n0,architecture_id=architecture_id,noise_id=noise_id,method=method,
                    theta=theta.tolist(),relative_theta_error=float(np.linalg.norm(theta-truth)/np.linalg.norm(truth)),
                    beta=float(a@(y[n0:]-m@y0)),se=float(se),response_information=mass,nuisance_rank=rank,
                    fixed_bias_z=parts['total']/se,covariance_bias_z=delta/se,
                    architecture_bias_z=parts['architecture']/se,training_noise_bias_z=parts['training_noise']/se,
                    oracle_remaining_z=parts['oracle_remaining']/se,
                    trace_bias_z=trace_bias/se,phenotype_noise_bias_z=phenotype_bias/se,fixed_moment_bias_z=moment_bias/se,
                    weighting_change_z=weighting_bias/se,
                    random_sd_ratio=float(np.sqrt(random_var)/se),fixed_sd_ratio=float(np.sqrt(fixed_var)/se),
                    decomposition_error=float(max(abs(parts['identity_error']),abs(delta-first-second))),
                    tangent_first_order_z=first/se,tangent_remainder_z=second/se,
                    random_zero_bias_tail_005=tail(0.,random_var,variance,.005),
                    fixed_oracle_mean_tail_005=tail(parts['oracle_remaining'],fixed_var,variance,.005))
                for label,alpha in [('05',.05),('005',.005)]:
                    row['random_tail_'+label]=tail(delta,random_var,variance,alpha)
                    row['fixed_tail_'+label]=tail(parts['total'],fixed_var,variance,alpha)
                records.append(row)
        print(json.dumps(dict(n0=n0,completed_learners=len(outcomes),seconds=round(time.perf_counter()-begin,2))),flush=True)
    summaries=summarize(records,args.seed+91)
    write_json(args.out/'results.json',dict(records=records,summaries=summaries,geometry=geometry_records,
        seconds=time.perf_counter()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        inference='All rates are exact Gaussian conditional probabilities averaged over the specified development learners; architecture-cluster bootstrap intervals measure finite-learner uncertainty.'))
    print(json.dumps(dict(completed_fits=len(records),summary_groups=len(summaries),
        seconds=round(time.perf_counter()-begin,2))),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training-sizes',default='128,256,512,1024')
    p.add_argument('--confirmation-samples',type=int,default=512)
    p.add_argument('--markers',type=int,default=512)
    p.add_argument('--methods',default='known_tangent,he,he_tangent,probe_he_tangent')
    p.add_argument('--architectures',type=int,default=32)
    p.add_argument('--noise-repeats',type=int,default=2)
    p.add_argument('--probes',type=int,default=128)
    p.add_argument('--design-seed',type=int,default=927613)
    p.add_argument('--probe-seed',type=int,default=871631)
    p.add_argument('--seed',type=int,default=714109)
    run(p.parse_args())


if __name__=='__main__':
    main()
