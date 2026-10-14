"""Interval calibration for single-trait binary G x E PCGC components.

Two simulation designs, both under the unit conditional liability model with
random Gaussian SNP effects whose prior covariance Omega is the truth:

fixed   Fixed Gaussian genotypes and contexts (optionally AR(1) LD blocks),
        population sampling. Effects, residuals and case status are redrawn
        for each replicate (the design of the 13 October 2026 review).
random  Genotypes, contexts and liabilities are redrawn for each replicate
        with fixed case and control counts, using an exact conditional
        sampler for Gaussian markers (case-control ascertainment).

Residual variance is set from the realized population genetic variance, so
the supplied risks are the realized population risks. Every replicate uses
exact (probe-free) reference rows and main's production fit_gxe. Reported
intervals: SNP-block jackknife, main's experimental individual-sampling SEs
(optional, with or without Gaussian architecture probes) and the
person-level cross-block correction. Results are written as compressed NumPy
archives; the driver refuses to overwrite.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def omega_matrices(q, h, annotations):
    """Prior Omega per annotation; Omega_01 is null, Omega_02 nonzero for Q=3."""
    om = np.zeros((q, q))
    om[0, 0] = h
    if q >= 2:
        om[1, 1] = h/3
    if q >= 3:
        om[2, 2] = h/4
        om[0, 2] = om[2, 0] = .3*np.sqrt(om[0, 0]*om[2, 2])
    if annotations == "one":
        return om[None]
    return np.stack((.6*om, .4*om))


def annotation_matrix(m, kind):
    if kind == "one":
        return np.ones((m, 1))
    if kind == "overlap":
        # Overlapping: all SNPs, plus a continuous weight on the first 60%.
        second = np.where(np.arange(m) < .6*m, np.linspace(.4, 1.6, m), 0.)
        return np.column_stack((np.ones(m), second))
    raise ValueError(kind)


def ld_blocks(m, ld):
    if ld == "none":
        return None
    width, rho = 16, .7
    index = np.arange(width)
    R = rho**np.abs(index[:, None]-index[None, :])
    return width, R, np.linalg.cholesky(R)


def correlate(x, blocks):
    if blocks is None:
        return x
    width, _, L = blocks
    m = x.shape[1]
    out = np.empty_like(x)
    for start in range(0, m, width):
        stop = min(m, start+width)
        out[:, start:stop] = x[:, start:stop] @ L[:stop-start, :stop-start].T
    return out


def population_effect_gram(beta, blocks):
    """beta.T R beta for population LD R."""
    if blocks is None:
        return beta.T @ beta
    width, R, _ = blocks
    out = np.zeros((beta.shape[1],)*2)
    for start in range(0, len(beta), width):
        stop = min(len(beta), start+width)
        b = beta[start:stop]
        out += b.T @ R[:stop-start, :stop-start] @ b
    return out


def context_levels(q):
    e1 = np.linspace(-1, 1, 5)
    if q == 1:
        return np.ones((1, 1)), np.ones(1)
    if q == 2:
        states = np.column_stack((np.ones(5), e1))
    else:
        states = np.array([[1., a, b] for a in e1 for b in (-1., 1.)])
    return states, np.full(len(states), 1/len(states))


def draw_effects(rng, a, omegas):
    m, q = len(a), omegas.shape[1]
    beta = np.zeros((m, q))
    for column, om in zip(a.T, omegas):
        e, v = np.linalg.eigh(om)
        root = v*np.sqrt(np.maximum(e, 0))
        beta += (rng.normal(size=(m, q)) @ root.T)*np.sqrt(column/column.sum())[:, None]
    return beta


class Setting:
    def __init__(self, spec):
        self.spec = spec
        n, m, q = spec["n"], spec["m"], spec["q"]
        self.states, self.weights = context_levels(q)
        self.a = annotation_matrix(m, spec["annotations"])
        self.omegas = omega_matrices(q, spec["h"], spec["annotations"])
        from summit.pcgc.gxe import context_pairs
        self.pairs = context_pairs(q)
        self.truth = np.array([om[u, v] for om in self.omegas for u, v in self.pairs])
        self.blocks_ld = ld_blocks(m, spec["ld"])
        self.mu = spec["risk_slope"]*(self.states[:, 1] if q > 1 else np.zeros(len(self.states)))
        K = spec["K"]
        # Unit total liability: population risk at level L is ndtr(mu_L - t).
        self.cut = brentq(lambda t: self.weights @ ndtr(self.mu-t)-K, -15, 15)
        self.k_level = ndtr(self.mu-self.cut)
        self.K = float(self.weights @ self.k_level)
        self.jk = np.asarray(np.arange(m)*spec["blocks"]//m, dtype=np.int64)
        if spec["design"] == "fixed":
            rng = np.random.default_rng(spec["seed"])
            self.level = rng.choice(len(self.states), size=n, p=self.weights)
            self.x = np.asfortranarray(correlate(rng.normal(size=(n, m)), self.blocks_ld))
            self.prepare_genotypes(self.x, self.level)

    def prepare_genotypes(self, x, level):
        a = self.a
        self.squares = x*x
        self.grams = []
        self.diag = []
        self.level_rows = [np.flatnonzero(level == L) for L in range(len(self.states))]
        base = self.squares @ a
        for rows in self.level_rows:
            xl = x[rows]
            self.grams.append(xl.T @ xl)
            self.diag.append(self.squares[rows].T @ base[rows])


def exact_moments(setting, x, risk, phi, level, method="pcgc"):
    """Exact GxE moments from per-level genotype Grams (contexts constant per level)."""
    from summit.pcgc.gxe import GxEMoments, context_pairs, population_metric
    n, m = x.shape
    q = phi.shape[1]
    pairs = context_pairs(q)
    p = len(pairs)
    a = setting.a
    k = a.shape[1]
    sd = np.ones(n)
    w = risk.sensitivity/sd
    states = setting.states
    w_level = np.array([w[rows[0]] if len(rows) else 0. for rows in setting.level_rows])
    for L, rows in enumerate(setting.level_rows):
        if len(rows) and not np.allclose(w[rows], w_level[L], rtol=1e-12, atol=0):
            raise ValueError("risk sensitivity must be constant within a context level")
    inverse = method == "pcgc-inverse"
    psi_level = states if inverse else states*w_level[:, None]
    cross = {}
    for u in range(q):
        for v in range(u, q):
            value = sum(psi_level[L, u]*psi_level[L, v]*G for L, G in enumerate(setting.grams))
            cross[u, v] = value
            cross[v, u] = value
    orient = [[(u, v)] if u == v else [(u, v), (v, u)] for u, v in pairs]
    ld = np.empty((m, p, k*p))
    for t, (u, v) in enumerate(pairs):
        hl = (1 if u == v else 2)*psi_level[:, u]*psi_level[:, v]
        for s, (c, d) in enumerate(pairs):
            product = sum(cross[j, l]*cross[i, h] for i, j in orient[t] for l, h in orient[s])
            other = (1 if c == d else 2)*psi_level[:, c]*psi_level[:, d]
            full = product @ a - sum((hl[L]*other[L])*D for L, D in enumerate(setting.diag))
            ld[:, t, s::p] = full/n**2
    psi = phi if inverse else phi*w[:, None]
    r = risk.z/w if inverse else risk.z
    scores = x.T @ (psi*r[:, None])
    rhs = np.empty((m, p))
    for t, (u, v) in enumerate(pairs):
        h = (1 if u == v else 2)*psi[:, u]*psi[:, v]
        rhs[:, t] = (1 if u == v else 2)*scores[:, u]*scores[:, v]-setting.squares.T @ (h*r*r)
    metric, variance = population_metric(phi, risk, sd)
    return GxEMoments(a, ld, rhs, q, n, method, metric, variance), psi, r


def simulate_fixed(setting, rng):
    spec = setting.spec
    beta = draw_effects(rng, setting.a, setting.omegas)
    gram = population_effect_gram(beta, setting.blocks_ld)
    v = np.einsum('lq,qr,lr->l', setting.states, gram, setting.states)
    if np.any(v >= 1):
        return None
    x, level = setting.x, setting.level
    phi = setting.states[level]
    g = np.sum((x @ beta)*phi, axis=1)
    liability = g+rng.normal(size=len(g))*np.sqrt(1-v[level])
    y = (liability > setting.cut-setting.mu[level]).astype(float)
    return x, phi, level, y


def simulate_random(setting, rng):
    """Exact retrospective sampler for Gaussian markers with fixed case counts."""
    spec = setting.spec
    n, m = spec["n"], spec["m"]
    beta = draw_effects(rng, setting.a, setting.omegas)
    gram = population_effect_gram(beta, setting.blocks_ld)
    states = setting.states
    v = np.einsum('lq,qr,lr->l', states, gram, states)
    if np.any(v >= 1):
        return None
    K, k = setting.K, setting.k_level
    cases = int(round(n*spec["case_fraction"]))
    y = np.r_[np.ones(cases), np.zeros(n-cases)]
    rng.shuffle(y)
    level = np.empty(n, dtype=int)
    for case in (0, 1):
        sel = y == case
        prob = setting.weights*(k/K if case else (1-k)/(1-K))
        level[sel] = rng.choice(len(states), size=sel.sum(), p=prob/prob.sum())
    t = setting.cut-setting.mu[level]
    u = rng.uniform(np.finfo(float).eps, 1-np.finfo(float).eps, n)
    liab = np.where(y == 1, -ndtri(u*ndtr(-t)), ndtri(u*ndtr(t)))
    x0 = correlate(rng.normal(size=(n, m)), setting.blocks_ld)
    b = (states @ beta.T)[level]
    if setting.blocks_ld is None:
        rb = b
    else:
        width, R, _ = setting.blocks_ld
        rb = np.empty_like(b)
        for start in range(0, m, width):
            stop = min(m, start+width)
            rb[:, start:stop] = b[:, start:stop] @ R[:stop-start, :stop-start]
    env = rng.normal(size=n)*np.sqrt(1-v[level])
    x = x0+(liab-np.sum(x0*b, axis=1)-env)[:, None]*rb
    x = np.asfortranarray(x)
    setting.prepare_genotypes(x, level)
    return x, states[level], level, y


def run_replicate(setting, index, options):
    from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk
    from summit.pcgc.gxe import fit_gxe, _omega
    from summit.pcgc.crossblock import CrossBlockAccumulator, crossblock_covariance
    from summit.pcgc.sampling import nuisance_derivatives
    from summit.ldscore.generalized_gxe_reference_v1 import reduce_generalized_gxe_reference_for_inference
    spec = setting.spec
    rng = np.random.default_rng([spec["seed"], 7919, index])
    for attempt in range(20):
        data = (simulate_fixed if spec["design"] == "fixed" else simulate_random)(setting, rng)
        if data is not None:
            break
    else:
        return dict(failure="no admissible effect draw")
    x, phi, level, y = data
    if y.sum() < 8 or (1-y).sum() < 8:
        return dict(failure="too few cases or controls")
    risk_covariates = None
    if spec["risk"] == "supplied":
        risk = prepare_binary_risk(y, setting.K, population_risk=setting.k_level[level])
    else:
        risk_covariates = phi[:, 1:] if phi.shape[1] > 1 else None
        if risk_covariates is None:
            risk = fit_binary_risk(y, setting.K)
        else:
            risk = fit_binary_risk(y, setting.K, risk_covariates)
    method = spec.get("method", "pcgc")
    moments, psi, response = exact_moments(setting, x, risk, phi, level, method)
    out = dict()
    fit = fit_gxe(moments, block_ids=setting.jk, native=False)
    theta = np.asarray(fit["components"])
    jk = np.asarray(fit["covariance"])
    out.update(estimate=theta, jk_var=np.diag(jk), h2=fit["population_heritability"],
               h2_jk_se=fit["population_heritability_se"])
    # Person-level cross-block correction.
    H, _ = moments.equations()
    labels = np.unique(setting.jk)
    directed, _, mass = moments.raw_equations()
    bd, _, _ = reduce_generalized_gxe_reference_for_inference(directional_ldscores=moments.ldscores,
        annotations=moments.annotations, variant_block_ids=setting.jk, block_labels=tuple(map(str, labels)),
        expected_directed_numerator=directed)
    blocks = moments.n_samples**2*bd/np.outer(mass, mass)[None]
    cases = risk.z > 0
    extra = {}
    if risk_covariates is not None:
        _, influence, log_weight, derivative = nuisance_derivatives(risk, risk_covariates, method, np.ones(len(y)))
        extra = dict(influence=influence, response_derivative=derivative, log_weight=log_weight)
    acc = CrossBlockAccumulator(setting.a, psi, response, cases, setting.jk, **extra)
    width = 512
    for start in range(0, x.shape[1], width):
        acc.add_block(start, min(x.shape[1], start+width), x[:, start:min(x.shape[1], start+width)])
    cb = acc.finalize()
    between, nuisance, details = crossblock_covariance(cb, theta, H, blocks)
    Hinv = np.linalg.inv(H)
    B = len(labels)
    add_between = Hinv @ between @ Hinv.T
    add_nuisance = Hinv @ nuisance @ Hinv.T
    out.update(between_var=np.diag(add_between)*B/(B-1), nuisance_var=np.diag(add_nuisance))
    # Total liability-scale heritability: contrast of theta with the fixed
    # population metric, as in fit_gxe; the metric and V_L are held fixed.
    metric = moments.population_second_moment
    contrast = np.array([(1 if u == v else 2)*metric[u, v] for _ in range(setting.a.shape[1]) for u, v in setting.pairs])
    total = jk+B/(B-1)*add_between+add_nuisance
    out.update(h2_new_se=float(np.sqrt(max(0., contrast @ total @ contrast)))/moments.population_liability_variance)
    if options.get("sampling"):
        out.update(sampling_replicate(setting, x, phi, level, risk, risk_covariates, psi, moments, theta,
                                      options, rng, method, response))
    out["cases"] = int(y.sum())
    return out


def sampling_replicate(setting, x, phi, level, risk, risk_covariates, psi, moments, theta, options, rng,
                       method="pcgc", response=None):
    """Main's experimental individual-sampling covariance on exact moments."""
    from dataclasses import replace
    from summit.pcgc.sampling import build_sampling_moments, partner_proposal, sample_partners
    from summit.pcgc.gxe import fit_gxe, pair_products
    a = setting.a
    n, m = x.shape
    mass = a.sum(0)
    partners_count = options["partners"]
    r = risk.z if response is None else response
    probabilities = partner_proposal(psi, r)
    seed = int(rng.integers(0, 2**31))
    partners = sample_partners(probabilities, partners_count, seed)
    rel = np.empty((n, partners_count, a.shape[1]))
    an = a/mass
    for l in range(partners_count):
        rel[:, l, :] = (x*x[partners[:, l]]) @ an
    diag = setting.squares @ an
    # Kernel actions sum_{l != i} K_c(i,l) r_l, annotation-major.
    scores = x.T @ (psi*r[:, None])
    from summit.pcgc.gxe import context_pairs
    pairs = context_pairs(phi.shape[1])
    p = len(pairs)
    actions = np.empty((n, a.shape[1]*p))
    for k in range(a.shape[1]):
        values = x @ (an[:, k, None]*scores)
        for j, (u, v) in enumerate(pairs):
            col = psi[:, u]*values[:, v]+(psi[:, v]*values[:, u] if u != v else 0)
            actions[:, k*p+j] = col-(1 if u == v else 2)*psi[:, u]*psi[:, v]*r*diag[:, k]
    sm = build_sampling_moments(pair_kernels=rel, partners=partners, partner_probabilities=probabilities,
        kernel_actions=actions, genotype_diagonal=diag, contexts=phi, features=psi, response=r, risk=risk,
        sd=np.ones(n), method=method, risk_covariates=risk_covariates, native=True, threads=1)
    out = {}
    with_sampling = replace(moments, sampling_moments=sm)
    try:
        fit = fit_gxe(with_sampling, native=True, threads=1)
        out["sampling_var"] = np.asarray(fit["standard_errors"])**2
    except ValueError as exc:
        out["sampling_failure"] = str(exc)
    if options.get("architecture"):
        from summit.pcgc.architecture import ArchitectureSketch
        from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
        from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
        from summit.prediction.genotype import native_module
        module = native_module()
        sketch = ArchitectureSketch(psi, a, r > 0, probes=options["architecture"], seed=seed,
            nn=ProtectedNNOperator(threads=1, native_module=module), tn=ProtectedTNOperator(threads=1, native_module=module),
            native=True, threads=1)
        width = 256
        for pass_number in (1, 2):
            for start in range(0, m, width):
                stop = min(m, start+width)
                sketch.read_block(pass_number, start, stop, np.asfortranarray(x[:, start:stop]))
        arch = replace(sm, architecture_left=sketch.left, architecture_right=sketch.right,
                       architecture_probes=sketch.probes)
        try:
            fit = fit_gxe(replace(moments, sampling_moments=arch), native=True, threads=1)
            out["architecture_var"] = np.asarray(fit["standard_errors"])**2
        except ValueError as exc:
            out["architecture_failure"] = str(exc)
    return out


SETTING = None


def _worker_init():
    # One BLAS thread per pool worker; the fixed design is inherited by fork.
    from threadpoolctl import threadpool_limits
    globals()["_LIMITS"] = threadpool_limits(1)


def _worker(args):
    index, options = args
    started = time.perf_counter()
    try:
        out = run_replicate(SETTING, index, options)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        out = dict(failure=f"{type(exc).__name__}: {exc}")
    out["seconds"] = time.perf_counter()-started
    out["index"] = index
    return out


def summarize(rows, truth):
    good = [r for r in rows if "failure" not in r]
    est = np.array([r["estimate"] for r in good])
    out = dict(replicates=len(rows), failures=len(rows)-len(good), truth=truth.tolist())
    if len(good) < 10:
        return out
    sd = est.std(0, ddof=1)
    out.update(mean=est.mean(0).tolist(), bias_over_mcse=((est.mean(0)-truth)/(sd/np.sqrt(len(est)))).tolist(),
               empirical_sd=sd.tolist())
    def calib(var):
        se = np.sqrt(np.maximum(var, 0))
        cover = (np.abs(est-truth) <= 1.959963984540054*se).mean(0)
        ratio = var.mean(0)/sd**2
        # Delta-method MCSE of the mean-variance / empirical-variance ratio.
        nrep = len(est)
        dev = (est-est.mean(0))**2
        g = var/sd**2-ratio*dev/sd**2
        ratio_mcse = g.std(0, ddof=1)/np.sqrt(nrep)
        return dict(variance_ratio=ratio.tolist(), variance_ratio_mcse=ratio_mcse.tolist(),
                    coverage=cover.tolist(), coverage_mcse=np.sqrt(cover*(1-cover)/nrep).tolist(),
                    rejection_at_truth=(1-cover).tolist())
    jk = np.array([r["jk_var"] for r in good])
    new = jk+np.array([r["between_var"]+r["nuisance_var"] for r in good])
    out["jackknife"] = calib(jk)
    out["crossblock"] = calib(new)
    out["jackknife_plus_nuisance_only"] = calib(jk+np.array([r["nuisance_var"] for r in good]))
    for name in ("sampling", "architecture"):
        rows_named = [r for r in good if name+"_var" in r]
        if len(rows_named) == len(good):
            out[name] = calib(np.array([r[name+"_var"] for r in rows_named]))
        elif rows_named:
            out[name+"_note"] = f"{len(good)-len(rows_named)} replicates failed"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--design", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--K", type=float, default=.02)
    parser.add_argument("--h", type=float, default=.36)
    parser.add_argument("--ld", choices=("none", "ar1"), default="none")
    parser.add_argument("--annotations", choices=("one", "overlap"), default="one")
    parser.add_argument("--risk", choices=("supplied", "fitted"), default="supplied")
    parser.add_argument("--method", choices=("pcgc", "pcgc-inverse"), default="pcgc")
    parser.add_argument("--risk-slope", type=float, default=.3)
    parser.add_argument("--case-fraction", type=float, default=.5)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--replicate-offset", type=int, default=0,
                        help="first replicate index; disjoint offsets give independent extensions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--sampling-partners", type=int, default=0)
    parser.add_argument("--architecture-probes", type=int, default=0)
    parser.add_argument("--sampling-replicates", type=int, default=None,
                        help="evaluate main's sampling SEs on the first R replicates only")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    spec = dict(design=args.design, n=args.n, m=args.m, q=args.q, K=args.K, h=args.h, ld=args.ld,
                annotations=args.annotations, risk=args.risk, risk_slope=args.risk_slope, method=args.method,
                case_fraction=args.case_fraction, blocks=args.blocks, seed=args.seed)
    options = dict(sampling=args.sampling_partners > 0, partners=args.sampling_partners,
                   architecture=args.architecture_probes)
    setting_probe = Setting(dict(spec, design="random"))  # truth/levels without genotypes
    import multiprocessing as mp
    started = time.perf_counter()
    jobs = []
    for index in range(args.replicate_offset, args.replicate_offset+args.replicates):
        opts = dict(options)
        if args.sampling_replicates is not None and index-args.replicate_offset >= args.sampling_replicates:
            opts["sampling"] = False
        jobs.append((index, opts))
    # Pool workers must resolve functions by module name, not runpy's __main__.
    import importlib
    module = importlib.import_module("scripts.pcgc.calibrate_gxe_intervals")
    module.SETTING = module.Setting(spec)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(args.workers, initializer=module._worker_init) as pool:
        for row in pool.imap_unordered(module._worker, jobs, chunksize=4):
            rows.append(row)
    rows.sort(key=lambda r: r["index"])
    summary = summarize(rows, setting_probe.truth)
    states, weights = setting_probe.states, setting_probe.weights
    mu = setting_probe.mu
    vl = 1+float(weights @ (mu-weights @ mu)**2)
    h2_truth = float(sum(np.einsum('l,lq,qr,lr->', weights, states, om, states) for om in setting_probe.omegas))/vl
    good = [r for r in rows if "failure" not in r]
    h2 = np.array([r["h2"] for r in good])
    h2_summary = dict(truth=h2_truth, mean=float(h2.mean()), empirical_sd=float(h2.std(ddof=1)))
    for name in ("h2_jk_se", "h2_new_se"):
        se = np.array([r[name] for r in good])
        h2_summary[name] = dict(variance_ratio=float(np.mean(se**2)/h2.var(ddof=1)),
                                coverage=float(np.mean(np.abs(h2-h2_truth) <= 1.959963984540054*se)))
    summary["population_heritability"] = h2_summary
    summary.update(spec=spec, options=options, population_prevalence=setting_probe.K,
                   replicate_offset=args.replicate_offset,
                   seconds=time.perf_counter()-started,
                   median_replicate_seconds=float(np.median([r["seconds"] for r in rows])),
                   failures_detail=sorted({r["failure"] for r in rows if "failure" in r}),
                   mean_cases=float(np.mean([r.get("cases", np.nan) for r in rows if "cases" in r])))
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key in ("estimate", "jk_var", "between_var", "nuisance_var", "sampling_var", "architecture_var",
                "h2", "h2_jk_se", "h2_new_se"):
        values = [r.get(key) for r in rows]
        if all(v is not None for v in values):
            arrays[key] = np.array(values)
        elif any(v is not None for v in values):
            arrays[key] = np.array([np.full(len(setting_probe.truth), np.nan) if v is None else v for v in values])
    arrays["failed"] = np.array(["failure" in r for r in rows])
    np.savez_compressed(out.with_suffix(".npz"), truth=setting_probe.truth, **arrays)
    with open(out, "x") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("spec", "seconds", "failures", "mean_cases")}))
    for name in ("jackknife", "crossblock", "sampling", "architecture"):
        if name in summary:
            print(name, "ratio", np.round(summary[name]["variance_ratio"], 3).tolist(),
                  "cover", np.round(summary[name]["coverage"], 3).tolist())


if __name__ == "__main__":
    main()
