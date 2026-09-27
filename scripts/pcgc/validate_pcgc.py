#!/usr/bin/env python3
"""Bounded, paired PCGC qualification; retain compact rows, never genotype pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import norm, t

from summit.sumstats.binary import fit_binary_risk, prepare_binary_risk
from summit.pcgc.moments import fit_moments
from summit.pcgc.research import exact_moments, exact_external_ld, external_ld_moments


def summit_he_baseline(x, annotations, y, prevalence, external_ld, nblocks):
    """Call the unchanged SUMMIT HE machinery, then apply global conversion."""
    from summit.inference.h2core import prepare_h2, fit_h2
    from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
    n, m = x.shape
    trace = SimpleNamespace(nsnps=m, nbins=annotations.shape[1], snps=np.arange(m),
                            annot=annotations, ldscores=external_ld, delta=None,
                            annot_header=[f"annotation_{i}" for i in range(annotations.shape[1])])
    matched = SimpleNamespace(nsnps=m, snps=trace.snps, nsamp=n, n_scale=n-1, n=np.full(m,n))
    jackknife = JackknifeDesign.from_trace_view(trace, JackknifeSpec.parse(nblocks))
    genotype = (x-x.mean(axis=0))/x.std(axis=0)
    response = (y-y.mean())/y.std()
    matched.beta = genotype.T @ response/n
    matched.se = np.sqrt((1-matched.beta**2)/(n-2))
    prepared = prepare_h2(trace, matched, jackknife)
    fit = fit_h2(prepared, report_tau=False)
    correction = fit_binary_risk(y, prevalence).sensitivity[0]**2
    components = fit.h2_reps[-1, :-1]/correction
    return dict(method="summit-he-global", conditional_components=components.tolist(),
                marginal_components=components.tolist(), conditional_total=float(components.sum()),
                marginal_total=float(components.sum()), conditional_total_standard_error=float(fit.h2[-1, 1]/correction),
                uncertainty_status="baseline", estimand="marginal_global_conversion")


SCENARIOS = {
    "S0": dict(K=.1, P=.1, cov="none", h=[.25]),
    "S1": dict(K=.1, P=.5, cov="none", h=[.25]),
    "S2": dict(K=.1, P=.5, cov="binary", h=[.25]),
    "S3": dict(K=.1, P=.5, cov="continuous", h=[.25]),
    "S4": dict(K=.01, P=.5, cov="strong", h=[.25]),
    "S5": dict(K=.1, P=.5, cov="binary", h=[.05, .20]),
    "S6": dict(K=.1, P=.5, cov="dependent", h=[.25]),
    "S7": dict(K=.1, P=.5, cov="strong", h=[0.]),
}


def ar_parameters(m, block_size):
    if m % block_size:
        raise ValueError("variants must be divisible by LD block size")
    rho = np.repeat(np.resize([.2, .7], m//block_size), block_size)
    rho[::block_size] = 0
    R = np.zeros((m, m))
    for start in range(0, m, block_size):
        r = rho[start+1]
        R[start:start+block_size, start:start+block_size] = r**np.abs(np.subtract.outer(np.arange(block_size), np.arange(block_size)))
    return rho, R


def draw_markers(rng, n, rho):
    x = rng.normal(size=(n, len(rho)))
    for j in range(1, len(rho)):
        x[:, j] = rho[j]*x[:, j-1] + np.sqrt(1-rho[j]**2)*x[:, j]
    return x


def truncated_liability(rng, y, cut):
    """Inverse-CDF conditional draws, with case tails computed by survival."""
    u = rng.uniform(np.finfo(float).eps, 1-np.finfo(float).eps, len(y))
    return np.where(y == 1, -ndtri(u*ndtr(-cut)), ndtri(u*ndtr(cut)))


def generate(seed, scenario, n, m, nref, block_size):
    rng = np.random.default_rng(seed)
    spec = SCENARIOS[scenario]
    rho, R = ar_parameters(m, block_size)
    annotations = np.ones((m, len(spec["h"])))
    if len(spec["h"]) == 2:
        groups = (np.arange(m)//block_size) % 2
        annotations = np.column_stack([groups == i for i in range(2)]).astype(float)
    beta = rng.normal(size=m)
    for j, h in enumerate(spec["h"]):
        b = beta * annotations[:, j]
        beta[annotations[:, j] > 0] *= np.sqrt(h/(b @ R @ b)) if h else 0
    realized = np.array([(beta*a) @ R @ (beta*a) for a in annotations.T])
    component_parameter = np.array([np.sum((beta*a)**2) for a in annotations.T])
    h = realized.sum()
    y = np.r_[np.ones(int(round(n*spec["P"]))), np.zeros(n-int(round(n*spec["P"])))]
    rng.shuffle(y)
    cov_kind = spec["cov"]
    gamma = 0 if cov_kind == "none" else (1 if cov_kind == "strong" else .5)
    Vc = gamma**2
    x0 = draw_markers(rng, n, rho)
    e0 = rng.normal(size=n)*np.sqrt(1-h)
    if cov_kind == "binary":
        threshold = brentq(lambda t: .5*(norm.sf(t-gamma)+norm.sf(t+gamma))-spec["K"], -10, 10)
        kp, km = norm.sf(threshold-gamma), norm.sf(threshold+gamma)
        prob_plus = np.where(y == 1, .5*kp/spec["K"], .5*(1-kp)/(1-spec["K"]))
        cov = np.where(rng.uniform(size=n) < prob_plus, 1., -1.)
        liability = truncated_liability(rng, y, threshold-gamma*cov)
        x = x0 + (liability-x0@beta-e0)[:, None]*(R@beta)[None, :]
    else:
        threshold = np.sqrt(1+Vc)*norm.isf(spec["K"])
        c0 = rng.normal(size=n)
        liability = np.sqrt(1+Vc)*truncated_liability(rng, y, threshold/np.sqrt(1+Vc))
        innovation = (liability-x0@beta-gamma*c0-e0)/(1+Vc)
        x = x0 + innovation[:, None]*(R@beta)[None, :]
        cov = c0 + innovation*gamma
    k = ndtr(gamma*cov-threshold)
    if cov_kind == "dependent":
        # Explicitly violated risk/kernel-independence sentinel. The fitted
        # raw kernel includes a known ancestry mean shift; residual liability
        # was generated on within-stratum markers. This is not a valid-use gate.
        x += .2*cov[:, None]*np.resize([-1., 1.], m)[None, :]
    ref = draw_markers(rng, nref, rho)
    return dict(x=x, y=y, cov=cov, k=k, Vc=Vc, annotations=annotations, reference=ref,
                truth=realized, random_effect_parameter=component_parameter, beta=beta,
                rho=rho, R=R, threshold=threshold, gamma=gamma)


def fit_row(moments, blocks):
    result = fit_moments(moments, block_ids=blocks)
    result.pop("jackknife_replicates", None)
    return result


def replicate(seed, scenario, args):
    start = time.perf_counter()
    if args.generator == "discrete":
        from discrete import generate as selected_generator
    else:
        selected_generator = generate
    data = selected_generator(seed, scenario, args.samples, args.variants, args.reference_samples, args.ld_block)
    x, a, y = data["x"], data["annotations"], data["y"]
    blocks = np.floor(np.arange(len(a))*args.jackknife/len(a)).astype(int)
    external = exact_external_ld(data["reference"], a)
    risks = {"supplied": prepare_binary_risk(y, SCENARIOS[scenario]["K"], population_risk=data["k"], covariate_variance=data["Vc"])}
    try:
        risks["fitted"] = fit_binary_risk(y, SCENARIOS[scenario]["K"], None if data["gamma"] == 0 else data["cov"][:, None])
    except (ValueError, RuntimeError) as exc:
        risks["fitted"] = str(exc)
    output = []
    reference_cache = {}
    for risk_name, risk in risks.items():
        if isinstance(risk, str):
            for method in ("pcgc", "pcgc-inverse", "pcgc-ld", "pcgc-basis-4"):
                output.append(dict(seed=seed, scenario=scenario, risk=risk_name, method=method,
                                   truth=data["truth"].tolist(), marginal_truth=(data["truth"]/(1+data["Vc"])).tolist(),
                                   failure="risk fit: "+risk))
            continue
        standard = exact_moments(x, a, risk, reference_cache=reference_cache)
        methods = {"pcgc": standard,
                   "pcgc-inverse": exact_moments(x, a, risk, "pcgc-inverse", reference_cache=reference_cache),
                   "pcgc-ld": external_ld_moments(standard.rhs_rows, a, risk, external)}
        # Piecewise-constant positive basis, deterministic given fitted risks.
        labels = np.searchsorted(np.quantile(risk.sensitivity, [.25, .5, .75]), risk.sensitivity)
        dhat = np.array([risk.sensitivity[labels == g].mean() for g in labels])
        methods["pcgc-basis-4"] = exact_moments(x, a, risk, "pcgc-basis", sensitivity=dhat, reference_cache=reference_cache)
        for method, moments in methods.items():
            row = dict(seed=seed, scenario=scenario, risk=risk_name, method=method,
                       truth=data["truth"].tolist(), random_effect_parameter=data["random_effect_parameter"].tolist(),
                       marginal_truth=(data["truth"]/(1+data["Vc"])).tolist(),
                       risk_diagnostics=risk.diagnostics(),
                       basis_relative_error=float(np.linalg.norm(dhat-risk.sensitivity)/np.linalg.norm(risk.sensitivity)))
            try:
                row.update(fit_row(moments, blocks))
                row["method"] = method
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row["failure"] = str(exc)
            output.append(row)
    constant = fit_binary_risk(y, SCENARIOS[scenario]["K"])
    row = dict(seed=seed, scenario=scenario, risk="global", method="liability",
               truth=data["truth"].tolist(), marginal_truth=(data["truth"]/(1+data["Vc"])).tolist())
    row.update(fit_row(exact_moments(x, a, constant, "liability", reference_cache=reference_cache), blocks))
    output.append(row)
    row = dict(seed=seed, scenario=scenario, risk="global", truth=data["truth"].tolist(),
               marginal_truth=(data["truth"]/(1+data["Vc"])).tolist())
    row.update(summit_he_baseline(x, a, y, SCENARIOS[scenario]["K"], external, args.jackknife))
    output.append(row)
    for row in output:
        row["replicate_seconds"] = time.perf_counter()-start
        row["generator_diagnostics"] = data.get("generator_diagnostics", {})
    return output


def summarize(rows):
    result = []
    keys = sorted({(r["scenario"], r["risk"], r["method"]) for r in rows})
    for key in keys:
        group = [r for r in rows if (r["scenario"], r["risk"], r["method"]) == key]
        good = [r for r in group if "failure" not in r]
        item = dict(zip(("scenario", "risk", "method"), key), replicates=len(group), failures=len(group)-len(good))
        if len(good) >= 2:
            est = np.array([r["marginal_total"] for r in good])
            truth = np.array([sum(r["marginal_truth"]) for r in good])
            se = np.array([r["conditional_total_standard_error"]/(1+r.get("risk_diagnostics", {}).get("covariate_variance", 0)) for r in good])
            err = est-truth
            mc = err.std(ddof=1)/np.sqrt(len(err))
            margin = max(.02, .1*truth.mean())
            bias_ci = err.mean()+np.array([-1, 1])*t.ppf(.975, len(err)-1)*mc
            item.update(estimand="marginal", mean=float(est.mean()), bias=float(err.mean()), empirical_sd=float(est.std(ddof=1)),
                        rmse=float(np.sqrt(np.mean(err**2))), bias_mcse=float(mc), bias_ci=bias_ci.tolist(),
                        bias_margin=margin, bias_equivalent=bool(np.all(np.abs(bias_ci) < margin)),
                        rms_se=float(np.sqrt(np.mean(se**2))), se_sd_ratio=float(np.sqrt(np.mean(se**2))/est.std(ddof=1)),
                        coverage=float(np.mean(np.abs(err) <= 1.96*se)),
                        null_rejection=float(np.mean(np.abs(est) > 1.96*se)))
        result.append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["S2"])
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--variants", type=int, default=4000)
    p.add_argument("--reference-samples", type=int, default=2000)
    p.add_argument("--ld-block", type=int, default=40)
    p.add_argument("--jackknife", type=int, default=50)
    p.add_argument("--generator", choices=("gaussian", "discrete"), default="gaussian")
    args = p.parse_args()
    if not 1 <= args.replicates <= 100 or not 0 <= args.seed_offset <= 100-args.replicates:
        p.error("use at most 100 unique replicates per scenario")
    if not 2 <= args.jackknife <= args.variants or args.samples < 40 or args.reference_samples < 40:
        p.error("invalid sample or jackknife dimensions")
    args.out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("discrete.py"), root/"src/summit/sumstats/binary.py", *sorted((root/"src/summit/pcgc").glob("*.py"))]
    manifest = dict(schema="summit.pcgc.qualification.v1", revision=revision,
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    scenarios=SCENARIOS, generator=args.generator,
                    reference="independent_population", reference_estimator="exact",
                    seed_rule="820000 + 1000*scenario_number + replicate_index",
                    threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for scenario in args.scenarios:
        scenario_rows = []
        for r in range(args.seed_offset, args.seed_offset+args.replicates):
            seed = 820000+1000*int(scenario[1:])+r
            output = replicate(seed, scenario, args)
            rows.extend(output)
            scenario_rows.extend(output)
            print(json.dumps(dict(scenario=scenario, replicate=r, seconds=output[0]["replicate_seconds"])), flush=True)
        (args.out/f"{scenario}.json").write_text(json.dumps(scenario_rows, indent=2, allow_nan=False)+"\n")
    (args.out/"replicates.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    report = dict(summary=summarize(rows), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.out/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
