#!/usr/bin/env python3
"""Paired sensitivity checks on existing pilot seeds; no calibration claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from validate_pcgc import generate, draw_markers, truncated_liability, SCENARIOS
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk, fit_binary_risk_logistic
from summit.pcgc.moments import BinaryMoments, fit_moments, external_ld_moments
from summit.pcgc.research import exact_moments, exact_external_ld


def matched_reference(data, spec, seed, n):
    """Independent people, same fixed effects and ascertainment as the study."""
    rng = np.random.default_rng(seed)
    gamma, threshold, beta, R = (data[k] for k in ("gamma", "threshold", "beta", "R"))
    h = sum(data["truth"])
    y = np.r_[np.ones(int(round(n*spec["P"]))), np.zeros(n-int(round(n*spec["P"])))]
    rng.shuffle(y)
    x0 = draw_markers(rng, n, data["rho"])
    e0 = rng.normal(size=n)*np.sqrt(1-h)
    if spec["cov"] == "binary":
        kp = norm.sf(threshold-gamma)
        prob_plus = np.where(y == 1, .5*kp/spec["K"], .5*(1-kp)/(1-spec["K"]))
        cov = np.where(rng.uniform(size=n) < prob_plus, 1., -1.)
        liability = truncated_liability(rng, y, threshold-gamma*cov)
        x = x0+(liability-x0@beta-e0)[:, None]*(R@beta)[None, :]
    else:
        c0 = rng.normal(size=n)
        liability = np.sqrt(1+gamma**2)*truncated_liability(rng, y, threshold/np.sqrt(1+gamma**2))
        innovation = (liability-x0@beta-gamma*c0-e0)/(1+gamma**2)
        x = x0+innovation[:, None]*(R@beta)[None, :]
        cov = c0+innovation*gamma
    if spec["cov"] == "dependent":
        x += .2*cov[:, None]*np.resize([-1., 1.], x.shape[1])[None, :]
    risk = prepare_binary_risk(y, spec["K"], population_risk=ndtr(gamma*cov-threshold), covariate_variance=gamma**2)
    return x, risk


def replicate(seed, scenario, args):
    data = generate(seed, scenario, args.samples, args.variants, args.reference_samples, 40)
    x, a, y, spec = data["x"], data["annotations"], data["y"], SCENARIOS[scenario]
    known = prepare_binary_risk(y, spec["K"], population_risk=data["k"], covariate_variance=data["Vc"])
    base = exact_moments(x, a, known)
    external = exact_external_ld(data["reference"], a)
    matched, matched_risk = matched_reference(data, spec, seed+6000000, args.reference_samples)
    weighted = exact_external_ld(matched*matched_risk.sensitivity[:, None], a)
    n = len(x)
    matched_moments = BinaryMoments(a, weighted*(n-1)/n, np.zeros_like(base.same_person), base.rhs_rows, n, "pcgc", known.covariate_variance)
    rho = np.full(args.variants, .85)
    rho[::40] = 0
    mismatch = draw_markers(np.random.default_rng(seed+7000000), args.reference_samples, rho)
    methods = {
        "study_known": base,
        "independent_matched_weighted": matched_moments,
        "population_factorized": external_ld_moments(base.rhs_rows, a, known, external),
        "mismatched_ld": external_ld_moments(base.rhs_rows, a, known, exact_external_ld(mismatch, a)),
        "estimated_population_scale": exact_moments((x-data["reference"].mean(axis=0))/data["reference"].std(axis=0), a, known),
    }
    rows = []
    for name, fit, K in (("study_probit", fit_binary_risk, spec["K"]),
                         ("study_logistic_backtransform", fit_binary_risk_logistic, spec["K"]),
                         ("prevalence_half", fit_binary_risk, spec["K"]/2),
                         ("prevalence_double", fit_binary_risk, spec["K"]*2),
                         ("omitted_covariate", fit_binary_risk, spec["K"])):
        try:
            risk = fit(y, K, None if name == "omitted_covariate" else data["cov"][:, None])
            methods[name] = exact_moments(x, a, risk)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            rows.append(dict(method=name, failure=str(exc)))
    for name, moments in methods.items():
        row = dict(method=name)
        try:
            row.update(fit_moments(moments))
            row["method"] = name
            row["normal_relative_difference"] = float(np.linalg.norm(moments.equations()[0]-base.equations()[0])/np.linalg.norm(base.equations()[0]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row["failure"] = str(exc)
        rows.append(row)
    for row in rows:
        row.update(seed=seed, scenario=scenario, marginal_truth=float(sum(data["truth"])/(1+data["Vc"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=("S2", "S3", "S4", "S6"), default=["S2", "S3", "S4", "S6"])
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--variants", type=int, default=4000)
    parser.add_argument("--reference-samples", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 19:
        parser.error("reuse only existing pilot replicate indices 1..19")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("validate_pcgc.py"), root/"src/summit/sumstats/binary.py",
               root/"src/summit/pcgc/research.py", root/"src/summit/pcgc/moments.py"]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    scope="paired sensitivity diagnostics, not independent qualification")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for scenario in args.scenarios:
        for index in range(1, args.replicates+1):
            start = time.perf_counter()
            rows.extend(replicate(820000+1000*int(scenario[1:])+index, scenario, args))
            print(json.dumps(dict(scenario=scenario, replicate=index, seconds=time.perf_counter()-start)), flush=True)
    (args.out/"replicates.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
