"""Complete-row empirical population experiments conditional on matched learning.

The public cohort preparation has already evaluated every frozen feature and
nuisance column on each donor. Resampling these complete rows is exactly the
same experiment as resampling all genotypes and covariates then evaluating the
same frozen functions. It does not break cross-chromosome dependence. Outcomes
are generated afresh, not resampled. Known means enter diagnostics only.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, ncx2, beta as beta_distribution, t as student_t

from scripts.epistasis.full_matched import load_reference, load_experiment_inputs, write_json
from summit.context.spec import canonical_sha256, array_sha256
from summit.epistasis.features import FeatureReference, write_feature_reference
from summit.epistasis.cli import main as cli
from summit.prediction._validation import digest


def binomial_interval(hits, count):
    if count == 0:
        return [None, None]
    return [float(beta_distribution.ppf(.025,hits,count-hits+1)) if hits else 0.,
            float(beta_distribution.ppf(.975,hits+1,count-hits)) if hits<count else 1.]


def generating_components(meta, data, setting, index, multiplier):
    """Decompose the fixed simulation mean; never used by the fitted test."""
    means = {name:data["means"][index,j] for j,name in enumerate(meta["settings"])}
    column = meta["settings"].index(setting)
    signal = multiplier * data["signals"][index,column]
    if setting == "finite":
        return {"finite_main":means[setting]}
    local = means["heavy"] - means["dense"]
    if setting == "sparse":
        components = dict(sparse_additive=means["sparse"]-local, local=local)
    else:
        components = dict(dense_additive=means["dense"])
        if setting != "dense":
            components["local"] = (means["local_withheld"]-means["dense"]
                if setting.startswith("local_withheld") else local)
        if setting in ("dominance", "structure", "structure_mixed"):
            components["dominance"] = means["dominance"] - means["heavy"]
        if setting in ("structure", "structure_mixed"):
            components["structure"] = means["structure"] - means["dominance"]
    components["interaction"] = signal
    expected = means[setting] + (multiplier-1)*data["signals"][index,column]
    np.testing.assert_allclose(sum(components.values()),expected,atol=1e-12,rtol=1e-12)
    return components


def aggregate_models(summaries):
    """Approximate Monte Carlo intervals with the trained model as the unit.

    Conditional draws are not independent learning replicates. Student intervals
    over per-model rates are approximate, especially with few models or zero
    observed between-model variation; those cases are explicitly unresolved.
    Exact conditional binomial intervals remain in each model's own record.
    """
    result = []
    for setting,method in sorted({(r["setting"],r["method"]) for r in summaries}):
        scheduled = [r for r in summaries if (r["setting"],r["method"])==(setting,method)]
        valid = [r for r in scheduled if not r["failed"]]
        record = dict(setting=setting,method=method,scheduled_learning_models=len(scheduled),
            unavailable_models=len(scheduled)-len(valid),
            failure_stages={stage:sum(r.get("failure_stage","parent_pipeline")==stage for r in scheduled if r["failed"])
                for stage in ("parent_pipeline","population_reference")},
            scheduled_conditional_draws=sum(r["scheduled_draws"] for r in scheduled),
            numerical_failures=sum(r["numerical_failures"] for r in valid),
            unsupported=sum(r["unsupported"] for r in valid))
        # Known population moments explain departures; they never calibrate
        # the reported test or enter its phenotype-dependent HC3 covariance.
        diagnostic = [r for r in valid if "known_population_covariance" in r]
        if diagnostic:
            noncentrality = [float(np.asarray(r["truth"]) @ np.linalg.solve(
                r["known_population_covariance"], r["truth"])) for r in diagnostic]
            record["mean_diagnostic_noncentrality"] = float(np.mean(noncentrality))
            covered = [r for r in diagnostic if "coverage" in r]
            record["models_with_coverage"] = len(covered)
            record["mean_projection_coverage"] = float(np.mean([r["coverage"] for r in covered])) if covered else None
            record["mean_rms_se_to_known_population_se"] = np.mean([
                np.asarray(r["rms_se"])/np.sqrt(np.diag(r["known_population_covariance"]))
                for r in covered],axis=0).tolist() if covered else None
            for label, alpha in [("05",.05),("005",.005)]:
                record[f"mean_diagnostic_asymptotic_rejection_{label}"] = float(np.mean([
                    ncx2.sf(chi2.ppf(1-alpha,len(r["truth"])),len(r["truth"]),value)
                    for r,value in zip(diagnostic,noncentrality)]))
        for label in ("05","005"):
            complete = [r for r in valid if r["scheduled_draws"]>0
                and r["denominator_all_numerical"]==r["scheduled_draws"]]
            rates = np.array([r[f"rejections_{label}_all_numerical"]/r["scheduled_draws"] for r in complete])
            record[f"complete_learning_models_{label}"] = len(rates)
            record[f"mean_rate_{label}"] = float(rates.mean()) if len(rates) else None
            informative = len(rates)>=10 and rates.std(ddof=1)>0
            record[f"model_mc_interval_{label}"] = (
                [float(max(0,rates.mean()-student_t.ppf(.975,len(rates)-1)*rates.std(ddof=1)/np.sqrt(len(rates)))),
                 float(min(1,rates.mean()+student_t.ppf(.975,len(rates)-1)*rates.std(ddof=1)/np.sqrt(len(rates))))]
                if informative else None)
            record[f"one_sided_model_mc_upper_{label}"] = (
                float(min(1,rates.mean()+student_t.ppf(.95,len(rates)-1)*rates.std(ddof=1)/np.sqrt(len(rates))))
                if informative else None)
        record["mc_method"] = "approximate Student interval over independent trained-model rates; no interval with <10 models or zero observed between-model variance"
        result.append(record)
    return result


class PopulationRegression:
    """Independent finite-regression reference, O(N k), never sample-square.

    A donor-population change of nuisance coordinates improves conditioning.
    Every sampled data set still refits all coefficients and its HC3 covariance.
    No generating mean, variance, or truth enters ``fit``.
    """
    def __init__(self, fixed, features):
        c, f = np.asarray(fixed, float), np.asarray(features, float)
        if c.shape[1] > 512 or f.shape[1] > 8:
            raise ValueError("bounded population regression needs <=512 nuisance and <=8 tested columns")
        norms = np.linalg.norm(c, axis=0)
        norms[norms == 0] = 1
        u, s, _ = np.linalg.svd(c / norms, full_matrices=False)
        u = u[:, s > s[0]*1e-11]
        r = f - u @ (u.T @ f)
        eigen, vectors = np.linalg.eigh(r.T @ r / len(c))
        if eigen[0] <= eigen[-1]*1e-10:
            raise ValueError("population interaction span is not identifiable")
        self.transform = (vectors / np.sqrt(eigen)) @ vectors.T
        self.design = np.column_stack([u*np.sqrt(len(c)), r @ self.transform])
        self.q, self.n = f.shape[1], len(c)

    def truth(self, mean, variance, sample_size):
        gamma = self.design.T @ mean / self.n
        residual = mean - self.design @ gamma
        influence = self.design[:, -self.q:] @ self.transform
        covariance = (influence.T * (variance + residual**2)) @ influence / self.n / sample_size
        return self.transform @ gamma[-self.q:], covariance

    def fit(self, donors, y):
        d = self.design[donors]
        gram = d.T @ d
        inverse = cho_solve(cho_factor(gram, lower=True), np.eye(d.shape[1]))
        coef = inverse @ (d.T @ y)
        residual = y - d @ coef
        hat = np.einsum("ij,ij->i", d, d @ inverse)
        if np.any(hat >= 1-1e-8):
            raise ValueError("sampled regression has unresolved unit leverage")
        influence = d @ inverse[:, -self.q:] @ self.transform
        adjusted = influence * (residual / (1-hat))[:, None]
        covariance = adjusted.T @ adjusted
        beta = self.transform @ coef[-self.q:]
        statistic = beta @ np.linalg.solve(covariance, beta)
        # Support is re-evaluated from each sampled design, not inherited from
        # the donor pool. Effective support follows the original feature axes.
        r = d[:, -self.q:] - d[:, :-self.q] @ cho_solve(
            cho_factor(gram[:-self.q, :-self.q], lower=True), gram[:-self.q, -self.q:])
        r = r @ np.linalg.inv(self.transform)
        h = r.T @ r
        normalized = r / np.sqrt(np.diag(h))
        corr = normalized.T @ normalized
        effective = float(np.min(1/np.sum(normalized**4, axis=0)))
        outside = []
        for bad, reason in [(len(y)<1000, "N below 1000"),
            (d.shape[1]/len(y)>.05, "fitted rank exceeds 5% of N"),
            (hat.max()>.1, "maximum leverage exceeds .1"),
            (effective<100, "feature effective support below 100"),
            (np.linalg.cond(corr)>1e6, "normalized information condition exceeds 1e6")]:
            if bad:
                outside.append(reason)
        return dict(beta=beta, coefficient_covariance=covariance,
            p=float(chi2.sf(statistic, self.q)), max_leverage=float(hat.max()),
            outside_scope=outside, minimum_feature_effective_support=effective)


def public_draw(parent, donors, y, output, *, threads, memory_gib):
    """Publish one independently identified donor cohort and run public reuse.

    This is simulation-side cohort construction, not a production bootstrap
    correction. The parent reference remains available to authenticate every
    frozen feature definition and its training separation.
    """
    output.mkdir(parents=True, exist_ok=False)
    ref = load_reference(parent)
    identity = canonical_sha256(dict(parent=ref.metadata["compatibility_id"],
        donors=array_sha256(donors), rule="complete IID donor rows; fresh errors"))
    samples = [(identity[:16], f"draw{i}") for i in range(len(donors))]
    pd.DataFrame(samples, columns=["FID", "IID"]).to_csv(output / "samples.tsv", sep="\t", index=False)
    table = pd.DataFrame(samples, columns=["FID", "IID"])
    table["y"] = y
    table.to_csv(output / "phenotype.tsv", sep="\t", index=False)
    meta = dict(component_names=list(ref.metadata["component_names"]),
        feature_names=list(ref.metadata["feature_names"]), n_samples=len(donors),
        compatibility_id=identity, sample_hash=digest(samples),
        cohort_sample_tokens=[canonical_sha256(list(s)) for s in samples], seed=1,
        job=dict(inference=dict(method="robust_mean", sampling_model="iid_population_projection")),
        definitions=dict(empirical_population=dict(parent_reference=str(parent.resolve()),
            parent_identity=ref.metadata["compatibility_id"], donor_count=len(ref.features),
            donor_indices_hash=array_sha256(donors),
            rule="complete IID donor rows; original feature scale; independent newly generated outcomes")))
    sampled = FeatureReference(ref.features[donors], ref.fixed_effects[donors], ref.component_index, meta)
    write_feature_reference(sampled, output / "cohort-reference.npz")
    np.save(output / "donors.npy", donors, allow_pickle=False)
    write_json(output / "reuse.json", dict(kind="summit.epistasis.prepare_traits", schema_version=1,
        reference="cohort-reference.npz", samples="samples.tsv",
        phenotypes=dict(file="phenotype.tsv", columns=["y"], unit="fixed reference units")))
    cli(["prepare-traits", str(output / "reuse.json"), "--out", str(output / "summary.npz"),
        "--num-threads", str(threads), "--memory-gib", str(memory_gib)])
    cli(["fit", str(output / "summary.npz"), "--out", str(output / "fit.json")])
    return json.loads((output / "fit.json").read_text())["fits"][0]


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    design = json.loads((a.input / "design.json").read_text())
    if (design.get("kind") == "matched_learning_donor_preparation"
            and a.samples != design["intended_confirmation_n"]):
        raise ValueError("expanded donor validation must retain the selected confirmation sample size")
    args = design["arguments"]
    reference = Path(design["reference"])
    meta, data, samples = load_experiment_inputs(reference,
        local_stress="local_withheld" in args["settings"], threads=a.num_threads)
    selected = pd.read_csv(a.input / "confirmation.tsv", sep="\t", dtype=str)
    index = pd.MultiIndex.from_frame(samples).get_indexer(pd.MultiIndex.from_frame(selected))
    if np.any(index < 0):
        raise ValueError("donor sample alignment failed")
    records = [json.loads(line) for line in (a.input / "replicates.jsonl").read_text().splitlines()]
    methods = a.methods.split(",")
    settings = a.settings.split(",") if a.settings else args["settings"].split(",")
    if (len(set(methods)) != len(methods) or len(set(settings)) != len(settings)
            or set(settings)-set(args["settings"].split(","))):
        raise ValueError("select distinct scheduled methods and settings")
    selected_records = [r for r in records if r["method"] in methods and r["setting"] in settings]
    expected = {(setting,rep,method) for setting in settings for rep in range(args["replicates"]) for method in methods}
    observed = {(r["setting"],r["replicate"],r["method"]) for r in selected_records}
    if observed != expected or len(selected_records) != len(expected):
        raise ValueError("every scheduled learning fit in the selected settings must finish before population assessment")
    write_json(a.out / "design.json", dict(parent=str(a.input.resolve()), phase=a.phase,
        seed=a.seed, conditional_draws_per_model=a.draws, scheduled_models=len(selected_records),
        evidence="known-moment diagnosis only; no observed calibration" if a.draws==0 else "complete-row outcome experiment",
        donors=len(index), confirmation_n=a.samples, methods=methods,
        population="uniform empirical distribution of complete held-out participant rows",
        training="original matched full-pipeline independent outcome learning; condition on each resulting model",
        truth="exact projection over the entire empirical donor distribution, separately for each frozen model",
        thresholds=[.05,.005], material_inflation_tolerances=[.075,.01]))
    summaries = []
    for record in selected_records:
        setting, rep, method = record["setting"], record["replicate"], record["method"]
        label = f"{setting}_{rep:03d}_{method}"
        if record["failed"]:
            summaries.append(dict(setting=setting, replicate=rep, method=method,
                failed=True, failure_stage="parent_pipeline", reason=record["error"], scheduled_draws=a.draws))
            continue
        if "batch_size" in args:
            batch = rep // args["batch_size"] * args["batch_size"]
            parent = a.input / f"{setting}_{batch:03d}/prepared/rep{rep:03d}_{method}.cohort-reference.npz"
        else:
            parent = a.input / f"{setting}_{rep:03d}/prepared/{method}.cohort-reference.npz"
        ref = load_reference(parent)
        if len(ref.features) != len(index):
            raise ValueError("reference and donor pool disagree")
        column = meta["settings"].index(setting)
        multiplier = args.get("signal_multiplier", 1.)
        mean = data["means"][index, column] + (multiplier-1)*data["signals"][index, column]
        variance = data["variance"][index]
        try:
            regression = PopulationRegression(ref.fixed_effects, ref.features)
            truth, known_cov = regression.truth(mean, variance, a.samples)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            failed = dict(setting=setting,replicate=rep,method=method,failed=True,
                failure_stage="population_reference",reason=str(error),scheduled_draws=a.draws)
            summaries.append(failed)
            write_json(a.out / f"{label}.json",dict(summary=failed,draws=[]))
            continue
        signal = multiplier*data["signals"][index, column]
        leakage, _ = regression.truth(mean-signal, variance, a.samples)
        components = generating_components(meta, data, setting, index, multiplier)
        # Projection is linear in the mean; one matrix product gives each
        # component's contribution in the original interaction coordinates.
        coefficients = regression.transform @ (regression.design[:,-regression.q:].T
            @ np.column_stack(list(components.values()))) / regression.n
        np.testing.assert_allclose(coefficients.sum(1), truth, atol=1e-10, rtol=1e-8)
        draws = []
        for draw in range(a.draws):
            # Common donor/error draws across methods allow paired comparisons.
            rng = np.random.default_rng(np.random.SeedSequence([a.seed, column, rep, draw]))
            donors = rng.integers(0, len(index), a.samples)
            error = rng.standard_t(5, a.samples)/np.sqrt(5/3) if setting == "heavy" else rng.normal(size=a.samples)
            y = mean[donors] + np.sqrt(variance[donors])*error
            try:
                fit = regression.fit(donors, y)
                if draw == 0:
                    public = public_draw(parent, donors, y, a.out / label,
                        threads=a.num_threads, memory_gib=a.memory_gib)
                    np.testing.assert_allclose(fit["beta"], public["beta"], rtol=1e-7, atol=1e-9)
                    np.testing.assert_allclose(fit["coefficient_covariance"], public["coefficient_covariance"], rtol=1e-7, atol=1e-10)
                    np.testing.assert_allclose(fit["p"], public["joint_p"], rtol=1e-7, atol=1e-10)
                    assert fit["outside_scope"] == public["diagnostics"]["outside_confirmation_design"]
                delta = fit["beta"]-truth
                draws.append(dict(setting=setting, replicate=rep, method=method, draw=draw,
                    failed=False, **fit, error=delta,
                    covered=bool(delta @ np.linalg.solve(fit["coefficient_covariance"], delta) <= chi2.ppf(.95, len(truth)))))
            except (ValueError, ArithmeticError, AssertionError, np.linalg.LinAlgError) as error:
                draws.append(dict(setting=setting, replicate=rep, method=method, draw=draw,
                    failed=True, reason=str(error)))
        valid = [r for r in draws if not r["failed"]]
        supported = [r for r in valid if not r["outside_scope"]]
        summary = dict(setting=setting, replicate=rep, method=method, failed=False,
            biological_null=meta["definitions"][setting]["biological_null"],
            scheduled_draws=a.draws, numerical_failures=a.draws-len(valid), unsupported=len(valid)-len(supported),
            truth=truth, additive_leakage=leakage, known_population_covariance=known_cov,
            diagnostic_generating_contributions={name:coefficients[:,j] for j,name in enumerate(components)})
        for suffix, values in [("all_numerical", valid), ("supported", supported)]:
            summary[f"denominator_{suffix}"] = len(values)
            for threshold, alpha in [("05",.05),("005",.005)]:
                hits = sum(r["p"]<alpha for r in values)
                summary[f"rejections_{threshold}_{suffix}"] = hits
                summary[f"conditional_mc95_{threshold}_{suffix}"] = binomial_interval(hits,len(values))
                summary[f"conditional_one_sided_mc_upper_{threshold}_{suffix}"] = (
                    None if not values else float(beta_distribution.ppf(.95,hits+1,len(values)-hits)) if hits<len(values) else 1.)
        if valid:
            summary.update(error_mean=np.mean([r["error"] for r in valid], axis=0),
                error_sd=np.std([r["error"] for r in valid], axis=0, ddof=1) if len(valid)>1 else None,
                mean_se=np.mean([np.sqrt(np.diag(r["coefficient_covariance"])) for r in valid], axis=0),
                rms_se=np.sqrt(np.mean([np.diag(r["coefficient_covariance"]) for r in valid], axis=0)),
                error_covariance=np.atleast_2d(np.cov(np.array([r["error"] for r in valid]).T)) if len(valid)>1 else None,
                coverage=float(np.mean([r["covered"] for r in valid])),
                coverage_conditional_mc95=binomial_interval(sum(r["covered"] for r in valid),len(valid)))
        summaries.append(summary)
        write_json(a.out / f"{label}.json", dict(summary=summary, draws=draws))
        print(label, len(valid), summary["rejections_05_all_numerical"], "elapsed", round(time.perf_counter()-start), flush=True)
    write_json(a.out / "results.json", summaries)
    write_json(a.out / "aggregate.json", aggregate_models(summaries))
    write_json(a.out / "resources.json", dict(seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        genotype_reads=0, independent_learning_models=len({(r["setting"],r["replicate"]) for r in selected_records})))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=61440)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--moments-only", action="store_true", help="known-mean population diagnosis without outcome draws; not a calibration estimate")
    p.add_argument("--methods", default="joint")
    p.add_argument("--settings", default="")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--phase", choices=["development", "confirmation"], default="development")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--memory-gib", type=float, default=16)
    a = p.parse_args()
    if a.moments_only:
        a.draws=0
    if (a.draws < 2 and not a.moments_only) or a.samples < 1:
        p.error("at least two conditional draws and a positive sample size required")
    run(a)


if __name__ == "__main__":
    main()
