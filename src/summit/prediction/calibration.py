"""Optional official CalPred interval adapter and paired evaluation uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.stats import norm

from ._validation import array, closed, metadata, positive_int
from .artifacts import file_digest


@dataclass(frozen=True)
class CalPredIntervals:
    mean_coefficients: np.ndarray
    log_variance_coefficients: np.ndarray
    mean_names: tuple[str, ...]
    variance_names: tuple[str, ...]
    provenance: dict

    def __post_init__(self):
        mean = array(self.mean_coefficients, name="CalPred mean coefficients", ndim=1)
        variance = array(self.log_variance_coefficients, name="CalPred variance coefficients", ndim=1)
        for names, coefficient in ((self.mean_names, mean), (self.variance_names, variance)):
            if len(names) != len(coefficient) or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
                raise ValueError("invalid CalPred feature names/coefficient dimensions")
        object.__setattr__(self, "mean_coefficients", mean)
        object.__setattr__(self, "log_variance_coefficients", variance)
        object.__setattr__(self, "mean_names", tuple(self.mean_names))
        object.__setattr__(self, "variance_names", tuple(self.variance_names))
        object.__setattr__(self, "provenance", metadata(self.provenance))

    def to_dict(self):
        return dict(kind="summit.prediction.calpred_intervals", schema_version=1,
            mean_coefficients=self.mean_coefficients.tolist(), log_variance_coefficients=self.log_variance_coefficients.tolist(),
            mean_names=list(self.mean_names), variance_names=list(self.variance_names), provenance=self.provenance)

    def predict(self, mean_features, variance_features, *, mean_names, variance_names, coverage=.9):
        x = np.asarray(mean_features, dtype=float)
        z = np.asarray(variance_features, dtype=float)
        if tuple(mean_names) != self.mean_names or tuple(variance_names) != self.variance_names:
            raise ValueError("CalPred feature identities differ from the fitted model")
        if x.ndim != 2 or z.ndim != 2 or x.shape != (len(z), len(self.mean_names)) or z.shape[1] != len(self.variance_names):
            raise ValueError("CalPred feature dimensions disagree")
        if not 0 < coverage < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
            raise ValueError("invalid CalPred features or coverage")
        mean = x @ self.mean_coefficients
        with np.errstate(over="raise", invalid="raise"):
            sd = np.exp(.5*(z @ self.log_variance_coefficients))
        if not np.all(np.isfinite(sd)) or np.any(sd <= 0) or not np.all(np.isfinite(mean)):
            raise FloatingPointError("invalid predicted phenotype variance")
        radius = norm.ppf((1+coverage)/2)*sd
        return mean, mean-radius, mean+radius


def fit_calpred(y, mean_features, variance_features, *, mean_names, variance_names,
                official_script, expected_sha256, scratch_root, provenance,
                rscript="Rscript", timeout_seconds=600):
    """Invoke a caller-installed, checksum-pinned official Gaussian CalPred script.

    Does not install software, reduce features, transform phenotypes or fit SNP
    weights. Caller supplies matched full-rank mean/variance designs and an
    approved temporary root. Warnings and script identity are retained.
    """
    y = array(y, name="CalPred phenotype", ndim=1)
    x = array(mean_features, name="CalPred mean design", ndim=2)
    z = array(variance_features, name="CalPred variance design", ndim=2)
    if x.shape != (len(y), len(mean_names)) or z.shape != (len(y), len(variance_names)) or not provenance:
        raise ValueError("invalid CalPred axes/provenance")
    if len(set(mean_names)) != len(mean_names) or len(set(variance_names)) != len(variance_names):
        raise ValueError("CalPred feature names must be unique")
    if not x.shape[1] or not z.shape[1] or np.linalg.matrix_rank(x) != x.shape[1] or np.linalg.matrix_rank(z) != z.shape[1] or len(y) <= max(x.shape[1], z.shape[1]):
        raise ValueError("CalPred needs explicit full-rank designs with residual degrees of freedom")
    script = Path(official_script).resolve()
    if file_digest(script) != expected_sha256:
        raise ValueError("official CalPred script checksum mismatch")
    positive_int(timeout_seconds, "timeout_seconds")
    with tempfile.TemporaryDirectory(prefix="summit-calpred-", dir=scratch_root) as tmp:
        root = Path(tmp)
        for name, values in (("y", y), ("x", x), ("z", z)):
            np.savetxt(root/f"{name}.txt", values)
        command = [rscript, "--vanilla", str(script), str(root/"y.txt"), str(root/"x.txt"), str(root/"z.txt"), str(root/"fit")]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        if result.returncode:
            raise RuntimeError(f"official CalPred exited with status {result.returncode}: {result.stderr[-2000:]}")
        mean = np.loadtxt(root/"fit.mean", ndmin=2)[:, 0]
        variance = np.loadtxt(root/"fit.sd", ndmin=2)[:, 0]
    if mean.shape != (x.shape[1],) or variance.shape != (z.shape[1],) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("invalid coefficients from official CalPred")
    return CalPredIntervals(mean, variance, tuple(mean_names), tuple(variance_names),
        dict(input=provenance, script_sha256=expected_sha256,
             diagnostics=(result.stdout+result.stderr).strip(), target="phenotype_prediction_interval"))


def load_calpred_intervals(path):
    from .artifacts import read_json
    value = read_json(path)
    closed(value, ("kind", "schema_version", "mean_coefficients", "log_variance_coefficients", "mean_names", "variance_names", "provenance"), name="CalPred intervals")
    if value["kind"] != "summit.prediction.calpred_intervals" or value["schema_version"] != 1:
        raise ValueError("unsupported CalPred interval artifact")
    return CalPredIntervals(**{k: v for k, v in value.items() if k not in ("kind", "schema_version")})


def paired_r2_gain(y, baseline, candidate, *, bootstrap=1000, seed=0):
    y = array(y, name="evaluation phenotype", ndim=1)
    baseline, candidate = array(baseline, name="baseline", ndim=1), array(candidate, name="candidate", ndim=1)
    if baseline.shape != y.shape or candidate.shape != y.shape or len(y) < 3 or np.var(y) == 0:
        raise ValueError("invalid paired evaluation arrays")
    positive_int(bootstrap, "bootstrap")
    gain = (y-baseline)**2 - (y-candidate)**2
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap):
        rows = rng.integers(0, len(y), len(y))
        variance = np.var(y[rows])
        if variance > 0:
            values.append(float(np.mean(gain[rows])/variance))
    if len(values) < max(2, bootstrap//2):
        raise ValueError("too few nondegenerate paired bootstrap samples")
    return dict(delta_r2=float(np.mean(gain)/np.var(y)), interval=np.quantile(values, [.025, .975]).tolist(),
        bootstrap=len(values), seed=seed, uncertainty="conditional_on_frozen_discovery_and_pilot_fits")
