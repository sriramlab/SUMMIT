"""Run the unmodified pinned FAME binary and reconcile its reported equations.

Synthetic inputs are temporary. Persist the exact command, revision, printed
normal equations, estimates/SEs and error measures, not individual-level data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np
from bed_reader import to_bed

from summit.epistasis.oracle import dense_summary, selected_kernels
from summit.epistasis.summary import fit_epistasis
from summit.epistasis.cli import _jsonable

REVISION = "2551e7fbeb19aed3383104e03d77cc75defffc0e"


def printed_equations(stdout, c):
    after = stdout.split("Xl", 1)[1]
    left, right = after.split("Yl", 1)
    t = np.fromstring(left, sep=" ").reshape(c, c)
    q = np.fromstring(right.split("sigms", 1)[0], sep=" ")
    return t, q


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rev = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION or subprocess.check_output(["git", "-C", str(args.source), "diff", "--name-only"], text=True).strip():
        raise ValueError("comparison requires a clean checkout at the pinned FAME revision")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(793)
    n, m, target = 128, 24, 2
    raw = np.column_stack([rng.permutation(np.repeat([0., 1., 2.], [32, 64, 32])) for _ in range(m)])
    x = (raw - 1) / np.sqrt(.5)
    cov = np.column_stack([np.ones(n), rng.normal(size=n)])
    phenotype = x @ rng.normal(size=m) / np.sqrt(m) + .3*x[:, target]*x[:, 10] + rng.normal(size=n)
    records = []
    with tempfile.TemporaryDirectory(prefix="summit-fame-comparison-") as directory:
        root = Path(directory)
        to_bed(root/"input.bed", raw, properties={"sid": [f"v{i}" for i in range(m)]})
        for mode in ("no_covariates", "phenotype_covariates", "pipeline_local_then_covariates"):
            bins = np.ones((m, 1))
            background = np.ones(m)
            y = phenotype.copy()
            flags = []
            if mode == "pipeline_local_then_covariates":
                bins = np.column_stack([np.arange(m) < 4, np.arange(m) >= 4]).astype(float)
                background = bins[:, 1].copy()
                local = np.column_stack([np.ones(n), raw[:, :4]])
                y -= local @ np.linalg.lstsq(local, y, rcond=None)[0]
            if mode != "no_covariates":
                with (root/"input.cov").open("w") as handle:
                    handle.write("FID IID intercept c\n")
                    for i in range(n):
                        handle.write(f"{i} {i} {cov[i,0]:.17g} {cov[i,1]:.17g}\n")
                flags = ["-c", str(root/"input.cov")]
            # Input phenotype to executable; its own covariate handling follows.
            with (root/"input.pheno").open("w") as handle:
                handle.write("FID IID pheno\n")
                for i, value in enumerate(y):
                    handle.write(f"{i} {i} {value:.17g}\n")
            if flags:
                y -= cov @ np.linalg.lstsq(cov, y, rcond=None)[0]
                y = (y-y.mean()) / np.std(y, ddof=1)
            else:
                y -= y.mean()
            background[target] = 0
            weights = np.column_stack([bins, background])
            modifiers = np.column_stack([np.ones((n, bins.shape[1])), x[:, target]])
            # The executable default residualizes only y and uses I for noise.
            kernels, p = selected_kernels(x, modifiers, weights, np.empty((n, 0)))
            names = tuple(f"additive_{i}" for i in range(bins.shape[1])) + ("epistasis", "residual")
            summary = dense_summary(kernels, y, component_names=names, trait_names=("y",),
                                    residual_rank=n, metadata={"comparator": REVISION})
            exact = fit_epistasis(summary)
            np.savetxt(root/"input.annot", bins, fmt="%d")
            cmd = [str(args.binary.resolve()), "-g", str(root/"input"), "-p", str(root/"input.pheno"),
                   *flags, "-gxgbin", str(bins.shape[1]-1), "-snp", str(target+1),
                   "-k", "8192", "-jn", "4", "-annot", str(root/"input.annot"),
                   "-o", str(root/"result.txt")]
            started = time.perf_counter()
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            elapsed = time.perf_counter()-started
            t, q = printed_equations(run.stdout, len(names))
            rows = re.findall(r"sigma\^2_\d+: ([^ ]+) se: ([^\n]+)", (root/"result.txt").read_text())
            observed = np.array([[float(a), float(b)] for a, b in rows])
            fitted = np.linalg.solve(t, summary.rhs[:, 0])
            cq = 2*np.einsum("c,acb->ab", fitted, summary.cubic[0])
            inverse = np.linalg.inv(t)
            se = np.sqrt(np.diag(inverse @ cq @ inverse.T))
            records.append(dict(mode=mode, elapsed_seconds=elapsed,
                expected_equation_agreement=(mode == "pipeline_local_then_covariates"),
                interpretation=("target outside background: equations agree" if mode == "pipeline_local_then_covariates"
                    else "known upstream self-removal discrepancy: one-based local index used as matrix index; product zeroed before centering"),
                command=[s.replace(str(root), "<temporary-input>") for s in cmd],
                printed_matrix=t.tolist(), printed_rhs=q.tolist(),
                upstream_coefficients=observed[:, 0].tolist(), upstream_se=observed[:, 1].tolist(),
                exact_coefficients=exact["coefficients"].tolist(), exact_se=exact["standard_errors"].tolist(),
                matched_trace_coefficients=fitted.tolist(), matched_trace_se=se.tolist(),
                max_rhs_error=float(np.max(abs(q-summary.rhs[:, 0]))),
                max_matched_coefficient_error=float(np.max(abs(fitted-observed[:, 0]))),
                max_matched_se_error=float(np.max(abs(se-observed[:, 1]))),
                trace_relative_frobenius_error=float(np.linalg.norm(t-summary.matrix)/np.linalg.norm(summary.matrix))))
    payload = dict(fame_revision=rev, binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                   seed=793, n=n, m=m, target_zero_based=target, records=records,
                   caveat="Upstream seeds sample probes from wall clock; printed matrices have six significant digits.")
    (args.out/"comparison.json").write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False)+"\n")
    print(json.dumps(_jsonable(payload), indent=2, allow_nan=False))
    if any(r["expected_equation_agreement"] and (r["max_matched_coefficient_error"] > 5e-5
            or r["max_matched_se_error"] > 5e-5) for r in records):
        raise RuntimeError("FAME comparison exceeds tolerance for six-digit printed equations")


if __name__ == "__main__":
    main()
