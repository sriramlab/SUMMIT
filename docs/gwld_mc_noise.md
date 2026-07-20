# Genome-wide LD-score Monte Carlo noise

SUMMIT reports a compact Monte Carlo (MC) noise diagnostic by default for the
randomized genome-wide LD-score estimator. Per-SNP MC variances are available
as an optional output because they require an additional `M x K` array.

For target SNP `i`, annotation `k`, and independent probe `v`, let

```text
Y_ikv = (X_i' W_kv / d)^2,
Lhat_ik = (1 / V) sum_v Y_ikv - M_k / d.
```

Here `d` is the residual normalization denominator and `M_k/d` is the
deterministic finite-sample null subtraction. Conditional on the decoded and
standardized reference genotype matrix, the sample variance of the probe mean
is

```text
Varhat_MC(Lhat_ik)
  = [sum_v Y_ikv^2 - (sum_v Y_ikv)^2 / V] / [V (V - 1)].
```

The null subtraction does not appear because subtracting a probe-independent
constant does not change MC variance.

## Default annotation-level diagnostic

For each annotation, SUMMIT reports the integrated pointwise MC variance

```text
S_k = sum_i Varhat_MC(Lhat_ik).
```

This is an estimate of the conditional expected squared Euclidean error in the
entire LD-score column. It is not `Var(sum_i Lhat_ik)`: that different quantity
contains cross-SNP probe-error covariances and does not summarize the typical
accuracy of the per-SNP scores. Cross-SNP and cross-annotation MC covariances
are deliberately not reported; overlapping or fractional annotations are
still valid for these marginal, per-annotation diagnostics.

The native phase-2 kernel accumulates only

```text
Q_k = sum_i sum_v Y_ikv^2
```

in addition to the existing `M x K` first sums. Thus

```text
S_k = [Q_k - sum_i (sum_v Y_ikv)^2 / V] / [V (V - 1)]
```

is computed with `O(K)` persistent additional storage. Python reduces the
existing `M x K` first-sum matrix in bounded row chunks, so it does not create
another full-size float64 square temporary. No probe-by-SNP array is created.
The output `<out>.gw.mc.tsv` contains one row per annotation, including:

- `annotation_mass`: `M_k = sum_j a_jk`;
- `nvecs`, `nsnps`, `seed`, `probe_distribution`, `dtype`, and
  `residual_correlation_denom`, which record the run defining the diagnostic;
- `integrated_mc_variance`: `S_k`;
- `rms_mc_se`: `sqrt(S_k / M)`, the root-mean-square per-SNP MC standard
  error;
- `ldscore_rms`: the root-mean-square reported LD score in the annotation;
- `relative_rms_mc_se`: `rms_mc_se / ldscore_rms`; and
- `numerical_status` plus aggregate and, when requested, per-SNP
  roundoff/failure counts for cancellation checks in the first/second-moment
  subtraction.

The last quantity is a rough scale diagnostic for choosing `--nvecs`. All
reported MC quantities are conditional on the realized reference panel and
measure random-probe error only. They do not include reference-panel sampling
error, genotype imputation uncertainty, or finite-sample LD bias.

With one probe, sample variance is undefined. SUMMIT still permits the LD-score
point estimate, emits a warning, and omits the default diagnostic. Optional
per-SNP variance/interval output requires at least two probes.

## Optional per-SNP output

`--write-ld-mc-var` (alias `--write-ld-mc-ci`) additionally stores
`sum_v Y_ikv^2` for every SNP and annotation and writes
`<out>.gw.mcvar.gz`. For every annotation it contains:

- `<annotation>_MC_VAR`;
- `<annotation>_MC_SE`;
- `<annotation>_MC_CI95_LO`; and
- `<annotation>_MC_CI95_HI`.

The interval is

```text
Lhat_ik +/- 1.9599639845 sqrt(Varhat_MC(Lhat_ik)).
```

It is an approximate pointwise, conditional Monte Carlo/CLT interval around
the reported LD score. It is not an exact finite-probe interval, a simultaneous
genome-wide interval, a population-LD interval, or a reference-panel sampling
interval. Lower bounds are not truncated. This option does not change the
LD-score point estimate or the annotation-level diagnostic.

The default diagnostic can be disabled with `--skip-ld-mc`. This is mainly an
escape hatch for a backend that does not expose the fourth-moment reduction.
At present, requesting CUDA with the default diagnostic causes a logged CPU
phase-2 fallback; use `--skip-ld-mc` to retain the CUDA phase-2 path.
The random-probe MC flags are rejected for deterministic windowed LD and for
the separate GxE estimator rather than being silently ignored.
