# Benchmarks

The measurements below describe specific tested workloads. Runtime depends on
panel size, annotations, context dimension, storage, and CPU configuration.

## PGS synthetic benchmark

The PGS benchmark uses 2,048 generated samples and 4,096 variants, two trait
masks, ten models per trait, five context columns, and one CPU thread. Each
process performs setup, one warmup, and three operator repetitions.

| Genotype storage | Setup, seconds | Median operator, seconds | Process peak RSS, MiB |
|---|---:|---:|---:|
| Stream | 0.1738 | 0.3730 | 222.4 |
| Compact | 0.1780 | 0.3296 | 224.1 |
| Standardized FP64 | 0.1708 | 0.1788 | 261.1 |

These Linux measurements used pthread BLIS, SNP blocks of 256, and RHS width 64.
Peak RSS includes fixture generation. The timing covers a covariance operation,
not a complete fit. Full-cohort PGS memory and throughput have not been measured.

Run each mode in a fresh process:

```bash
python scripts/prediction/checkout.py benchmark --mode stream --threads 1
python scripts/prediction/checkout.py benchmark --mode compact --threads 1
python scripts/prediction/checkout.py benchmark --mode standardized --threads 1
```

Cached modes read the source once during setup. Streaming reads it on every
operator call. Repeated outputs were identical in the recorded runs.

## Numerical checks

The initial PGS tests cover dense heteroskedastic GLS comparisons, singular and
zero priors, trait-specific sample/SNP subsets, allele swaps, fractional dosages,
artifact reload, invalid inputs, and convergence failure. A synthetic six-model
fit reached a maximum relative true residual of `7.64e-11` with `rtol=1e-10` and
scored 24 held-out samples in one traversal.

The combined prediction and relevant context/genotype regression run passed
107 tests, with six other CLI/end-to-end tests excluded from that selection.
The prediction-only suite passed 24 tests. Reproduction commands are in
[Development](Development.md).

## Generalized G×E simulation results

The recorded 100-replicate simulation used approximately 10,000 samples,
454,207 variants, three context columns, and the complete six-component residual
context basis. Phenotypes were simulated on a fixed genotype panel.

At 1,024 random vectors, the three null off-diagonal components had empirical
standard deviations 0.0415, 0.0438, and 0.0471. Mean reported jackknife standard
errors were 0.0379, 0.0378, and 0.0451, with rejection counts 8, 9, and 4 out of
100 at nominal 5%. These figures show some standard-error underestimation;
100 replicates also leave substantial Monte Carlo uncertainty.

At 128 vectors, two components showed appreciable point-estimate shifts and
rejection counts of 13 and 16 out of 100. Probe-count sensitivity should be
checked for the model being fitted.

In a nonzero cross-response simulation, the mean generating coefficient was
0.18051. At 1,024 vectors its mean estimate was 0.17491, with 94 of 100 replicates
rejecting zero. These are results for that simulation, not general power claims.

## Batch h² timing

A recorded four-trait comparison used 7,774,235 SNPs, 59 annotation columns,
chromosome jackknife, and four threads:

| Mode | Total wall time, seconds | Peak RSS, GB |
|---|---:|---:|
| Regular | 689.8 | 31.92 |
| Fast, uncached text | 724.8 | 32.03 |
| Fast, cached binary | 532.4 | 31.93 |

The 480 serialized component values differed by at most `2.7e-12`. Cached trait
processing took 6.43 seconds for four traits, but the common model load dominated
the end-to-end run. Uncached batching was not faster in this comparison.

## Measuring a new workload

Measure complete wall time through result writing and reload, peak RSS, and
input traversals. Compare estimates at the same solver tolerance or random-vector
count. For PGS, include held-out scoring; for randomized references, check more
than one seed and vector count before choosing a production setting.
