# Fast batched h2

## Scope

`--h2-batch-fast` is an opt-in execution path for large collections of
univariate h2 analyses that share one LD-score/annotation model. It does not
change the default h2, rg, manifest-rg, or LD-score-generation paths.

Fast mode currently requires chromosome jackknife (`--njack chr[:...]`). A
contiguous block jackknife is deliberately rejected because the legacy mode
rebuilds block boundaries after trait-specific SNP filtering. Reusing fixed
pre-drop blocks would not have identical jackknife semantics.

## Exact calculation

For each trait, the streamed reader applies the same intrinsic QC, effective-N
scale, and chi-square filter as `Sumstats`. It then evaluates the existing exact
score moment

```
z*_j = sqrt(N - 1) * beta_j / sqrt(beta_j^2 + (n_j - 2) * se_j^2)
y_j  = z*_j^2
```

on the immutable Trace SNP axis. Excluded or absent SNPs have `y_j = 0` and a
separate Boolean active mask.

For a batch of traits, each chromosome-jackknife unit is evaluated with one
float64 matrix multiplication:

```
Ay[u, :, traits] = A[u, :].T @ Y[u, traits]
```

All trait-independent structural statistics are calculated once. For every
trait, full-axis structural statistics are reduced by explicit contributions
from its dropped SNPs. This is exact for the fixed chromosome units. The
resulting sufficient statistics enter the existing normal-equation builder and
`fit_h2`; no new estimator or approximate solver is used.

If the streamed reader encounters duplicate SNP IDs, it falls back to the
legacy `Sumstats` reader, preserving its max-N/first-row duplicate resolution.
Verbose chi-square diagnostics also use the legacy reader.

## Reusable cache

The optional cache stores, per trait:

- exact float64 `y` on the full Trace axis;
- a bit-packed active mask;
- N/covariate-rank and matched-SNP metadata;
- source-file size and nanosecond-mtime fingerprints;
- a deterministic SHA-256 digest of the ordered Trace SNP axis;
- SHA-256 checksums for both cached arrays.

Entries are written to temporary files, fsynced, atomically renamed, and
published by writing metadata last. A POSIX file lock prevents concurrent jobs
from constructing the same entry. Array checksums can be revalidated with
`--h2-cache-verify-checksum`.

The cache is independent of annotation values and LD-score columns, but it is
not independent of the Trace SNP axis. Therefore, it can be built with a
single-component model and reused for baseline+cell-type models only when their
ordered SNP-axis digest matches exactly.

At 7.8 million SNPs, one entry is approximately 63 MB: about 62 MB for float64
`y` and 1 MB for the packed mask. The full 2,940-protein plus 349-trait cache
(3,289 entries) is therefore approximately 208 GB before filesystem metadata.

## Commands

Build and checksum-validate cache entries once, preferably with the
single-component model:

```bash
summit \
  --h2 /path/to/targets/chr@ \
  --ldscores /path/to/single/chr@.ldscore.gz \
  --out /scratch/cache_build \
  --njack chr \
  --max-chisq auto \
  --chisq-action drop \
  --h2-batch-fast \
  --h2-cache-dir /scratch/h2_trait_cache \
  --h2-cache-mode readwrite \
  --h2-cache-only \
  --h2-cache-verify-checksum \
  --h2-workers 4 \
  --h2-batch-size 4 \
  --num-threads 4
```

Run an annotation model from validated cache entries:

```bash
summit \
  --h2 /path/to/targets/chr@ \
  --ldscores /path/to/model/chr@.ldscore.gz \
  --annot /path/to/model/chr@.annot.gz \
  --out /scratch/results/model \
  --njack chr \
  --max-chisq auto \
  --chisq-action drop \
  --h2-batch-fast \
  --h2-cache-dir /scratch/h2_trait_cache \
  --h2-cache-mode read \
  --h2-workers 4 \
  --h2-batch-size 8 \
  --num-threads 4
```

`--h2-workers` controls concurrent text/cache loaders. `--num-threads` remains
the BLAS/OpenMP cap. `--h2-batch-size` bounds the full-axis `Y` and active-mask
buffers; each additional in-memory trait costs approximately 70 MB at the
current imputed SNP count, excluding transient parser memory.

For repeated annotation models, build the cache once in independent trait
shards with a lightweight one-component Trace, then run one job per annotation
model over the complete trait collection. The per-entry file lock makes
overlapping retries safe. On storage where compressed text parsing does not
scale with threads, use one loader per cache-build shard and obtain throughput
from independent scheduler tasks. Cache-reading model jobs can use a bounded
batch of four to eight traits and four BLAS threads.

Do not use uncached fast mode as a general speed assumption. Its matrix work is
batched, but compressed-text parsing can remain the bottleneck. The reusable
binary cache is the intended optimization when the same traits are fitted
against many tissue or cell-type models.

## Imputed-panel benchmark (2026-07-11)

The validation used four UKB-PPP proteins, 7,774,235 SNPs, 59 baseline bins,
chromosome delete-1 jackknife, and four threads. All 480 serialized h2/component
estimates agreed with the legacy path (`max_abs_diff = 2.7e-12`).

| Run | Wall time | Peak RSS | Four-trait batch |
| --- | ---: | ---: | ---: |
| Legacy | 689.8 s | 31.92 GB | approximately 220 s after fixed model work |
| Fast, uncached text | 724.8 s | 32.03 GB | 254.5 s |
| Fast, cached binary | 532.4 s | 31.93 GB | 6.43 s |

The four-trait end-to-end cache run was 1.30x faster despite the shared model
load dominating the test. The reusable trait stage was over 30x faster than
legacy in this run. Extrapolating the observed 6.43 seconds per four traits to
3,268 currently available protein/continuous-trait targets gives about 1.5
hours of trait work plus one model load. The eventual 3,289-entry target is
effectively the same scale. These are throughput estimates rather than
scheduler/runtime guarantees.

The cache was built against a one-component 1 Mb windowed Trace and read under
the 59-bin stochastic baseline model. Both produced the same ordered-axis
SHA-256 digest, directly validating cross-model reuse. The four entries occupied
242 MiB and their persisted array checksums were independently revalidated.

An isolated Hoffman smoke (`14014217`) then ran legacy, uncached-fast, cache
construction, and checksum-verified cache-read h2 through the production SUMMIT
launcher. It exited successfully in 20.6 seconds with 1.06 GB maximum virtual
memory; both fast outputs were identical to legacy at serialized precision.

## Validation requirements

Before production use on a new panel:

1. Compare all h2/component estimates and SEs against legacy mode on multiple
   traits, including traits with dropped and missing SNPs.
2. Run cache construction once with checksum verification.
3. Audit persisted entries with `--h2-cache-only --h2-cache-mode read` and
   `--h2-cache-verify-checksum`, or perform an equivalent checksum pass.
4. Rerun from `--h2-cache-mode read` and compare against the uncached fast run.
5. Benchmark peak RSS at the intended worker count.
6. Use a new output directory; do not mix legacy and fast partial results.
