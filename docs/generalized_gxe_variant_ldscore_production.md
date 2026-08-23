# Generalized per-variant GxE LD scores: production use

The production estimator implements `generalized_gxe_variant_ldscore_v1`. It
uses global variant-axis Rademacher probes, completes one source traversal,
crosses a hard source-sealing barrier, and completes one target-scoring
traversal. A successful reference run must report exactly two complete genotype
passes, `2*M` retained-variant visits, and zero duplicate, retry, repair,
fallback, and integrity-failure counts.

This is not the sample-probe contextual covariance/action estimator.

## Qualified native backend

The accepted Linux FP64 backend is a private, statically linked, pthreaded BLIS
runtime with immutable process-start thread ownership:

- BLIS commit `e8566eb3e773fb54d11b33e371d13f22d2941e50`;
- BLIS source-tree SHA-256
  `eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154`;
- BLIS archive SHA-256
  `720068171eea951a0bc634d2d1a829561d5a2bae630bce41a24c4f0edbef9d9b`;
- BLIS configuration family `zen` and threading layer `pthreads`; and
- CMake configuration with `GXELDCORE_USE_PRIVATE_BLIS=ON` plus the pinned
  archive, include directory, source commit, source-tree digest, and
  configuration family.

The resulting `gxeldcore.build_info()` must identify `BLIS`,
`private_static`, `upstream_blis`, `pthreads`,
`blas_runtime_owner_thread_enforced=true`, and
`blas_runtime_environment_immutable=true`. The extension has no dynamic
BLAS/BLIS dependency and does not export CBLAS/BLIS symbols.

Set thread ownership before Python imports SUMMIT. Keep it unchanged for the
life of the process:

```bash
export BLIS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export OMP_THREAD_LIMIT=32
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

Changing vendor thread ownership after initialization is outside the qualified
envelope. Shared/system OpenBLAS and dynamically reconfigured vendor-thread
teams are compatibility paths, not the accepted generalized GxE production
backend.

## Plan before execution

An installed build provides a clearly separated planning and inspection
command. This target-shape dry run performs no genotype I/O:

```bash
summit-generalized-gxe-variant-ldscore plan \
  --samples 300000 --variants 1000000 \
  --basis 3 --annotations 1 --probes 128 --jackknife-blocks 200 \
  --memory-bytes 68719476736 --genotype-format bed --threads 32 \
  --variant-block-width 4096 --rhs-tile-columns 36 --rhs-policy tiled
```

The qualified planner output records `probe_axis="variant"`, two planned
passes, 2,000,000 planned variant visits, 245 decoded blocks per pass, and a
13,173,684,173-byte modeled peak for this configuration. Planning is not a
substitute for admission on the execution host.

## Reproducible end-to-end example

The example below reads the repository's real-format `example/small` BED/BIM/FAM
fixture, constructs deterministic `Q=3`, `K=1` inputs, executes the native
estimator, enforces the clean-run ledger, writes the full per-variant
directional panel, and reloads the closed V1 artifact:

```bash
env BLIS_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 \
  python example/estimate_generalized_gxe_variant_ldscore.py \
  --output /tmp/summit-generalized-example \
  --probes 16 --blocks 20 --threads 4
```

The checked run had `N=8,430`, `M=14,821`, `P=C=6`, exactly 29,642 variant
visits, 15 decoded blocks in each pass, and zero duplicate, retry, repair,
fallback, or integrity-failure counts. Its inline directional panel had shape
`[14821, 6, 6]`. Inspect any completed artifact with:

```bash
summit-generalized-gxe-variant-ldscore inspect \
  /tmp/summit-generalized-example.generalized-gxe-variant-ldscore-v1.npz
```

The executor accepts stable read-only BED/BIM/FAM descriptors and retains
descriptor/decode/imputation ownership natively through both passes. The native
result seals the actual retained-variant order, BIM A1 orientation, per-variant
means, and inverse sample standard deviations observed in pass 1; pass 2 must
reproduce the same hashes. The artifact adapter rejects a caller-supplied scale
identity that differs from this decoder-derived plan.

## Jackknife boundary

Reference construction only accumulates full-genome directed numerators,
target-block directed numerators, and block annotation masses. It does not
materialize delete-block genetic Grams and does not run jackknife fits.

At inference time,
`reference_moments_after_deleting_variant_blocks_v1` subtracts requested target
SNP rows from the fixed full-genome sums, renormalizes by retained annotation
mass, and reuses the full same-person matrix. It never reopens genotypes or
recomputes retained-SNP LD scores. Cached delete Grams remain readable only for
older-artifact compatibility.

A changed-block-count invariance experiment is not a production requirement.
The enforced construction gates are the two-pass/`2*M` ledger, sealed source
and genotype-scale identities, block reconstruction, and clean integrity
counters.

## Qualification scope

The current production execution path is descriptor-owned PLINK BED, FP64,
dense protected NN/TN, on the pinned private pthread-BLIS backend. Qualification
includes dense-oracle differential tests, packed-versus-dense controls,
artifact corruption and source-mutation rejection, inference-time deletion and
fit tests, ASan+UBSan and UBSan runs, `B=128` and `B=1024`, a complete 454,207
variant real-BED traversal, and an `N=300,000` production-row execution.

The exact combined `N approximately 300,000, M approximately 1,000,000` data
set was not available locally, so no measured full-target runtime is claimed.
The planner admits that shape within 64 GiB, and the two large dimension-split
runs establish bounded memory and clean exact-two-pass behavior without
changing the architecture. See
`reports/generalized_gxe_variant_ldscore/STAGE09_PRIVATE_BLIS_REQUALIFICATION.md`
for exact commands, binary identity, timings, and limitations.
