# Generalized per-variant GxE LD scores: production use

The production estimator implements `generalized_gxe_variant_ldscore_v1`.
It uses global variant-axis Rademacher probes, one source traversal, a hard
source-sealing barrier, and one target-scoring traversal. A successful run has
exactly two genotype passes, `2*M` retained-variant visits, and zero duplicate,
retry, repair, fallback, and integrity-failure counts.

This is not the sample-probe contextual action estimator.

The same-person term is formed exactly as
`component_kernel_diagonal @ component_kernel_diagonal.T`. Its diagonal rows
are accumulated during the existing pass 2; no third genotype pass is used.

## Output mode

The default is release-safe summary mode:

```bash
python example/estimate_generalized_gxe_variant_ldscore.py \
  --mode summary --output /tmp/summit-generalized-example
```

For internal annotation reuse, opt in explicitly:

```bash
python example/estimate_generalized_gxe_variant_ldscore.py \
  --mode composable --output /secure/internal/reference
```

Composable output contains the annotation matrix, the full directional panel,
and a sample-aligned `C x N` component-kernel diagonal. SUMMIT warns when this
mode is written. Do not publicly share these artifacts. Use
`compose_generalized_gxe_variant_references_v1` to select columns from
compatible bundles and build a normal fit-ready reference without genotype
access. Trait artifacts that retained per-SNP sufficient statistics can be
aligned to the selected annotation panel with
`reaggregate_generalized_gxe_trait_summary`.

Composable artifacts are written as uncompressed NPZ containers. Their large
FP64 directional panels are effectively incompressible, so DEFLATE added a
long serial publication step without materially reducing storage. Summary
artifacts remain compressed.

## Definitive jackknife boundary

The LD-score estimator has no block count, block-ID vector, or jackknife
argument. Pass 2 emits one fixed directional LD-score row for every target
SNP. It never performs either-sided deletion.

After the complete genome-wide panel exists, a separate inference reducer uses
`--njack` to sum target-SNP contributions by block. Normal-equation replicate
`g` subtracts block `g`, renormalizes by retained annotation mass, and keeps
all retained-SNP LD scores fixed. It does not reopen genotypes or rebuild the
pass-1 source sketches. Trait-side scores, information, and heteroskedastic
information are also generated per SNP before this block reduction.

This is the canonical method:

```text
frozen_full_genome_variant_ldscore_delete_block_v1
```

## Planning the two-pass estimator

The planner performs no genotype I/O and intentionally has no jackknife
option:

```bash
summit-generalized-gxe-variant-ldscore plan \
  --samples 300000 --variants 1000000 \
  --basis 3 --annotations 1 --probes 128 \
  --memory-bytes 68719476736 --genotype-format bed --threads 8 \
  --variant-block-width 4096 --rhs-tile-columns 36 --rhs-policy tiled
```

`variant-block-width` is an I/O/memory tile and is unrelated to jackknife
blocks.

The plan reports separate target and source probe widths plus source/target
annotation batch widths. These are execution choices only: batching joins
independent GEMM columns so that a decoded genotype block can be reused, while
each probe contribution and component reduction retains its defined order.
The planner accounts for the batched RHS, GEMM output, pair-reduction, and
component-diagonal buffers before admitting a plan.

When every variant has exactly one nonzero annotation entry (as in disjoint
MAF--LD bins), the native executor automatically compacts pass-1 source
products by bin and fuses pass-2 pair formation with the ordered bin
reduction. Overlapping annotations retain the general dense path. Detection is
exact, is reported in native telemetry, and does not alter either genotype
traversal or the FP64 estimator.

With `rhs-policy=auto`, the planner retains a read-only, probe-tile-major RHS
when the complete allocation fits the memory limit. Otherwise it constructs a
bounded RHS tile inside pass 2. Both paths use the same two genotype passes and
produce the same estimator. The production workflow leaves the target probe
width planner-selected by default; `--probe-tile-width` remains available for
an explicit override. The real-data and simulation runners use the full probe
range as the preferred pass-1 source tile, subject to the planner's memory
admission.

## Progress and cancellation

Production reference runs call the native executor with progress reporting
enabled. In a terminal this is a lightweight `tqdm` block bar with the current
phase/subphase, unit count, and peak RSS. Non-interactive logs emit a start
record, phase changes, periodic status (five minutes by default), and
completion; they include the current subphase age and a phase ETA after the
first completed block. The native snapshot is also available through
`executor.progress()` and includes block, variant, subphase-unit,
elapsed-time, heartbeat, and peak-RSS fields.

`executor.request_cancel()` requests termination at the next safe native tile
boundary. Cancellation never publishes a partial reference; the context moves
to the failed/cancelled state instead.

## End-to-end example

The example estimates per-SNP reference scores first and then creates an
inference-ready artifact with 20 post-hoc normal-equation blocks:

```bash
env BLIS_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 \
  python example/estimate_generalized_gxe_variant_ldscore.py \
  --output /tmp/summit-generalized-example \
  --probes 16 --njack 20 --threads 4
```

Inspect the completed artifact with:

```bash
summit-generalized-gxe-variant-ldscore inspect \
  /tmp/summit-generalized-example.generalized-gxe-variant-ldscore-v1.npz
```

The active real-data runner is
`/home/bronsonj/summit_general_gxe/scripts/run_age_sex_full.py`, which delegates
to `run_age_sex_pilot.py`. Core reference and trait routines are in
`scripts/generalized_gxe/workflow.py`. On Hoffman, request at most eight slots;
locally, use available cores and memory without oversubscribing.

## Compatibility and verification

Compatibility is checked from the concrete scientific axes and values needed
for the computation: sample/variant counts, component order, annotation names
and masses, inference block assignment, residual names, and genotype affine
scale. The generalized path does not require file, manifest, source-tree,
binary, or reference-panel digests.

The descriptor-owned native executor retains decode/imputation ownership
through both passes and publishes the observed per-variant affine means and
inverse scales for the study-side scan. Qualification requires dense-oracle
comparisons, clean pass ledgers, finite outputs, per-SNP/full/block
reconstruction, heteroskedastic simulations, probe-count/seed sensitivity,
and calibrated post-hoc jackknife coverage.
