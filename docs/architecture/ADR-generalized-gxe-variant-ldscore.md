# ADR: generalized G×E uses a two-pass per-variant LD-score estimator

Status: accepted
Date: 2026-08-21

## Context

SUMMIT contains a generalized contextual covariance estimator based on
sample-axis Hutchinson probes and kernel actions. SUMMIT also contains a mature
non-general G×E LD-score estimator based on variant-axis probes, global source
sketches, target scoring, and per-variant directional LD-score panels.

The generalized production requirement is to estimate each retained variant's
full-genome contextual/G×E LD-score contribution and use those fixed values in
summary normal equations and a local-LD SNP-block jackknife.

The two estimators can target the same aggregate genetic Gram in expectation,
but they have different probe axes, finite-probe realizations, artifacts,
computational dependencies, and jackknife semantics.

## Decision

Implement generalized G×E LD scores as a distinct variant-probe two-pass
estimator:

1. pass 1 constructs completed global source sketches
   `Y[k,b]=P diag(phi_b) G sqrt(A_k) Xi`;
2. pass 2 computes every target variant's cross-sketches and directional
   generalized LD scores;
3. only after the genome-wide per-variant panel is complete, a separate
   `--njack` reducer forms annotation-weighted full and block sums; and
4. SNP-block jackknife replicates subtract target rows from those fixed sums
   and renormalize masses. Retained SNP LD scores are not recomputed.

Exactly two complete reference-genotype traversals are a production invariant.
Probe and RHS tiling occur while one decoded genotype block remains resident.

The existing sample-probe contextual covariance estimator remains supported as
a separate estimator with a separate artifact identity. Its grouped-action
numerators are not used as the per-variant LD-score jackknife.

## Consequences

Positive:

- directly generalizes the mature additive/non-general source-target framework;
- produces the per-variant object required for the intended summary method;
- block-indexed inference sufficient statistics are reduced post hoc from the
  completed per-variant panel;
- science-neutral native optimizations can be reused; and
- estimator/artifact semantics are explicit.

Costs:

- pass 2 requires `K Q^2 B` target RHS columns;
- per-variant directional output has `K P_g^2` columns;
- global source panels must remain resident between passes; and
- the executor must invert legacy tile-major loops to guarantee two physical
  scans.

## Rejected alternatives

- Treating the sample-probe action Gram as though it were the per-variant
  generalized LD-score artifact.
- Exact delete-block recomputation for every jackknife group.
- Context-specific post-projection SNP normalization.
- A new independent genotype I/O/GEMM stack.
