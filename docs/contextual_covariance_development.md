# Experimental contextual covariance framework

This document describes the private, correctness-first Python framework under
`summit.context`. It is not a public CLI or a production artifact contract.
Existing additive SUMMIT and standardized SUMMIT-GxE modes are unchanged.

## Feature and component conventions

The development path records `feature_mode: raw_projected` and constructs

```text
F_q = P diag(phi_q) G
```

from one pre-scaled genotype matrix. Contextual feature columns are not
separately normalized after projection. Genetic pairs use diagonal-first order,
followed by lexicographic off-diagonals. The factor two for `q < r` is part of
the kernel and score component; the stored coefficient remains `Omega[q,r]`.

## Trait-summary schema version 1

`build_context_trait_summary` creates a private `summit.context.trait_summary`
object with:

- basis, named raw fixed-effect design, ordered variant, annotation, component,
  and approximate-LOO grouping hashes;
- common `N`, residual rank, `Q`, `K`, and residual-basis dimension;
- exact genetic right-hand sides and traces;
- exact genetic--residual entries and the complete residual-only system;
- raw per-SNP numerators for genetic `q`, genetic traces, and every
  genetic--residual entry;
- per-SNP annotation weights used to update remaining annotation masses.

Deleting a declared approximate-LOO group subtracts all of those raw
numerators, divides by the remaining annotation mass, and leaves residual-only
moments unchanged. It does not recompute an exact deleted genotype kernel.

The development writer emits one JSON manifest and one compressed NPZ file.
Both are SHA-256 bound, written atomically, and rejected on identity mismatch.
The payload can be inefficient and is not a commitment to the eventual public
or native layout.

## Reference schema version 1

`build_context_reference` constructs all annotation-by-context-pair genetic
kernels under the same component order. It offers two explicitly distinct
development modes:

- exact dense `T_R` and exact diagonal-kernel Gram `D_R` for deterministic
  oracle checks;
- a shared individual-probe Hutchinson estimate of `T_R` and a signed,
  variant-probe U-statistic estimate of `D_R`.

Both modes store a symmetric raw numerator for each SNP and component pair.
Those numerators sum to `M_a M_b T_R[a,b]` for the exact target or the same
fixed-probe randomized target. Approximate-LOO deletion subtracts declared SNP
groups, divides by both remaining component masses, and reuses the full `D_R`.
It is not an exact deleted-kernel or deleted-same-person calculation.

The manifest records probe counts, seeds or caller-supplied probe digests,
tiling, basis moments, rank/leverage, pre-symmetry and signed-eigenvalue
diagnostics, the genotype scaling convention, and every identity hash. The
backend is declared `python_numpy_reference`; native protected GEMM and ABFT
are marked not applicable at this prototype stage. Population transfer is a
separate API and uses sample counts `N` and `N(N-1)`, never residual rank.

## Fit schema version 1

`fit_context_model` consumes only a `ContextReference` and a
`ContextTraitSummary`. It cross-checks the shared basis, scale, variant,
annotation, component, and deletion-group identities; the two cohort-specific
fixed-effect hashes remain separately recorded. The canonical system is ordered
as all annotation-major genetic pairs followed by the declared residual basis.

The solver uses a full symmetric eigendecomposition with one absolute-plus-
relative rank policy. It reports signed eigenvalues, singular values, rank,
condition, retained and null directions, and solve residual. It does not add a
ridge or assume that the last residual component is an identity kernel.
Non-identifiable full models and leave-one-group replicates fail explicitly.

The summary-only jackknife subtracts the compatible trait and reference SNP
numerators, updates both annotation masses, and reuses full reference `D_R`.
For balanced equal-weight groups it writes all coefficient replicates and the
complete joint covariance. Raw coefficients and raw `Omega_k` matrices are
always retained. A separately requested covariance-aware PSD projection is
available only for disjoint annotations; singular jackknife covariance is
handled in its estimable subspace with a declared Euclidean tie-break.

For a supplied context grid, the fit records raw covariance, variance, and
correlation surfaces, undefined states, amplification contrasts, directional
orthogonal heterogeneity, basis-metric operator eigenvalues, PSD-defined
rank-one share, and separately labelled coefficient-times-trace summaries for
the full estimate and every approximate-LOO replicate.

## Numerical scope

All final accumulators are float64. The current implementation accepts dense
in-memory genotype arrays and streams them by variant block to exercise the
one-pass contract. Native decode, C++ optimization, production memory admission,
and ABFT integration are intentionally deferred until the complete framework is
scientifically validated.
