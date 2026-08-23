# Reference transport: identity, assumption, and diagnostics

## Algebra conditional on exchangeable moments

For kernel pair `(a,b)`, write the reference total trace product as

```text
T_R = sum_i K_a(ii)K_b(ii) + sum_{i != j} K_a(ij)K_b(ij)
    = D_R + O_R.
```

If the expected same-individual contribution per exchangeable person and the
expected ordered different-individual contribution per pair are common between
reference and study, then

```text
E[D_S] = (N_S/N_R) D_R,
E[O_S] = N_S(N_S-1) / [N_R(N_R-1)] O_R,
```

which gives

```text
T_S|R = (N_S/N_R) D_R
      + N_S(N_S-1)/[N_R(N_R-1)] (T_R-D_R).
```

`N_R` and `N_S` are counts of exchangeable individuals. They are never replaced
by residual ranks, average per-variant `N`, or `sqrt(N_j N_m)`. At `N_S=N_R`
the transformation is exactly the algebraic identity `T_S=T_R` for the stored
reference quantities; away from equality it is a moment-transport estimator,
not a finite-sample identity for newly projected study kernels.

## Why projection makes transport an assumption

Cohort centering, fixed-effect projection, separately estimated PCs, leverage,
environment scaling, and post-projection column standardization all change
kernel entries. Transport therefore requires stability of the induced feature
moments, not merely the same nominal ancestry label. Sufficient asymptotic
conditions include:

- exchangeable sampling within the target population and stable ordered-pair
  moments;
- a fixed feature convention and compatible variant/annotation axis;
- uniformly bounded fourth moments of genotype and environment features;
- bounded maximum leverage with `rank(C)/N -> 0`;
- stable environment second and fourth moments and genotype-environment
  dependence;
- common external PC/loadings or asymptotically equivalent projector spans;
- no phenotype-selection mechanism that changes unmodeled GxE moments; and
- reference and study population mixtures converging to the same limit.

Under these conditions, empirical same-person and ordered-pair averages
converge to common limits, and the displayed scaling is consistent. Separate
PC estimation, ancestry/prevalence shifts, high leverage, or selection on
unmodeled GxE can violate the premise even as probe count tends to infinity.

## Implemented diagnostics

Transfer now records reference/study `N`, reference/study residual rank, both
scales, whether `N_S>N_R`, same-person symmetry error, and the same-person
minimum eigenvalue. Material asymmetry is rejected before a within-tolerance
symmetrization. The unbiased finite-probe same-person U-statistic is allowed to
be indefinite and is not projected to PSD.

When `N_S>N_R`, `extrapolation=true`: the different-person scale grows faster
than the same-person scale and any ordered-pair moment error is amplified. The
tested conservative regime is therefore `N_R >= N_S`, with matched projector,
ancestry, environment distribution, and variant axis. This is a recommendation,
not a proof of negligible transport error.

## Available quantitative evidence

The deterministic independent-reference simulation in
`benchmarks/reference_transfer_simulation.json` used seed 20260814. Median
absolute relative errors for three reported quantities were
`[2.1716%, 3.1353%, 2.2305%]` under same-/different-person transport versus
`[9.2679%, 86.4641%, 23.5139%]` under naive squared-rank scaling. Unit tests
cover `N_S<N_R`, equality, and `N_S>N_R` and verify the extrapolation flag.

This evidence does not cover the requested ancestry, overlapping-cohort,
separately estimated PC, prevalence, leverage, selection, or missingness grid.
No improved-admixed-cohort claim is supported by this audit.
