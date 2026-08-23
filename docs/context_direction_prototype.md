# Experimental summary-level context direction

Stage 07B implements a deliberately narrow research prototype for learning one
fixed-metric context direction

\[
e_\omega=Z\omega,\qquad \omega^\mathsf{T}\Sigma_Z\omega=1.
\]

It is not a replacement for the full covariance-surface model or for ENGINE.
The fitted reduced model contains exactly four kernels, in this order:

1. additive genetic, \(G\);
2. interaction genetic, \(G\times E(\omega)\);
3. identity residual, \(N=P\);
4. context-dependent residual, \(N\times E(\omega)\).

The first version excludes additive--interaction covariance. That exclusion is
part of the estimand and is recorded in every manifest.

## Compact contractions

For \(L\leq3\), `ContextPairIndex` stores the \(L(L+1)/2\) environment pairs in
diagonal-first order. The direction vector

\[
c_p(\omega)=
\begin{cases}
\omega_l^2,&p=(l,l),\\
\omega_l\omega_m,&p=(l,m),\ l<m
\end{cases}
\]

does not contain an extra factor of two. The off-diagonal genetic basis kernel
itself is \(F_lF_m^\mathsf{T}+F_mF_l^\mathsf{T}\). Likewise, the residual
off-diagonal basis is
\(P\operatorname{diag}(2Z_lZ_m)P\). Consequently,

\[
K_I(\omega)=\sum_p c_pK_{I,p},\qquad
D(\omega)=\sum_p c_pR_p.
\]

`DirectionReferenceContractions` stores the Gram and same-person matrices for
`[G, I_pairs]`. `DirectionTraitContractions` stores the Gram, phenotype RHS,
and traces for `[G, I_pairs, N, D_pairs]`. These objects contain only small
arrays whose dimensions depend on \(L\). `evaluate_context_direction` and
`optimize_context_direction` cannot accept or revisit genotype, phenotype, or
row-level context arrays.

Reference genetic moments are transported with separate \(N\) and
\(N(N-1)\) same-person/different-person scaling. Genetic--residual and residual
moments come from the study contractions exactly. Every direction is normalized
with the immutable declared \(\Sigma_Z\), then displayed with the largest
absolute coordinate positive (lowest-index tie).

## Objectives

Each fixed-direction call assembles and solves the same full-rank four-kernel
normal system without a ridge or clipped eigenvalues. The objective must be
named explicitly:

- `interaction_coefficient`: the raw fitted interaction coefficient;
- `interaction_trace_contribution`: the coefficient times
  \(\operatorname{tr}(K_I)/r\);
- `he_moment_gain`: \(q^\mathsf{T}T^{-1}q\) for the four-kernel system minus
  the corresponding value for the same-direction nuisance system `[G,N,D]`.

The last quantity is a descriptive HE moment-fit improvement. It is not a
likelihood ratio, calibrated test statistic, or variance estimate. The
optimizer uses deterministic whitened axes and pair sums, SLSQP on the unit
sphere, and an independent fixed angular/Fibonacci grid check. Rank failures
and candidates above the declared condition-number ceiling fail closed.

## Exact variant-fold cross-fitting

`build_context_direction_crossfit_contractions` assigns whole user-declared
variant/LD blocks to two deterministic, mass-balanced folds. It builds fresh
reference and trait tensors from each exact disjoint subset. It never obtains a
half-genome object by subtracting the Stage-04 approximate-jackknife
contributions, and it never reuses a full-fold same-person matrix as a proxy for
a large deletion.

`crossfit_context_direction` optimizes on fold A and evaluates that fixed
direction on fold B, then reverses the roles. The equal-weight mean of the two
held-out objective values is reported with both directions, both training
objectives, both held-out objectives, subset hashes, and their metric
alignment. The in-sample optimized objective is explicitly not called
unbiased. Two-fold direction variation is descriptive; nonlinear selection
inference is not calibrated in this prototype.

## Development limits

The builders are dense NumPy correctness oracles. They materialize feature and
kernel arrays and are suitable only for small validation data. Optimization is
cheap after contraction, but contraction construction is not genome-scale.
There is no public artifact format or CLI in Stage 07B. Production use would
require block-streamed protected GEMMs, bounded aggregate storage, file-backed
alignment, more than two folds or a calibrated uncertainty procedure, and a
separate scientific decision that the reduced direction adds value beyond the
full \(\Omega\) modes from Stage 07A.
