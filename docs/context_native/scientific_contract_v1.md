# Contextual native scientific contract V1

> **Scope guardrail.** Sections 4–5 below define the separate sample-probe
> aggregate contextual covariance estimator (`Z in R^{N x B_T}`, kernel actions,
> and grouped action numerators). They are not the generalized per-variant G×E
> LD-score estimator. The latter uses variant-axis probes, two complete genotype
> passes, and fixed-full-genome-LD-score SNP-row deletion; see
> `docs/generalized_gxe_variant_ldscore_contract.md` and the architecture ADR.


Status: the scientific contract was frozen in Stage 0 and is implemented by
both the dense differential path and the private stable-V1 path. The latter
uses descriptor-backed native reference and trait executors plus a Python
summary fit. It is not exposed through the public `summit` CLI, and bounded
qualification does not by itself authorize target-scale production use.

## Active implementation bindings

The authoritative executable implementation in the current tree is under
`src/summit/context`. The formulas below bind to these active symbols:

| Contract area | Active symbol |
|---|---|
| Pair/component order and hashes | `spec.ContextPairIndex`, `spec.ContextComponentIndex` |
| Projection and feature order | `oracle.rank_revealing_projector`, `oracle.common_scale_features` |
| Dense kernels and exact moments | `oracle.dense_genetic_kernels`, `oracle.kernel_gram`, `oracle.exact_same_person_matrix` |
| Fixed-probe actions | `oracle.context_kernel_actions`, `oracle.hutchinson_gram` |
| Reference construction and grouped deletion | `reference.build_context_reference`, `reference.reference_moments_after_deleting_groups` |
| Variant-probe U-statistic | `reference.same_person_ustatistic` |
| Trait moments and grouped deletion | `summary.build_context_trait_summary`, `summary.trait_moments_after_deleting_groups` |
| Population transfer | `oracle.transfer_reference_gram` |
| Raw solve and optional PSD interpretation | `fit.fit_context_model`, `fit.project_genetic_coefficients_psd` |
| Strict-disjoint annotations | `annotations.build_disjoint_annotation_partition`, `annotations.fit_annotation_context_model` |
| V1 axes, scale identity, and admission | `schema.ContextSchemaIdentityV1`, `schema.GenotypeScalePlanV1`, `schema.ContextAdmissionLedgerV1` |
| Stable native reference execution and artifact | `reference_v1.run_contextual_reference_v1`, `reference_v1.adapt_native_contextual_reference_v1`, `reference_v1.load_contextual_reference_v1` |
| Stable native trait execution and artifact | `trait_v1.run_contextual_trait_v1`, `trait_v1.adapt_native_contextual_trait_v1`, `trait_v1.load_contextual_trait_v1` |
| Stable summary fit and artifact | `fit_v1.fit_contextual_model_v1`, `fit_v1.load_contextual_fit_v1` |

The fixture corpus in `tests/fixtures/context_native_stage0` and the tests
listed at the end of this document are part of this contract.

## 1. Inputs, common genotype scale, and projection

Let `G in R^{N x M}` denote the retained genotype matrix after **one sealed per-SNP affine transformation**. The same transformed SNP column is used for every contextual coordinate. There is no coordinate-specific scale and no post-projection per-coordinate or per-SNP normalization.

Let `U in R^{N x c}` have orthonormal columns spanning the fixed-effect design and define

```math
P = I-UU^T.
```

Let `Phi=[phi_0,...,phi_{Q-1}] in R^{N x Q}` and `D_q=diag(phi_q)`. The contextual feature matrices are

```math
F_q=P D_q G.
```

This order is binding. In general,

```math
P D_q G \ne P D_q(PG),
P D_q G \ne D_qPG.
```

Never materialize `P` for large `N`; apply it as `X-U(U^T X)`.

## 2. Pair and component indexing

Pairs are all diagonals followed by lexicographic off-diagonals:

```text
(0,0),...,(Q-1,Q-1),(0,1),(0,2),...,(Q-2,Q-1).
```

Let

```math
P_g=Q(Q+1)/2,
\eta_{qr}=1\quad(q=r),
\eta_{qr}=2\quad(q<r).
```

Components are annotation-major and pair-minor:

```math
c(k,p)=kP_g+p.
```

Serialize the full pair map and component map with digests. Never infer order from shape.

## 3. Annotation-normalized kernels

For annotation `k`, let `A_k=diag(A_{1k},...,A_{Mk})`, with finite nonnegative weights and

```math
M_k=\sum_j A_{jk}>0.
```

For pair `p=(q,r)`, `q<=r`, define raw and normalized kernels

```math
\widetilde K_{k,qq}=F_qA_kF_q^T,
K_{k,qq}=\widetilde K_{k,qq}/M_k,
```

```math
\widetilde K_{k,qr}=F_qA_kF_r^T+F_rA_kF_q^T,
K_{k,qr}=\widetilde K_{k,qr}/M_k\quad(q<r).
```

The genetic covariance model is

```math
\Sigma_g=\sum_k\sum_{q\le r}\omega_{k,qr}K_{k,qr}.
```

`omega_{k,qr}` is stored once. Unpacking to a symmetric matrix sets both `Omega[k,q,r]` and `Omega[k,r,q]` to the same stored value. Do not multiply or divide off-diagonal coefficients by two.

For contextual rows `phi(e),phi(e')`,

```math
C_k(e,e')=\phi(e)^T\Omega_k\phi(e').
```

Factors 1, 2, and 4 arise because an off-diagonal kernel contains two directional terms. No additional packing factor is allowed in the Gram, same-person matrix, RHS, or covariance surface.

## 4. Fixed sample-probe reference Gram

Let `Z in R^{N_R x B_T}` be explicit or counter-generated Rademacher sample probes, fixed independently of tiling. Let `Z_P=PZ`.

For every contextual coordinate,

```math
S_q=G^TD_qZ_P\in R^{M\times B_T}.
```

For annotation `k` and source coordinate `t`,

```math
H_{k,t}=G(A_kS_t)\in R^{N_R\times B_T}.
```

For component `c=(k,(q,r))`, define raw and normalized actions

```math
\widetilde W_c=P[D_qH_{k,r}+1_{q\ne r}D_rH_{k,q}],
W_c=\widetilde W_c/M_k=K_cZ.
```

The fixed-probe Gram is

```math
\widehat T_R[c,d]=B_T^{-1}\langle W_c,W_d\rangle_F.
```

Normalize by the probe count exactly once. Record pre-symmetry error before an optional final symmetry copy.

## 5. Exact grouped reconstruction of the fixed-probe Gram

Stable output reduces directly to group numerators; it never retains per-SNP
`C x C` values.

A direct raw-action contraction, retained as the differential oracle and possible tiny-`C` fallback, is

```math
L_{u,d}=G^TD_u\widetilde W_d\in R^{M\times B_T}.
```

For source component `c=(k_c,(q_c,r_c))`, define directional attribution to target `d` within deletion group `g`:

```math
d_{g,c\to d}=\sum_{j\in g}A_{j,k_c}\sum_v
\left[
S_{r_c,jv}L_{q_c,d,jv}
+1_{q_c\ne r_c}S_{q_c,jv}L_{r_c,d,jv}
\right].
```

The symmetric unnormalized group numerator is

```math
GNUM_g[c,d]=\frac{d_{g,c\to d}+d_{g,d\to c}}{2B_T}.
```

It must satisfy, for every component pair,

```math
\sum_gGNUM_g[c,d]
=M_{k(c)}M_{k(d)}\widehat T_R[c,d].
```

This is a mandatory semantic checksum. If normalized actions are used instead, restore the target component mass exactly once. Do not mix raw and normalized conventions inside one accumulator. Do not add an `eta` multiplier beyond the explicit directional expansion.

The two equivalent TN forms are both scientifically valid:

```math
G_v^T(D_u\widetilde W_d)
=(D_uG_v)^T\widetilde W_d.
```

The implementation may benchmark which operand to row-scale, but it must produce the same fixed-probe output and preserve protected-TN coverage.


A second exact formulation is implemented as the group-restricted-action path
at moderate component count and is selected only by an admitted and qualified
physical plan. Let `B_g` mask deletion group `g`, define

```math
H_{g,k,t}=GB_gA_kS_t,
```

```math
X_{g,c}=D_qH_{g,k,r}+1_{q\ne r}D_rH_{g,k,q},
```

and use the resident projected full raw action `\widetilde W_d`:

```math
C_g[c,d]=B_T^{-1}\langle X_{g,c},\widetilde W_d\rangle_F,
\qquad
GNUM_g=(C_g+C_g^T)/2.
```

Projection of `X_{g,c}` is unnecessary because `\widetilde W_d=P\widetilde W_d`. This produces exactly the same group numerator and can avoid the `Q*C*B_T` grouped-TN width. See `COMMON/03B_GROUPED_NUMERATOR_ALGORITHMS.md`.

## 6. Same-person matrix: signed variant-probe U-statistic

This is not a sample-probe diagonal estimator. Let `Xi in R^{M x B_D}` be shared Rademacher **variant probes**, fixed independently of tiling, with `B_D>=2`.

For annotation `k` and coordinate `q`,

```math
R_{kq}=P D_q G(\sqrt{A_k}\odot\Xi)\in R^{N_R\times B_D}.
```

For component `c=(k,(q,r))`, sample `i`, and probe `v`,

```math
g_{civ}=\eta_{qr}R_{kq,iv}R_{kr,iv}/M_k.
```

Accumulate globally across every probe tile:

```math
a_{ci}=\sum_vg_{civ},
B_{cd}=\sum_{i,v}g_{civ}g_{div}.
```

Then

```math
\widehat D_R[c,d]
=\frac{\sum_i a_{ci}a_{di}-B_{cd}}{B_D(B_D-1)}.
```

The finite-probe result may be signed or indefinite. Never clip, PSD-project, or replace it. Tile-local finalization is wrong because it omits cross-tile probe pairs.

## 7. Phenotype normalization and trait moments

For raw phenotype `y_raw`, project and normalize once:

```math
y=Py_{raw}\sqrt{r/(y_{raw}^TPy_{raw})},
\quad y^Ty=r,
\quad r=rank(P)=N-rank(U).
```

Because `Py=y`, define contextual scores

```math
a_{qj}=g_j^TD_qy=f_{qj}^Ty.
```

For component `c=(k,(q,r))`, the unnormalized group moments are

```math
RHSNUM_g[c]=\sum_{j\in g}\eta_{qr}A_{jk}a_{qj}a_{rj},
```

```math
TRNUM_g[c]=\sum_{j\in g}\eta_{qr}A_{jk}f_{qj}^Tf_{rj},
```

and, for residual/context basis column `d_h`,

```math
GRNUM_g[c,h]
=\sum_{j\in g}\eta_{qr}A_{jk}f_{qj}^Tdiag(d_h)f_{rj}.
```

Full moments divide the summed numerator by `M_k`. If an alternative implementation stores scores divided by `sqrt(r)`, it must restore `r` exactly once; never combine both conventions.

All `Q*L` phenotype score columns and all feature reductions for one admitted trait batch must be computed while each genotype block is resident. One descriptor traversal per admitted trait batch is the contract.

## 8. Residual/context kernels

For residual-basis column `d_h`, define

```math
R_h=Pdiag(d_h)P.
```

These are exact study-side moments and are never reference-transferred. Let `ell_i=sum_aU_{ia}^2` and `C_h=U^Tdiag(d_h)U`. Then

```math
y^TR_hy=\sum_id_{hi}y_i^2,
```

```math
tr(R_h)=\sum_id_{hi}-tr(C_h),
```

```math
tr(R_hR_l)=\sum_id_{hi}d_{li}
-2\sum_id_{hi}d_{li}\ell_i+tr(C_hC_l).
```

## 9. Population transfer

For reference size `N_R` and study size `N_S`,

```math
T_{S|R}
=\frac{N_S}{N_R}D_R
+\frac{N_S(N_S-1)}{N_R(N_R-1)}(T_R-D_R).
```

Use sample counts, never residual rank. When `N_S=N_R`, return stored `T_R` exactly apart from an optional final symmetry copy.

## 10. Approximate grouped deletion

Let `m_{gk}=sum_{j in g}A_{jk}`. For components `c,d`, deleting group `g` gives

```math
T_R^{(-g)}[c,d]
=\frac{M_{k(c)}M_{k(d)}T_R[c,d]-GNUM_g[c,d]}
{(M_{k(c)}-m_{g,k(c)})(M_{k(d)}-m_{g,k(d)})}.
```

For multiple deleted groups, subtract their numerators and masses. Reuse the full `D_R` unchanged. Apply analogous numerator subtraction and retained-mass normalization to trait RHS, trace, and genetic-residual terms. Residual-only moments remain fixed.

This is deliberately an approximate summary-only deletion procedure. Do not claim exact deleted reference kernels.

## 11. Small normal system and solve

The canonical ordering is genetic components first, residual components second:

```math
A=\begin{bmatrix}T_{S|R}&G\!\times\!R\\(G\!\times\!R)^T&RGRAM\end{bmatrix},
\qquad
b=\begin{bmatrix}genetic\ RHS\\residual\ RHS\end{bmatrix}.
```

Use one documented symmetric rank policy. The default raw solve fails on rank deficiency. No hidden ridge, inverse, pseudoinverse, residual-component elimination, clipping, or PSD replacement.

Raw coefficients and raw `Omega_k` are primary. Optional interpretations must
have separately named arrays, methods, tolerances, and diagnostics. Stable
`fit_v1` currently implements the optional PSD interpretation only;
`regularized_coefficients` is reserved contract vocabulary and is not an
accepted stable-V1 fit member.

## 12. Annotation interpretation

Strict-disjoint mode requires exactly one finite binary membership per retained variant. Every full and requested-deletion annotation mass must remain positive.

In overlap mode, coefficients are conditional component-regression coefficients. Do not expose them as standalone per-annotation covariance matrices. A combined total surface may be reported; per-component surfaces must be labelled contributions.

## 13. Closed identity and ownership requirements

The closed schema defines `pre_scaled_dense_v1` and
`sealed_variant_affine_v1`. The dense differential path can use the former;
stable descriptor-backed reference, trait, and fit artifacts require
`sealed_variant_affine_v1`. Scale identity includes the retained-variant
order, allele orientation/coding, centering source and formula, scaling
formula, missing-imputation rule, ploidy rule, and canonical digests of the
affine mean and inverse-scale arrays. Reference and trait plans must have
identical scale-plan digests; a shared free-text label is not an identity.

Artifact family, logical schema version, grouped physical encoding, native API
version, native backend version, and build ID are independent axes. Pair and
component maps are serialized and their digests are recomputed. Arrays are
owned, C-contiguous, and read-only after construction; canonical digests use
little-endian dtype and shape encodings. Nested artifact metadata is
recursively immutable.

Strict-disjoint binary and generic nonnegative-overlap modes are closed and
explicit. In overlap mode, individual annotation outputs are conditional
contributions only; the combined total is the default covariance surface.

## 14. Responsibility boundary

Native code owns descriptor/file identity, decode/impute/scale, projection and
all wide protected operations, fixed probes, full/grouped sufficient
statistics, scratch/thread/NUMA/integrity telemetry, mutation checks, phase
transitions, and compact output publication. Python owns basis, annotation and
group specifications; validates native output and schema; performs population
transfer and summary-only deletion renormalization; assembles and solves the
small normal system; creates explicitly named optional interpretations; and
writes fit/jackknife/surface artifacts.

No native result may contain an `N`- or `M`-axis array. Raw coefficients and
raw `Omega` remain primary and byte-distinct from optional PSD fields.
Summary-only deletion is always labelled approximate and reuses the full
same-person matrix.

A stable fit may persist a context evaluation grid only with the exact
`non_row_evaluation_grid_v1` role, its provenance digest, and the caller's
explicit non-row assertion. The grid must be independently constructed for
surface evaluation; copying subject context rows into it violates the stable
summary boundary even though the loader can validate only the declared role
and digest.

## 15. Executable invariant map

| Invariant | Executable gate |
|---|---|
| Pair/component order and frozen Q1--Q4 digests | `tests/test_context_stage0_schema.py` |
| Canonical endian/layout hashing and deep ownership | `tests/test_context_stage0_schema.py` |
| Closed schema axes, exact scale-plan identity, checked admission | `tests/test_context_stage0_schema.py` |
| `PDG` versus `PDPG` and `DPG` | `tests/test_context_stage0_fixtures.py` projection microcase |
| Directional factors 1/2/4, negative off-diagonal `Omega`, packing | `tests/test_context_stage0_fixtures.py` directional microcase and Q2--Q4 fixtures |
| Sample-count rather than residual-rank transfer | `tests/test_context_stage0_fixtures.py` transfer microcase |
| Duplicate-overlap rank failure | `tests/test_context_stage0_fixtures.py` rank microcase |
| Direct grouped TN equals group-restricted actions | `tests/test_context_stage0_fixtures.py` overlap grouped microcase |
| Full/grouped reference, U-statistic, reconstruction, every deletion | `tests/test_context_stage0_fixtures.py` Q1--Q4 oracle cases |
| Trait moments, transfer, raw fit, surfaces, and every deletion fit | `tests/test_context_stage0_fixtures.py` Q1--Q4 oracle cases |
| Strict-disjoint validation and overlap target mathematics | existing `tests/test_context_annotations.py` and `tests/test_context_dense_oracle.py` |
| Raw solve versus named optional PSD interpretation | existing `tests/test_context_fit.py` |
| Stable family/API/backend/state migration rejection | `tests/test_context_stage7_schema_migration.py` |
| Stable row-leakage and private-surface isolation | `tests/test_context_stage7_leakage_and_public_surface.py` |

All floating-point fixture comparisons use reviewed `atol=1e-9,
rtol=1e-11`. Exact index, hash, rank, count, and schema assertions use no
tolerance.
