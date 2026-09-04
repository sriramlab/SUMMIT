# Generalized G×E per-variant LD-score contract

Contract identifier:

```text
generalized_gxe_variant_ldscore_v1
```

## Features and kernels

With one sealed transformed genotype matrix `G`, fixed-effect projector
`P=I-UU^T`, basis `Phi=[phi_0,...,phi_{Q-1}]`, and `D_q=diag(phi_q)`, define

```math
F_q=P D_qG.
```

Pairs are all diagonals followed by lexicographic off-diagonals. For variant
`j`,

```math
B_{j,qq}=f_{qj}f_{qj}^T,
```

and, for `q<r`,

```math
B_{j,qr}=f_{qj}f_{rj}^T+f_{rj}f_{qj}^T.
```

For annotation `k`, mass `M_k=sum_j A[j,k]`,

```math
K_{k,p}=M_k^{-1}\sum_jA[j,k]B_{j,p}.
```

## Per-variant directional LD scores

Let `O(q,q)={(q,q)}` and `O(q,r)={(q,r),(r,q)}` for `q<r`. Define

```math
R_{ab}(j,m)=f_{aj}^Tf_{bm}/r.
```

For target pair `p` and source component `(ell,s)`,

```math
L_{j,p\to(\ell,s)}
=
\sum_mA[m,\ell]
\sum_{(a,b)\in O(p)}
\sum_{(c,d)\in O(s)}
R_{bc}(j,m)R_{ad}(j,m).
```

For target component `c=(k,p)` and source `d=(ell,s)`,

```math
DNUM[c,d]=\sum_jA[j,k]L_{j,p\to d},
```

```math
T[c,d]
=
\frac{r^2}{M_kM_\ell}
\frac{DNUM[c,d]+DNUM[d,c]}2.
```

## Exact component diagonals and same-person term

For pair `p=(q,r)`, let `kappa(q,r)=1` on the diagonal and `2` off the
diagonal. The normalized component-kernel diagonal is stored component-major:

```math
d[k,p,i]
=
\frac{\kappa(q,r)}{M_k}
\sum_j A[j,k]F_q[i,j]F_r[i,j].
```

The same-person trace matrix is exact, not probe-estimated:

```math
D=d d^T.
```

The numerator is accumulated once per already decoded pass-2 genotype block,
outside annotation/probe scoring loops. This adds no genotype traversal.

## Required randomized construction

Use variant probes `Xi in R^{M x B}`.

Pass 1:

```math
V_k=G(\sqrt{A_k}\odot\Xi),
\qquad
Y_{k,b}=P D_bV_k.
```

All `Y` are complete before pass 2.

Pass 2:

```math
U_{a,b,k}=G^TD_aY_{k,b}/r.
```

For each target SNP, form `Lhat` from rowwise products of the required `U`
panels using the same orientation expansion as above.

A normal reference run performs exactly two full genotype traversals. Tiling
must not add scans.

## Artifact modes and annotation composition

`summary` is the default, public-release-oriented mode. It publishes the
fixed-annotation aggregate moments and omits both the annotation matrix and
sample-aligned component diagonals.

`composable` additionally publishes the full directional panel, annotation
columns, and `component_kernel_diagonal` with shape `C x N`. Because the last
array is sample-aligned, composable artifacts currently emit a warning and
should not be publicly shared.

Container compression is not part of the estimator. Composable NPZ files use
uncompressed array members for high-throughput publication of their large,
effectively incompressible FP64 panels; summary NPZ files remain compressed.

Compatible composable bundles can be combined with
`compose_generalized_gxe_variant_references_v1`. The operation selects and
concatenates annotation columns in caller order, concatenates the matching
source-component columns and diagonal rows, then reruns only the in-memory
target reducer. It does not access genotypes. Compatibility checks cover
sample/variant dimensions, basis and component order, fixed-effect dimensions,
probe convention, and genotype scaling; callers remain responsible for using
the same ordered samples and SNPs.

## Post-hoc inference-time SNP-block jackknife

Reference pass 2 emits one fixed directional score row per target SNP. It does
not receive block IDs or accumulate deletion summaries. After the completed
genome-wide panel is available, a separate reducer computes

```math
BDNUM[g,c,d]
=
\sum_{j\in g}A[j,k(c)]\widehat L_{j,p(c)\to d}.
```

For block `g`, use

```math
DNUM^{(-g)}=DNUM-BDNUM[g]
```

and retained annotation masses. Do not recompute source sketches or retained
SNP LD scores. Reuse the full same-person matrix.

The inference-ready reference publishes `BDNUM` and block annotation masses,
but the generalized LD-score estimator itself has no block or jackknife input.
The inference adapter constructs each requested deleted Gram on demand without
genotype access. Trait-side scores, information, and heteroskedastic
information follow the same rule: estimate per SNP first, reduce by `--njack`
afterward.

Changing `--njack` therefore never changes a per-SNP reference score or
requires another genotype pass. The reference-estimation gates are the
two-pass ledger, `2M` retained-variant visits, and clean integrity counters.

Canonical method:

```text
frozen_full_genome_variant_ldscore_delete_block_v1
```

## Scope distinction

The sample-probe contextual estimator described in
`docs/context_native/scientific_contract_v1.md` estimates an aggregate kernel
Gram through `K_c Z`. It is separate and is not an implementation substitute
for this per-variant LD-score contract.
