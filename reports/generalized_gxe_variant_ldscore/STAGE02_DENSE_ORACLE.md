# Stage 02: dense oracle and scientific contract tests

Date: 2026-08-22

Base commit: `1885c640b3141c18cac337ad9743b05ac9a86203`

Scope: independent FP64 oracle and tests; no production executor

## Result

**PASS.** Exact per-SNP directional aggregation reconstructs the dense kernel
Gram for every required Q/K configuration. The intended frozen-full-genome
row-deletion jackknife differs from literal two-sided kernel deletion in a
frozen regression fixture. All focused, contextual, and mature G×E tests pass.

No discrepancy with the authoritative mathematical contract was found.

## Files

- `tests/generalized_gxe_variant_ldscore_oracle.py` is a test-only dense and
  fixed-probe oracle with no native-extension dependency.
- `tests/test_generalized_gxe_variant_ldscore_oracle.py` contains 14 focused
  contract tests.
- this report records the stage evidence.

No source under `src/` and no production schema, executor, artifact, CLI, or
numerical output path changed in this stage.

## Implemented equations

For an orthonormal nuisance basis `U`, the oracle constructs

```text
P   = I - U U'
F_q = P diag(phi_q) G.
```

For diagonal-first pairs `p=(q,r)`, each SNP atom is

```text
B[j,(q,q)] = f[q,j] f[q,j]'
B[j,(q,r)] = f[q,j] f[r,j]' + f[r,j] f[q,j]'  for q < r.
```

For annotation `k` with mass `M_k=sum_j A[j,k]`, normalized kernels are

```text
K[k,p] = sum_j A[j,k] B[j,p] / M_k.
```

The exact all-pairs correlation tensor is

```text
R[a,b,j,m] = f[a,j]' f[b,m] / residual_rank.
```

Using diagonal orientation set `{(q,q)}` and off-diagonal set
`{(q,r),(r,q)}`, the target-row directional score is

```text
L[j,p -> (ell,s)] =
    sum_m A[m,ell]
    sum_(a,b in O(p)) sum_(c,d in O(s))
    R[b,c,j,m] R[a,d,j,m].
```

The oracle separately stores directed and symmetric numerators:

```text
DNUM[c,d] = sum_j A[j,k(c)] L[j,p(c) -> d]
SNUM       = (DNUM + DNUM') / 2
T[c,d]     = residual_rank^2 SNUM[c,d] / (M[k(c)] M[k(d)]).
```

Exact trait objects are computed literally as

```text
rhs[c]    = y' K[c] y
trace[c]  = tr(K[c]).
```

The fixed-probe path exposes each layer independently:

```text
V[k]       = G (sqrt(A[:,k]) * Xi)
Y[k,b]     = P diag(phi_b) V[k]
U[a,b,k]   = G' diag(phi_a) Y[k,b] / residual_rank
Lhat       = probe means of the required rowwise U products.
```

It also exposes full and SNP-block directed numerators and the signed global
same-person U-statistic formed from completed `Y` panels.

For block `g`, the production jackknife oracle uses

```text
DNUM[-g] = DNUM - BDNUM[g]
M[-g]    = M - BMASS[g]
```

without changing any retained target row's full-genome LD score. A separate
literal oracle removes the block from both genotype/annotation SNP axes and
recomputes kernels. It is validation-only and is not the production method.

## Proof sketches encoded by the tests

Expanding `tr(K_c K_d)` into target and source SNP atoms gives a double SNP sum.
Expanding both atoms' diagonal/off-diagonal orientations gives exactly the four
`R` factors represented in the directional-score loop. Grouping first by the
target SNP produces `DNUM`; averaging the two directions removes reduction
orientation, and multiplying by `residual_rank^2/(M_c M_d)` reconstructs the
dense kernel Gram.

For the randomized construction, `E[Xi Xi']=I` converts the pass-1 weighted
source and pass-2 cross-sketch row products into the same source-SNP sum. The
fixed-probe tests compare literal panels, so Monte Carlo convergence is not used
to conceal a formula or tiling disagreement.

The same-person U-statistic subtracts equal-probe products from the product of
probe sums and divides by `B(B-1)`. It therefore uses distinct probe pairs and
requires only completed global pass-1 sources.

Frozen deletion subtracts target-row contributions but deliberately retains
source-block terms already embedded in every full-genome target score. Literal
two-sided deletion removes those source terms too, so the two objects need not
and generally do not agree.

## Fixtures and coverage

Fixed-seed fixtures use 11 samples and 10 variants with an intercept and a
nontrivial nuisance covariate. Genotypes are deliberately correlated with both
the context and nuisance direction. Annotation weights are continuous,
positive, overlapping when `K=2`, and not restricted to a partition.

Coverage includes:

- `Q=1,K=1` additive identity and literal squared feature correlations;
- `Q=2,K=1` diagonal/off-diagonal factors 1, 2, and 4;
- `Q=3,K=1` all six pairs;
- `Q=3,K=2` all 12 annotation-major components with overlapping annotations;
- signed off-diagonal target and source scores;
- explicit SNP atoms, normalized kernels, trait RHS, and traces;
- execution widths `(4,5)`, `(6,11)`, and `(10,37)` for variant/probe tiling;
- jackknife block boundaries at variants 3 and 7, which cross execution blocks;
- fixed-probe pass-1 sources, pass-2 cross-sketches, scores, aggregate
  numerators, block numerators, and masses;
- two-pass ledger equality to `2*M` visits with zero duplicate visits;
- frozen row deletion versus literal two-sided deletion;
- global same-person and normal-Gram Monte Carlo convergence; and
- a deliberately matched common-scale bridge to the mature X/W source and
  target helpers.

The X/W bridge compares only XX, XW, WX, and WW directions formed by diagonal
generalized pairs. It makes no claim that the mature constrained model contains
the generalized off-diagonal kernel coefficient.

## Numerical results

Maximum absolute exact Gram reconstruction errors:

| Fixture | Maximum absolute error |
|---|---:|
| `Q=1,K=1` | `0` |
| `Q=2,K=1` | `2.4868995751603507e-14` |
| `Q=3,K=1` | `5.6843418860808015e-14` |
| `Q=3,K=2` | `2.2737367544323206e-13` |

The overall exact maximum, `2.2737367544323206e-13`, is below the required
approximately `1e-11` FP64 tolerance.

Fixed-probe tiling maxima:

| Variant/probe width | Source maximum | Directional-score maximum |
|---|---:|---:|
| `4/5` | `3.5527136788005009e-15` | `2.1316282072803006e-14` |
| `6/11` | `3.5527136788005009e-15` | `2.8421709430404007e-14` |
| `10/37` | `0` | `7.1054273576010019e-15` |

Additional observed values:

```text
frozen-vs-literal deletion maximum absolute difference  38.383528751921929
same-person relative Frobenius error, B=40000            0.0061243376064175044
normal-Gram relative Frobenius error, B=40000             0.0016667036553635972
```

## Validation commands and results

Focused syntax and scientific suite:

```bash
/home/bronsonj/anaconda3/envs/summit/bin/python -m py_compile \
  tests/generalized_gxe_variant_ldscore_oracle.py \
  tests/test_generalized_gxe_variant_ldscore_oracle.py

PYTHONPATH=/tmp/summit-generalized-stage02-release.KvPKEh/install \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_generalized_gxe_variant_ldscore_oracle.py
```

Result:

```text
14 passed in 1.78s
```

Required contextual and mature G×E regression scope:

```bash
PYTHONPATH=/tmp/summit-generalized-stage02-release.KvPKEh/install \
SUMMIT_STAGE6_TEST_INSTALL=/tmp/summit-generalized-stage02-release.KvPKEh/install \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_generalized_gxe_variant_ldscore_oracle.py \
  tests/test_context*.py tests/test_gxe*.py tests/gxe_completion
```

Result:

```text
1478 passed, 5 skipped, 1 xpassed in 114.57s
```

The fresh Release build used live source commit
`1885c640b3141c18cac337ad9743b05ac9a86203` and the canonical tracked-source
digest `4d6d7647e72989f69083db9a7d3ea8c10140389942552729190595e30e8b09cc`.
An initial direct-CMake run omitted the digest because the new untracked tests
made the worktree dirty; that diagnostic run produced 43 provenance-rejection
failures and 1,435 passes. Rebuilding with the exact digest computed by the
same CMake manifest algorithm resolved all 43 failures without a source change.

No formatter is installed in the qualified environment. `py_compile`, an
explicit over-100-character line scan, and `git diff --check` are the available
focused static checks.

## Stop gate

Stage 02 passes: every exact per-SNP aggregation fixture reconstructs the dense
kernel Gram, fixed-probe tiling agrees at FP64 reduction error, the jackknife
counterexample is a regression test, and the required existing regression
scope is green. Stage 03 may begin after this focused oracle/test/report commit.
