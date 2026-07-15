# Score-scale constrained LDSC weighting

## Scope

`--weight-mode ldsc` changes the univariate h2 estimating instrument while
retaining SUMMIT's summary-statistic moment. It is currently supported only by
the ordinary `--h2` path. The default `--weight-mode he` is unchanged.

This mode does not change the rg/genetic-covariance estimator. In particular,
`--intercept-weight-mode ldsc` controls only the bivariate nuisance-intercept
fit and is not a main covariance-LDSC estimator. Fast cached h2 is also not yet
wired to the LDSC path.

## Mean model and weighted estimating equation

For SNP `j`, SUMMIT reconstructs its exact score-scale statistic `z*_j` from
`BETA`, `SE`, and `N`, under the current h2 convention `cov_rank=0`, and uses

```text
q_j = (z*_j)^2 - 1.
```

Let `L_jk` be the LD score for annotation `k`, let `M_k` be that annotation's
fixed mass on the effect-reference SNP universe, and let
`n* = N_max - 1`. Define

```text
D_jk = n* L_jk / M_k.
```

The constrained score-scale LDSC estimate at one IRWLS update is

```text
h = (D' W D)^(-1) D' W q.
```

SUMMIT solves this as least squares on `sqrt(W) D`, using an SVD and requiring
full column rank. It does not solve the explicitly formed normal matrix, which
would square the condition number for correlated annotation columns.

This is a different estimating instrument from the default HE/SUMMIT system.
After eliminating the residual component, the default system uses annotation
columns as instruments, schematically

```text
(A' D) h = A' q,
```

whereas LDSC mode uses `W D` as the instrument. It therefore cannot be
implemented by changing a scalar weight inside the existing HE normal equations.

## IRWLS weights

Write

```text
L_j+ = sum_k L_jk,
M_+  = sum_k M_k,
h_+  = sum_k h_k.
```

Given a scalar regression weight LD score `Lw_j`, SUMMIT uses

```text
mu_j = 1 + clip(h_+, 0, 1) n* max(L_j+, 1) / M_+,
w_j  = 1 / {2 max(mu_j, 1e-3)^2 max(Lw_j, 1)}.
```

Only the variance and overcounting weights are floored. The regression design
`D` is never floored, so negative unbiased or sub-unit LD scores remain in the
mean equation.

`--ldscores-w` should point to a one-column LD-score file computed over the
regression SNP set. Its SNP order may differ from the primary LD file; SUMMIT
aligns by SNP ID and restricts the regression axis to the intersection. Finite
values below one, including negative unbiased estimates, are retained and
floored only when forming `w_j`. If `--ldscores-w` is omitted, `Lw_j=L_j+`.
That fallback matches the experimental implementation but a separate scalar
weight LD score is preferable for overlapping annotations.

The default is three closed-form updates. For one component, initialization is
the usual aggregate constrained-LDSC estimate

```text
h0 = M_+ sum_j q_j / {n* sum_j L_j+}.
```

For multiple components, initialization is the full-rank unweighted
multivariate least-squares estimate. This matches SUMMIT's experimental
closed-form IRWLS implementation. It intentionally differs from historical
`ldsc.py`, which freezes its initial weights for partitioned regression instead
of continuing multivariate IRWLS.

## Fixed reference moments

`M_k` is fixed across GWAS filtering and every jackknife deletion. It must not
be recomputed from the retained regression SNPs. When `--annot` is supplied,
SUMMIT reads its full pre-regression rows to calculate

```text
M_k    = sum_j A_jk,
O_kl   = sum_j A_jk A_jl,
M_ref  = number of reference annotation rows.
```

An explicit `--ldsc-m` overrides `M_k`, with chromosome-split `@` files summed,
but it must agree with the masses calculated from that same annotation file.
This validation prevents combining `.M_5_50` with an unfiltered annotation
overlap matrix. With no annotation and one LD-score column, an external `M`
directly supplies the all-ones reference count and overlap moment.

New genome-wide LD-score runs write `<out>.gw.M`. External LDSC `.l2.M` files
are also accepted when their annotation universe matches `--annot`.

For raw component contributions `h_k`, SUMMIT reports

```text
tau_k       = h_k / M_k,
h2_cat,k    = sum_l O_kl tau_l,
h2_total    = sum_k h_k.
```

Thus category h2 uses the same overlap adjustment as the existing H2 result
contract. `--enrich-mode` selects the enrichment definition; it does not change
the reported category h2 values.

## Jackknife semantics and cost

Every delete-block replicate repeats initialization and all IRWLS updates with
the same fixed reference moments. This is the exact refit semantics of the
experimental estimator. It differs from historical `ldsc.py`, which freezes
the full-data final weights when constructing delete values.

The implementation retains only the coefficient vector for each replicate and
the full-fit diagnostics, so memory does not grow as `R` full SNP-length weight
vectors. Runtime is nevertheless proportional to the number of exact refits:
roughly `O((R+1) I M K^2)` for `R` replicates, `I` updates, `M` regression SNPs,
and `K` annotation columns. Failed or rank-deficient delete refits are fatal;
SUMMIT does not silently compute an SE from a partial set.

## Difference from literal LDSC

Literal univariate LDSC ordinarily regresses a Wald `Z_j^2-1` response with
per-SNP sample size `N_j`. SUMMIT instead retains its exact score response and
the scalar `n*` used by the HE estimator. The two conventions coincide only in
special cases, such as effectively constant sample size with equivalent score
and Wald statistics. This mode should therefore be described as constrained
score-scale LDSC-style IRWLS, not as package-identical LDSC.

