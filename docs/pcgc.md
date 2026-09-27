# PCGC scientific contract

The binary implementation lives in `summit.pcgc`, with risk preparation in
`summit.sumstats.binary`. The quantitative HE path is unchanged when no binary
option is supplied. The [implementation plan](pcgc_implementation_plan.md)
defines the staged qualification protocol.

For commands and input formats, start with
[Binary traits and PCGC](wiki/Binary-traits-and-PCGC.md).

All five binary methods report point estimates and SNP-block jackknife SEs
for components and totals. Binary inference defaults to 200 contiguous SNP
blocks; `--njack` selects another integer count of at least two. No research
flag is required. The quantitative command's defaults are unchanged.

The [interface and calibration review](pcgc_interface_audit.md) explains the
upstream comparison, the block count, and why the earlier qualification screen
no longer controls SE availability. The [release validation](pcgc_release_validation.md)
retains the measured calibration results across methods. These results do not
establish calibration outside the tested sampling and risk models.

The [second uncertainty and scaling audit](pcgc_second_audit.md) compares the
actual upstream jackknife routines, checks the expected jackknife covariance
analytically, and documents the single-feature reference specialization and
full-sample memory plans.

## Scientific contract

Let population prevalence be K and the analyzed sample case fraction be P.
Sampling is assumed to depend on case status alone. For population risk k_i,

```
a   = K (1-P) / [P (1-K)]
p_i = k_i / [k_i + a (1-k_i)]
z_i = (y_i-p_i) / sqrt[p_i (1-p_i)]
d_i = normal_pdf(normal_quantile(k_i)) sqrt[p_i (1-p_i)] / [k_i (1-k_i)]
```

To first order in between-person liability correlation,
`E[z_i z_j] = d_i d_j sum_a theta_a K_a(i,j)`, for different people. The
population-scaled annotation kernel is `K_a = X diag(A_a) X.T / M_a`, with
`M_a = sum_j A_ja`. Residual liability variance, including genetic variance,
is one. The residual environmental variance is not fixed to one separately.

The standard method uses `B_a = diag(d) K_a diag(d)` and raw score response
`v = d*z`. Its equations are

```
H_ab = trace(B_a B_b) - dot(diag(B_a), diag(B_b))
b_a  = sum_j A_ja [(X_j.T v)^2 - sum_i X_ij^2 v_i^2] / M_a
H theta = b
```

No phenotype renormalization, quantitative variance row, residual variance
parameter, ridge, nonnegative constraint, or clipping of negative estimates
is added. Singular or nonpositive normal matrices fail explicitly. Small
probe counts can fail this check even if the exact matrix is positive.

Population risks can be supplied or fitted by the ascertainment-aware probit
likelihood. The fit includes an intercept internally, rescales covariate units
for optimization, checks rank, diagnoses complete/quasi separation by linear
programming, and checks its analytic score and information matrix. It does not
substitute ordinary logistic GWAS beta/SE or SAIGE statistics for raw scores.

Marginal component estimates equal `theta/(1+V_cov)` under the stated
independent-covariate liability model. Supplied-risk input can include known
population `V_cov`; otherwise the variance of the risk predictor is estimated
with inverse-ascertainment weights. Outputs retain the scale and risk metadata.
The jackknife does not refit risks or capture uncertain external prevalence.

## Methods

| Method | Reference and score | Important limit |
|---|---|---|
| `liability` | Constant-risk PCGC, equivalent to a scalar off-diagonal HE conversion | Requires individual risks equal to population prevalence |
| `pcgc` | Trait-specific `d` features, raw `X.T (d*z)` scores | First-order liability approximation; unprojected, population-scaled genotypes |
| `pcgc-inverse` | Ordinary genotype features, raw `X.T (z/d)` scores | Different pair weighting and potentially heavy-tailed noise |
| `pcgc-basis` | Generalized features `diag(phi_q) X`, contracted using `d=Phi c` | CLI requires an exact span; approximate spans remain a private research API |
| `pcgc-ld` | Independent population LD with finite-reference diagonal subtraction | Explicit factorization approximation requiring suitable risk/genotype dependence and reference matching |

Study-specific univariate references use the single-feature specialization of
SUMMIT's generalized variant-probe machinery, with no projection. Cross-trait
preparation uses the shared multi-feature engine. Exactly two study genotype traversals
produce the reference and raw trait rows together. BED/PGEN reading, dosage
precision, missing-value handling, affine scaling, protected matrix products,
probe counters and reference work planning reuse existing SUMMIT code.

Basis off-diagonal coefficients are `c_q*c_r`, once: SUMMIT's symmetric pair
kernel already includes both orientations. Fixed-probe tests establish exact
contraction, including nontrivial off-diagonal terms. For one requested risk
contraction, preparation now forms `d = Phi c` before reference calculation:
one feature produces the same answer without unused basis-pair products.
The shared cross-trait reference still retains its multiple features. The private approximate
basis API reports both relative L2 and maximum relative sensitivity errors;
these diagnostics alone do not establish unbiasedness.

The external-LD path uses one study scoring traversal and two independent
reference traversals. For reference size R, its per-SNP population moment is

```
L_pop(j,b) = sum_k A_kb [(X_j.T X_k)^2 - sum_i X_ij^2 X_ik^2] / [R(R-1)]
```

It transfers this through `sum_{i!=j} d_i^2 d_j^2`, computed as
`(sum_i d_i^2)^2 - sum_i d_i^4`. Using `N(N-1) mean(d^2)^2` would omit a finite
risk-weight correction. Even with this correction, transfer approximates the
weighted study normal matrix, not an identity that removes ascertainment or
reference mismatch. The reference must have the identical variant/allele
axis and declared population scale. Overlapping study/reference IDs fail.

PC adjustment is not performed. In general, `P diag(d) X` and
`diag(d) P X` differ, and projecting heteroskedastic binary residuals creates
off-diagonal noise. Passing ordinary covariate-projection options is rejected.
Risk covariates in the current qualified-model candidates are exogenous;
genotype means that depend on those covariates require a separate derivation.

## Preparation and inference

The sample TSV has `FID IID Y` columns, with Y coded 0/1. Extra genotype samples
may be excluded by omitting them from this table. IDs are aligned to genotype
order; unknown or duplicate IDs and missing phenotype/covariate values fail.
Risk covariates and basis columns must be numeric; duplicate headers fail.

`--binary-scale` accepts an existing sealed SUMMIT affine scale with
`provenance.population_scale=true`, or a TSV containing exactly
`SNP A1 A2 MEAN INV_SD`. Each genotype SNP must appear once. A1 is the counted
allele: BED A1 or PGEN REF in the shared reader. The vectors describe the
population, not the ascertained study. Missing genotypes are imputed to the
supplied population mean. No scale is silently re-estimated.

For example, with exogenous columns AGE and RISK_FACTOR:

```sh
summit --binary-method pcgc \
  --make-binary-sumstats people.tsv --geno study.bed \
  --binary-prevalence 0.1 --binary-genome-build GRCh38 \
  --binary-scale population_scale.tsv \
  --binary-covariates AGE,RISK_FACTOR \
  --binary-probes 256 --num-threads 4 --out study_pcgc

summit --binary-method pcgc \
  --h2 study_pcgc.binary.npz --out study_pcgc_fit

# Choose a different block count when appropriate for the analyzed region:
summit --binary-method pcgc \
  --h2 study_pcgc.binary.npz --njack 100 --out study_pcgc_100_blocks
```

Use `--binary-risk-column RISK` instead of `--binary-covariates` for supplied
population risks, optionally with `--binary-covariate-variance`. Annotation TSVs
use `SNP` plus nonnegative weight columns. `pcgc-basis` additionally takes
`--binary-basis-columns` and `--binary-basis-coefficients`. `pcgc-ld` additionally
takes `--binary-reference-geno` pointing to an independent population sample.

Preparation writes a single `.binary.npz` joint reference/trait artifact;
inference writes `.binary.json` with component and total estimates, SEs,
jackknife covariance, block sizes, and conditioning metadata. The archive
records variant/allele axes,
annotation names, sample/risk/scale identities, prevalence, sample fraction,
method, scientific contract, probe specification, native build, feature
identity and numeric checksums. This initial format is
deliberately joint: it cannot accidentally pair ordinary LD scores with
incompatible binary trait summaries. Outputs never replace existing files.

The method at inference must match preparation. To change prevalence, risks,
sample selection or method, recompute the affected raw moments. Arbitrary
ordinary summary-statistic files, `--weight-mode`, `--rg`, ordinary projection
options, and reference overrides are rejected in binary mode.

## Uncertainty and qualification

PCGC uses `pcgc_frozen_offdiagonal_estimating_equations_v2`. Let `U_jb` be
the raw directional kernel-product row after subtracting its exact same-person
term. For retained target SNPs T, the delete-block equations are

```
H_ab(T) = sum_(j in T) A_ja U_jb / [M_a(T) M_b(full)]
b_a(T)  = sum_(j in T) A_ja rhs_j / M_a(T)
```

Only target SNP rows are removed. Full source kernels and their annotation
masses stay fixed, so every replicate estimates the original full-genome
component parameters. `H(T)` can be asymmetric; it must not be symmetrized.
The full directional probe matrix is also solved as stored, with diagnostics
for its actual condition number and the positive definiteness of its symmetric
part. Exact full matrices remain symmetric.

The previous PCGC implementation incorrectly applied retained masses on both
axes and subtracted the full same-person matrix in each deletion. The current
two-pass adapter accumulates per-SNP diagonal corrections, stores only corrected
LD rows, and discards the temporary diagonal rows. Reference estimation still
has no block IDs. Legacy schema-1 artifacts remain readable through the Python
point-estimate API.
CLI inference requires regeneration as schema 2 because those old archives lack
the per-SNP diagonal corrections needed for SEs. Generalized GxE's
separate deletion contract and implementation are unchanged.

SNP blocks do not remove people or change case fraction, risk covariates, or the
risk fit. Risks and population covariate variance therefore remain fixed in
these deletions. Both conditional and marginal SEs are reported, with explicit
conditioning metadata. These SEs do not propagate uncertain prevalence, fitted
nuisance-model uncertainty, or independent reference/probe Monte Carlo error.
Use comparable, sufficiently large contiguous genomic blocks; block sizes are
reported. The ordinary equal-group jackknife variance formula is used.

CLI inference always computes SEs. The default is 200 blocks; fewer than 200
SNPs require an explicit smaller `--njack`. Chromosome and delete-d schemes
are not exposed for binary inference. Choose blocks large enough to contain
local LD. The 50-block validation design used 4,000 SNPs with 40-SNP LD blocks;
Each of the 50 deletion groups contained two complete LD blocks. It was not a
production default. Reference preparation is independent of this choice.

Block membership comes from `JackknifeSpec` and `JackknifeDesign`. Annotation
reduction reuses the generalized reference reducer, and covariance uses the
same equal-block helper as contextual and cross-trait inference. The
chromosome module's unequal-unit pseudovalue calculation is a different
scheme and is not substituted for this SNP-block covariance. The low-level
`fit_moments` API can still omit `block_ids` for point-only numerical checks.

Inverse weighting has an additional statistical limitation. At the null,
`Var(z/d | C, sampled) = 1/d(C)^2`. For a Gaussian population covariate
`C ~ N(0,1)` and risk predictor `eta = gamma*C-t`, the lower-tail contribution
to its unconditional second moment is proportional to

```
exp[((gamma^2-1)*C^2 - 2*gamma*t*C + t^2)/2] / |gamma*C-t|.
```

This follows from the Gaussian tail approximation `Phi(eta) ~ phi(eta)/|eta|`.
Combining both tails shows that the moment diverges for `gamma^2 >= 1`.
This is a counterexample under the stated Gaussian model, not a rule that a
bounded covariate with variance one must fail. Clipping inverse weights would
change the estimating equation; the implementation does not silently do so.

## Cross-trait research API

`summit.pcgc.cross.prepare_pair` accepts a master genotype axis, each cohort's
aligned rows and its own risk model. A quantitative response must be on a
known population residual scale, without sample variance renormalization.
Both within-trait moments and the cross moment use the same two genotype
passes. `fit_pair` reports covariance before rg; rg remains undefined when
either estimated genetic variance is nonpositive. Ratios are not clipped.

The caller must declare `marginal_case_status_sampling_v1`: each cohort's
marginal inclusion law depends only on its own case status. Shared controls
selected by joint eligibility alone need not satisfy that law. The simulation
driver draws shared controls from the joint control population and compensates
the private-control mixture to preserve each marginal sampling distribution.
Other joint-selection designs are unsupported.

The unprojected features satisfy
`F_1.T F_2 = X.T diag(phi_1*phi_2) X`, which is symmetric. This gives the exact
ordered rectangular moment as `L_(01,01)/2 - L_(00,11)` in the existing symmetric
generalized-reference representation. The identity also holds with fixed
shared probes. Same-person products are subtracted only for actual aligned
overlaps, using `v_1*v_2`; neither an overlap count alone nor an ordinary
phenotype covariance supplies that correction. This identity is specific to
the unprojected features and does not change generalized GxE calculations.

Bivariate APIs remain research-only. There is no binary `--rg` dispatch or
public bivariate artifact contract. The optional uncertainty
uses the existing paired delta calculation on common frozen block deletions;
its assumptions and fixed-nuisance conditioning are the same as above.

The reproducible research drivers are:

- `scripts/pcgc/validate_pcgc.py`: paired Gaussian or discrete-haplotype
  simulations, known/fitted risks, all candidate methods, and the unchanged
  SUMMIT HE implementation followed by global liability conversion.
- `scripts/pcgc/validate_numerics.py`: numerical error from 16/64/256 probes
  against exact off-diagonal matrices and estimates.
- `scripts/pcgc/report_pcgc.py`: bias equivalence, SE/SD, coverage and null
  rejection with Monte Carlo intervals, failure accounting, component results
  and paired method differences. Pilot and confirmation are kept separate.
- `scripts/pcgc/validate_risk_reference.py`: small paired diagnostics for
  matched, population and mismatched references, estimated population scales,
  sampled-logistic risk fitting, prevalence errors and omitted risk covariates.
- `scripts/pcgc/validate_basis.py`: four/eight-bin continuous-risk bases,
  with individual/global sensitivity, moment and estimate errors.
- `scripts/pcgc/validate_cross.py` and `report_cross.py`: bivariate covariance
  and rg qualification, with explicit selection and overlap accounting.
- `scripts/pcgc/validate_cross_numerics.py`: fixed-probe rectangular-moment
  errors on one shared-control dataset, separated from sampling variability.
- `scripts/pcgc/validate_external_correction.py`: re-evaluation of the finite
  risk-pair correction on existing confirmation draws without new Gram products.
- `scripts/pcgc/run_native_tests.py`: explicit source-checkout testing against
  existing ABI-matching extensions, including test subprocesses.
- `scripts/pcgc/validate_jackknife.py` and `report_jackknife.py`: paired replay
  of the original SE failures under corrected source normalization and exact
  per-SNP diagonal removal; see the [audit](pcgc_jackknife_audit.md).
- `scripts/pcgc/validate_real_data.py`: bounded local hypertension comparison,
  including an exact small-reference oracle and authenticated PCGC fit reuse.
- `scripts/pcgc/benchmark_basis.py`: numerical parity and timing for collapsing
  a single requested basis contraction before reference calculation.

Gaussian liability sampling is tested against independent population rejection.
The discrete generator uses two independent Markov haplotypes; allele frequency
and LD are checked independently. Its prevalence threshold uses numerical
inversion of the exact liability characteristic function, verified against
Gaussian and finite-mixture cases and independent population sampling.

The scientific starting points are [Golan et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267399/)
and [PCGC-s at fdc5089](https://github.com/omerwe/PCGCs/tree/fdc5089f485fe25c04a8972665fee4216570764f).
The independent risk oracle checks the original `u0+u1` sensitivity expression
against the derivative of an ascertained bivariate-normal liability integral.

The external-reference comparison used S-PCGC revision `5211d17`, including
its [summary creator](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_sumstats_creator.py),
[reference calculation](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_r2.py)
and [inference](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_main.py).
Those routines have their own score normalization, annotation-square masses,
LD windows and projection/deflation conventions. This adapter uses the
explicit conditional-response, full-genome, finite-reference moment derived
above; it does not claim numerical or file-format equivalence to S-PCGC.
