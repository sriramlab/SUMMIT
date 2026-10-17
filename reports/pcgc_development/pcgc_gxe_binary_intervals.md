# Calibrated intervals for binary G×E PCGC components

Date: 2026-10-13. Branch `fix/pcgc-gxe-binary-intervals-20261013`, from
`origin/main` at `cc25375`. Simulation outputs are under
`/data1/bronsonj/summit_pcgc_binary_intervals_20261013/`.

**What was wrong.** The SNP-block jackknife for binary G×E PCGC assumes
that SNP blocks' estimating equations are independent. They are sums over the
same participants, and participant-level terms — threshold noise whose
variance depends on the genetic value, each participant's own signal,
case–control selection, and shared risk and context weights — make them
covary. On main (`fit_gxe`, exact references), with K = 0.02,
N = 32,768, M = 1,024, the jackknife variance of the null additive-by-context
covariance was half the true variance (ratio 0.48–0.50, coverage 82–84%
over 3,000 replicates). The omitted covariance changes sign with prevalence,
selection and risk fitting: jackknife variance ratios ranged from 0.44 to
1.26 across the grid.

**The fix.** Preparation can record each participant's row of each SNP
block's estimating equation in the existing first genotype pass. The fit adds
only the between-block participant covariance (an inclusion–exclusion over
"shares a SNP block" and "shares a participant", so nothing is counted twice)
and, with fitted probit risks, the risk-fit term the jackknife holds fixed.

**Calibration.** Over 22 result sets (73 component cells, excluding one
design that violates the jackknife's own LD assumption), the corrected
variance ratio was 0.89–1.14 (mean 1.00; 70 cells within 2 MCSE of 1). Null
components had 4.0–7.3% type-I error (mean 5.7%; jackknife 4.4–19.4%).
Coverage of 95% normal intervals was 90.8–96.0% (mean 93.9%). The
remaining shortfall is in variance components and comes from the Wald
interval rather than the variance: the SE estimate is strongly correlated
with the estimate (0.4–0.8), and coverage using the true SD is 94.2–96.1%. The jackknife has the same shortfall wherever its own
variance is unbiased. Main's experimental sampling SEs were also evaluated:
without architecture probes they are far too small for Ω; with them they
behaved like the new estimator at N = 8,192 but cost minutes per replicate.

**Update, 17 October 2026: opt-in again.** Simulations at realistic N/M reversed the
15 October decision.
- On real UK Biobank genotypes (EUR_338k, M = 448,562, 200 blocks, K = 0.02,
  h = 0.36, 500 replicates), the jackknife alone was calibrated at N = 337,845. Its variance
  ratios were 1.01, 1.12 and 1.02, and its null type-I error was 4.0%.
- In the same runs the correction inflated the null covariance's variance by 22%: ratio
  1.25 ± 0.09, E[z²] 0.80.
- With Gaussian genotypes at N/M = 0.75, in four designs (base, risk slope 0.6, fitted risks,
  Q = 3), the jackknife was calibrated or mildly anti-conservative, with ratios 0.84–1.06.
  The correction never improved coverage, and its E[z²] reached 1.68.
- The jackknife's deficit appears as N/M grows: ratio 0.83 at N/M = 2, 0.65 at 8, and 0.5
  at 32.
- Outputs, drivers and notes are in `/data1/bronsonj/summit_pcgc_interval_shape_20261015/`
  (`realscale/`, `gaussian/`, `PLAN_20261015.md`).

**Update, 15 October 2026 (superseded).** The correction was made the CLI default for binary G×E
PCGC (`--binary-crossblock-blocks 0` turns it off). Bronson judged the rule below
too stringent: the corrected variance is reasonably calibrated (ratio 0.89–1.14,
null type-I 4–7%), while the jackknife alone reaches 17–19% type-I error. The
Wald-interval shortfall for variance components is being studied separately.
The Python API still takes `crossblock_blocks` explicitly. Additive PCGC is unchanged.

**Default (13 October).** The predeclared rule (§5) was not met, so the correction was
opt-in (`--binary-crossblock-blocks`, or `crossblock_blocks` in Python). Fit
outputs say which estimator was used and label jackknife-only results as not
calibrated.

## 1. The estimator and its jackknife

Notation follows `src/summit/pcgc/gxe.py`. Participant i has population-scaled
genotypes x_i, contexts φ_i (Q columns, including the intercept), features
ψ_iu = w_i φ_iu with w_i = d_i/s_i (risk sensitivity over the declared liability
SD) and response r_i = z_i. (`pcgc-inverse` uses ψ = φ and r = z/w; nothing
below depends on the choice.) For annotation a with mass M_a and context pair
p = (u, v),

```
G_a(i,l)  = sum_j A_ja x_ij x_lj / M_a
h_p(i,l)  = psi_iu psi_lv + [u != v] psi_iv psi_lu
K_c(i,l)  = h_p(i,l) G_a(i,l),            c = (a, p)
H_cd      = sum_{i != l} K_c(i,l) K_d(i,l)
b_c       = sum_{i != l} r_i r_l K_c(i,l),   theta_hat = H^{-1} b.
```

These are main's normalized equations (`GxEMoments.equations()`). For a SNP
block F let G_a^F use only the target SNPs in F (still divided by the full
M_a), and let K_c^F, H^F (target rows in F, full source kernel) and b^F follow.
Then H = Σ_F H^F, b = Σ_F b^F and the estimating function splits into block
terms:

```
U(theta) = b - H theta = sum_F U_F(theta),   U_F(theta) = b^F - H^F theta,
theta_hat - theta ≈ H^{-1} U(theta).
```

`fit_gxe` deletes target rows: θ_{-F} = (H − H^F)^{-1}(b − b^F), so
θ_{-F} − θ̂ = −(H − H^F)^{-1} U_F(θ̂) exactly. For B blocks of similar
weight (H^F ≈ H/B) and Σ_F U_F(θ̂) = 0,

```
V_JK = (B-1)/B sum_F (theta_-F - mean)(...)'  ≈  B/(B-1) H^{-1} [sum_F U_F(theta_hat) U_F(theta_hat)'] H^{-T}.
```

With U_F(θ̂) ≈ U_F(θ) − (1/B) Σ_G U_G(θ), V_F = Var U_F and
C = Σ_{F≠G} Cov(U_F, U_G),

```
E[V_JK] ≈ H^{-1} [ sum_F V_F − C/(B−1) ] H^{-T},
Var(theta_hat) ≈ H^{-1} [ sum_F V_F + C ] H^{-T}.                       (1)
```

The jackknife is unbiased only if the block estimating equations are
uncorrelated. That is true of the randomness in SNP effects when LD stays
within blocks, which is what the jackknife is designed for. It is not true of
the randomness in participants: every U_F is a sum over the same people.

## 2. Why block equations covary through people

Write U_F as a second-order U-statistic in participants,
U_F(θ) = Σ_{i≠l} k^F(i,l) with k^F_c(i,l) = K^F_c(i,l){r_i r_l − Σ_d θ_d K_d(i,l)}.
Its Hájek projection gives U_F ≈ 2 Σ_i ψ^F(i), with the participant row

```
psi^F_c(i) = sum_{l != i} k^F_c(i,l) = alpha^F_c(i) − [Lambda^F(i) theta]_c,
alpha^F_c(i) = r_i sum_{l != i} K^F_c(i,l) r_l,
Lambda^F_cd(i) = sum_{l != i} K^F_c(i,l) K_d(i,l).                       (2)
```

Two blocks covary through participant i by 4 Cov{ψ^F(i), ψ^G(i)}. Write
r_i = m(g_i) + ε_i with g_i the genetic value, m(g) = E(r | g) ≈ w g and
Var(ε | g) = σ²(g). Then α^F(i) contains ε_i a^F(i) with
a^F(i) = Σ_l m_l K^F(i,l) ≈ c_i g_i^F, where g_i^F is the part of the genetic
value carried by block F. For blocks F ≠ G with independent genotypes and
effects, and Gaussian g,

```
Cov{eps_i a^F(i), eps_i a^G(i)} ∝ E[sigma^2(g) g^F g^G] = 2 kappa h_F h_G,
```

where κ is the curvature of σ²(g) in g. For a homoscedastic trait κ = 0 and
this signal×noise term does not cross blocks. For a liability threshold the
standardized case indicator has Var(z | g) = π(g){1 − π(g)}/{p(1 − p)}, with
π(g) the sampled risk given g. For population sampling and small h its
second-order coefficient is

```
kappa = [ −eta phi(eta)(1 − 2k) − 2 phi(eta)^2 ] / {2 k (1 − k)},   eta = Phi^{-1}(k):
K = 0.5: −0.64,   K = 0.1: 0.66,   K = 0.02: 2.32,   K = 0.005: 3.66.
```

Summed over block pairs, the between-block signal×noise covariance is about
2κh times the within-block signal×noise variance (h the genetic variance).
The jackknife sees only the latter. This is the threshold heteroscedasticity
identified in the 13 October review, which approximated the curvature by
δ ≈ (η² − 1)/2 and predicted a variance ratio near 0.47 for the null
covariance (we observe 0.48–0.50 on main). Two consequences follow that the
review did not state:

- κ changes sign with prevalence. For rare disease the omitted covariance is
  positive and the jackknife is anti-conservative; near K = 0.5 it is negative
  and the jackknife is conservative for this term.
- It is not the only person-level term. ψ^F(i) also contains the person's own
  signal m_i a^F(i), whose between-block covariance Cov(g g^F, g g^G) = h_F h_G
  is nonzero for any trait, including quantitative ones; and shared person-level
  weights and contexts enter every block through Λ^F(i)θ. This is consistent
  with the review's homoscedastic quantitative control, which already had
  variance ratio 0.81 for the null covariance. The signal×signal person term
  is small relative to the SNP-effect (architecture) variance when N/M is
  large, but it is not zero.

The share of these terms in Var(θ̂) grows with the signal in the data,
roughly ĥ·N/M times the squared risk sensitivity, because the signal×noise
and signal×signal person variances grow with it while the noise×noise variance
does not. A purely multiplicative SE correction therefore cannot fix the
jackknife.

## 3. The estimator

### Crossed decomposition, no double counting

Each summand k^F(i,l) belongs to one SNP block and two participants. Two
summands covary if they share a SNP block (SNP effects and LD: architecture),
share a participant (person-level terms above), or both. This is a crossed
design, and the variance is an inclusion–exclusion sum over the two kinds of
sharing:

```
Var(U) = [shared block] + [shared person] − [shared block and person].
```

The jackknife estimates the first bracket (everything within a block,
including the within-block person part). The person-level Hájek covariance
estimates the second. Their overlap is the within-block person part. Hence

```
Var(U) = sum_F V_F + C = E[jackknife] + [person total − person within-block]
       = E[jackknife] + C,      C = 4 sum_i sum_{F != G} Cov{psi^F(i), psi^G(i)}.
```

Only the between-block person covariance is added; the within-block person
part stays with the jackknife, and the SNP-effect part is never estimated
from participant rows. The calibrated covariance is

```
V = V_JK + B/(B−1) H^{-1} C_hat H^{-T} + H^{-1} N_hat H^{-T},                 (3)
C_hat = Gamma_total − sum_F Gamma_F = 4 sum_s sum_{i in s} sum_{F != G} psi~^F(i) psi~^G(i)',
Gamma_X = 4 sum_s sum_{i in s} psi~^X(i) psi~^X(i)',
```

where ψ̃ are rows centered within the case and control strata (fixed case and
control counts, as in main's sampling covariance) evaluated at θ̂, and
ψ^total = Σ_F ψ^F. The factor B/(B − 1) removes the jackknife's own
−C/(B − 1) term in (1).

N̂ is the fitted-risk term. The jackknife holds a same-study probit risk fit
fixed, so none of its variance is in V_JK. Its participant rows are
ν(i) = J(θ̂) IF_i, with IF_i main's probit influence rows
(`sampling.nuisance_derivatives`) and J = ∂U/∂γ = 2 Σ_i α(i)/r_i ⊗ (∂r_i/∂γ +
r_i ∂log w_i/∂γ) − 4 Σ_i (∂log w_i/∂γ) Λ(i)θ. Since all of it is missing,
N̂ = Σ_i [2ψ̃ ν̃' + 2ν̃ ψ̃' + ν̃ ν̃'] is added whole. With supplied risks N̂ = 0.

### Computing the rows without participant pairs

α^F(i) is exact and cheap. With s_jq = Σ_l x_lj ψ_lq r_l,

```
alpha^F_c(i) = r_i/M_a [ psi_iu sum_{j in F} A_ja x_ij s_jv + [u != v] psi_iv sum_{j in F} A_ja x_ij s_ju ]
               − (1 + [u != v]) r_i^2 psi_iu psi_iv sum_{j in F} A_ja x_ij^2 / M_a.
```

This is one more N × (block width) product per genotype block, in the first
existing genotype pass. No extra traversal, participant-pair matrix or
partner sampling is needed.

Λ^F(i) needs Σ_l K^F_c(i,l) K_d(i,l), a sum of products of a context factor
h_p h_r and a genotype factor G^F_a G_b. We use the separable approximation

```
Lambda^F_cd(i) ≈ g^F_{a(c)a(d)} S_{p(c)p(d)}(i),    S_pr(i) = sum_{l != i} h_p(i,l) h_r(i,l),
```

with S computed exactly (O(N Q^4), context only) and the genotype factor g^F
fitted by least squares to the block's own normal-matrix entries,
H^F_{(a,p),(b,r)} ≈ g^F_{ab} S^tot_{pr}, weighting context pairs by S^tot.
(A ratio H^F/S^tot per entry, the first version, is unstable when a context
product averages to nearly zero, as for an unweighted centered exposure in
`pcgc-inverse`; a small dense comparison against exact rows caught this.) The neglected part of
Λ^F(i) is the participant's own genotype deviation within block F (for
example |x_i^F|² − E|x^F|²). For unlinked blocks these deviations are
independent across blocks, so they change within-block terms (which the
jackknife already holds) but not C, up to O(1/N) terms.

The statistics needed for (3) at any θ are small and are accumulated in one
pass: per block Σ_i α̃^F α̃^F' (C×C) and Σ_i α̃^F ⊗ S̃(i) (C×P×P), the same
for the total, Σ_i S̃ ⊗ S̃ and Σ_i S, plus the fitted-risk arrays. Memory
is O(N(C + P²)) during preparation and O(B C (C + P²)) in the artifact.

### Assumptions

1. SNP effects are independent between blocks and LD is contained within
   blocks — the jackknife's own assumption.
2. Participants are independent within the case and control strata (no
   relatedness). Case and control counts are conditioned on.
3. The first-order (Hájek) term carries the between-block person covariance.
   The second-order between-block pair term has mean zero for unlinked
   blocks. Keeping the Hájek product includes it twice; for LD-linked
   neighbouring blocks this adds a small nonnegative amount (conservative).
4. The separable approximation above.
5. The liability model with the declared conditional liability scale equal
   to the realized one (the declared SD is the total liability SD given the
   realized genome). If the realized total liability variance drifts from the
   declared scale, as in the review's "E1" generator, a global
   effect-dependent term appears that no block or person decomposition holds.
6. As for the jackknife, the population metric, liability variance, genotype
   scaling, prevalence and reference probes are held fixed. Fitted probit
   risks are propagated through N̂; supplied risks are fixed.

## 4. Independent checks

Each key step was checked a second way, independently of the streamed code.

**Exact unit oracles** (`tests/test_pcgc_gxe_crossblock.py`, run in CI):

- The streamed per-block statistics equal sums over dense participant-pair
  kernels K_c^F(i,l) for two overlapping annotations, Q = 3, a non-contiguous
  block partition and read widths 1, 4 and 31 (relative error ≤ 1e-11). The
  context sums S equal Σ_l h_p h_r computed from dense context kernels.
- The between-block covariance equals 4 Σ_i Σ_{F≠G} ψ̃^F(i) ψ̃^G(i)' built
  from dense rows, with the genotype factor fitted by an independent
  least-squares solve (inclusion–exclusion identity).
- The risk-fit Jacobian equals central finite differences of b(γ) under
  perturbed probit coefficients (relative 2e-6).
- Streamed preparation (NumPy and native, `pcgc`, `pcgc-inverse`,
  `pcgc-basis`, and `pcgc-ld`) equals the direct accumulator, leaves the
  reference and score rows bit-identical, and `fit_gxe` returns exactly
  V_JK + B/(B−1) H^{-1} C H^{-T}.
- Combined and split artifacts round-trip, and a tampered statistic fails
  its checksum.
- A 200-replicate simulation (N = 4,096, M = 64, K = 0.05, 16 blocks) checks
  that the jackknife variance ratio for the null covariance is below 0.7 and
  the corrected one between 0.8 and 1.25 (about 15 s).

**Dense Monte Carlo oracle** (`scripts/pcgc/oracle_gxe_crossblock.py`).
Fixed genotypes, N = 2,400, M = 120, 15 blocks, K = 0.05 (population
sampling), h = 0.36, Q = 2, 2,000 replicates. Each replicate forms all
participant-pair block kernels, so the block equations U_F(θ_true) are
available exactly and their between-block covariance can be estimated
empirically across replicates, with no Hájek approximation.

| Quantity (equation units) | Ω_00 | Ω_11 | Ω_01 (null) |
|---|---:|---:|---:|
| Empirical Σ_{F≠G} Cov(U_F, U_G) (MCSE) | −3,421 (599) | 194 (89) | 6,916 (921) |
| Mean Hájek estimate, exact rows (MCSE) | −3,370 (61) | 253 (12) | 6,634 (113) |
| Mean Hájek estimate, separable rows | −3,378 | 254 | 6,642 |
| Mean V_JK / predicted H^{-1}[ΣV_F − C/(B−1)]H^{-T} (×10⁴) | 62 / 63 | 61 / 61 | 32 / 32 |
| Variance ratio: jackknife | 0.995 | 0.979 | 0.591 |
| Variance ratio: jackknife + between (exact rows) | 1.011 | 1.029 | 1.008 |
| Variance ratio: jackknife + between (separable rows) | 1.011 | 1.029 | 1.008 |

The participant-row estimate matches the directly measured between-block
covariance within Monte Carlo error for every component, including its sign
(negative for the intercept here). The separable rows change it by less than
0.3%. The jackknife's expectation follows (1), including the −C/(B − 1) term.

**Mechanism predictions.** The sign of κ predicts the direction of the
jackknife error for the null covariance: anti-conservative at K = 0.02 and
0.1 (variance ratios 0.48 and 0.75) and conservative at K = 0.5 (1.10).
Conditioning on case status also induces a negative within-stratum
correlation between blocks' genetic values (selection on the liability). This
is consistent with, but was not separately shown to cause, the conservative
jackknife for the intercept-only (Q = 1) model at K = 0.02 (1.26) and the
negative intercept term in the oracle; the participant rows captured both
(corrected ratios 1.07 and 1.01).

## 5. Calibration grid

### Design

All runs use main's production `fit_gxe` on exact (probe-free) per-SNP
reference rows, so reference-probe noise is excluded. Effects are Gaussian
with prior covariance Ω, which is the truth; residual liability variance is
set from the realized population genetic variance at each context (unit
total liability, so supplied risks are the realized population risks).
Ω = diag(h, h/3) for Q = 2 with a null Ω_01; for Q = 3, Ω_22 = h/4,
Ω_02 = 0.3(h·h/4)^½ and null Ω_01, Ω_12. Contexts take five levels of
E ∈ [−1, 1] (ten for Q = 3), and population risk is Φ(0.3E − t) unless
noted.

- **Fixed** (the review's design): fixed Gaussian genotypes and contexts,
  population sampling (P ≈ K); effects, residuals and outcomes redrawn.
- **Asc.**: genotypes, contexts and liabilities redrawn each replicate with
  P = 0.5 by an exact conditional sampler for Gaussian markers.
- LD: AR(1) ρ = 0.7 within 16-SNP blocks. Overlapping annotations: all SNPs,
  plus a continuous weight on the first 60% (Ω split 0.6/0.4).
- 128 jackknife blocks unless noted; 1,000 replicates unless noted
  (coverage MCSE ≈ 0.7%, variance-ratio MCSE ≈ 0.04).

Cells give variance ratio (mean estimated variance / empirical variance,
delta-method MCSE) and 95% normal-interval coverage in percent.

| Setting (replicates) | Component | Truth | Jackknife | Cross-block |
|---|---|---:|---|---|
| Fixed K=0.5 h=0.36 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.36 | 1.02 (0.05); 95.3 | 0.98 (0.04); 94.7 |
|  | Ω_11 | 0.12 | 0.98 (0.05); 93.7 | 0.96 (0.04); 93.4 |
|  | Ω_01 | 0 | 1.10 (0.05); 95.6 | 0.97 (0.05); 94.0 |
| Fixed K=0.1 h=0.36 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.36 | 1.03 (0.05); 94.1 | 0.99 (0.04); 93.4 |
|  | Ω_11 | 0.12 | 1.00 (0.04); 94.3 | 1.03 (0.04); 94.7 |
|  | Ω_01 | 0 | 0.75 (0.03); 90.8 | 1.08 (0.05); 95.9 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.36 | 0.88 (0.04); 92.2 | 0.97 (0.04); 92.9 |
|  | Ω_11 | 0.12 | 0.89 (0.04); 93.8 | 0.95 (0.04); 94.5 |
|  | Ω_01 | 0 | 0.48 (0.02); 82.4 | 0.98 (0.04); 94.3 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; replicates 1000–2999 (2000) | Ω_00 | 0.36 | 0.90 (0.03); 92.2 | 0.99 (0.03); 93.2 |
|  | Ω_11 | 0.12 | 0.96 (0.03); 94.2 | 1.02 (0.03); 94.7 |
|  | Ω_01 | 0 | 0.50 (0.02); 83.8 | 1.02 (0.03); 95.0 |
| Fixed K=0.5 h=0.1 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.1 | 1.03 (0.04); 96.0 | 1.01 (0.04); 95.4 |
|  | Ω_11 | 0.0333 | 0.99 (0.04); 94.8 | 0.99 (0.04); 94.8 |
|  | Ω_01 | 0 | 1.06 (0.05); 95.4 | 1.01 (0.05); 95.1 |
| Fixed K=0.1 h=0.1 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.1 | 0.98 (0.04); 94.1 | 0.96 (0.04); 93.8 |
|  | Ω_11 | 0.0333 | 1.03 (0.05); 94.8 | 1.03 (0.05); 94.8 |
|  | Ω_01 | 0 | 0.97 (0.05); 94.7 | 1.08 (0.05); 96.0 |
| Fixed K=0.02 h=0.1 Q=2 N=32k M=1024 (1000) | Ω_00 | 0.1 | 0.93 (0.04); 93.7 | 0.95 (0.04); 93.5 |
|  | Ω_11 | 0.0333 | 1.00 (0.05); 94.4 | 1.01 (0.05); 94.2 |
|  | Ω_01 | 0 | 0.89 (0.04); 92.6 | 1.02 (0.05); 95.3 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; risk slope 0.6 (1000) | Ω_00 | 0.36 | 0.61 (0.03); 85.7 | 0.99 (0.05); 92.9 |
|  | Ω_11 | 0.12 | 0.98 (0.05); 93.9 | 1.04 (0.05); 94.1 |
|  | Ω_01 | 0 | 0.53 (0.02); 85.7 | 1.00 (0.04); 93.8 |
| Fixed K=0.02 h=0.36 Q=1 N=32k M=1024 (1000) | Ω_00 | 0.36 | 1.26 (0.06); 96.6 | 1.07 (0.05); 94.7 |
| Fixed K=0.02 h=0.36 Q=3 N=32k M=1024 (1000) | Ω_00 | 0.36 | 0.93 (0.04); 93.7 | 1.06 (0.05); 94.9 |
|  | Ω_11 | 0.12 | 0.92 (0.04); 94.0 | 0.97 (0.04); 94.6 |
|  | Ω_22 | 0.09 | 1.00 (0.05); 95.0 | 1.03 (0.05); 95.1 |
|  | Ω_01 | 0 | 0.49 (0.02); 83.2 | 1.03 (0.05); 94.4 |
|  | Ω_02 | 0.054 | 0.51 (0.02); 83.2 | 1.07 (0.05); 95.9 |
|  | Ω_12 | 0 | 0.72 (0.03); 90.3 | 0.92 (0.04); 94.1 |
| Fixed K=0.02 h=0.36 Q=2 N=8k M=256 (1000) | Ω_00 | 0.36 | 0.91 (0.04); 91.6 | 1.01 (0.04); 92.7 |
|  | Ω_11 | 0.12 | 0.89 (0.04); 91.3 | 0.95 (0.04); 91.9 |
|  | Ω_01 | 0 | 0.50 (0.02); 83.4 | 1.00 (0.05); 93.2 |
| Fixed K=0.02 h=0.36 Q=2 N=64k M=2048 (800) | Ω_00 | 0.36 | 0.86 (0.04); 92.8 | 0.95 (0.05); 94.2 |
|  | Ω_11 | 0.12 | 0.90 (0.05); 94.0 | 0.96 (0.05); 94.4 |
|  | Ω_01 | 0 | 0.50 (0.03); 83.4 | 1.03 (0.05); 94.9 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=256 (1000) | Ω_00 | 0.36 | 0.93 (0.04); 92.1 | 0.98 (0.05); 92.9 |
|  | Ω_11 | 0.12 | 0.90 (0.04); 91.7 | 0.99 (0.04); 92.9 |
|  | Ω_01 | 0 | 0.49 (0.02); 83.0 | 1.04 (0.05); 95.0 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=4096 (800) | Ω_00 | 0.36 | 0.98 (0.05); 95.5 | 1.07 (0.05); 95.8 |
|  | Ω_11 | 0.12 | 0.99 (0.04); 95.4 | 1.02 (0.05); 95.1 |
|  | Ω_01 | 0 | 0.65 (0.03); 88.1 | 0.98 (0.05); 93.8 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; fitted risk (1000) | Ω_00 | 0.36 | 1.19 (0.06); 95.9 | 1.00 (0.05); 93.5 |
|  | Ω_11 | 0.12 | 0.90 (0.04); 93.9 | 0.94 (0.04); 93.9 |
|  | Ω_01 | 0 | 1.13 (0.05); 95.4 | 0.96 (0.04); 93.4 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; inverse (1000) | Ω_00 | 0.36 | 0.85 (0.04); 91.7 | 0.95 (0.04); 93.0 |
|  | Ω_11 | 0.12 | 0.85 (0.04); 93.0 | 0.93 (0.04); 93.4 |
|  | Ω_01 | 0 | 0.44 (0.02); 80.6 | 0.95 (0.04); 94.0 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; LD, 64 blocks, 2 overlapping annot. (1000) | A0 Ω_00 | 0.216 | 0.98 (0.05); 92.9 | 0.99 (0.05); 93.1 |
|  | A0 Ω_11 | 0.072 | 1.04 (0.05); 93.5 | 1.05 (0.05); 93.5 |
|  | A0 Ω_01 | 0 | 0.80 (0.04); 91.9 | 0.95 (0.04); 93.9 |
|  | A1 Ω_00 | 0.144 | 0.99 (0.04); 92.6 | 1.00 (0.04); 92.7 |
|  | A1 Ω_11 | 0.048 | 0.97 (0.04); 93.6 | 0.98 (0.04); 93.8 |
|  | A1 Ω_01 | 0 | 0.92 (0.04); 92.7 | 1.00 (0.05); 93.7 |
| Fixed K=0.02 h=0.36 Q=2 N=32k M=1024; LD, 128 blocks, 2 overlapping annot. (1000) | A0 Ω_00 | 0.216 | 0.81 (0.04); 90.2 | 0.88 (0.04); 91.4 |
|  | A0 Ω_11 | 0.072 | 0.90 (0.04); 92.1 | 1.05 (0.05); 93.9 |
|  | A0 Ω_01 | 0 | 0.69 (0.03); 89.3 | 0.91 (0.04); 93.4 |
|  | A1 Ω_00 | 0.144 | 0.82 (0.04); 90.8 | 0.86 (0.04); 91.4 |
|  | A1 Ω_11 | 0.048 | 0.84 (0.04); 91.8 | 0.95 (0.04); 93.8 |
|  | A1 Ω_01 | 0 | 0.78 (0.04); 90.5 | 0.93 (0.04); 92.7 |
| Asc. K=0.1 h=0.36 Q=2 N=16k M=512 (1000) | Ω_00 | 0.36 | 1.00 (0.04); 94.5 | 0.98 (0.04); 94.1 |
|  | Ω_11 | 0.12 | 1.04 (0.05); 94.7 | 1.04 (0.05); 94.6 |
|  | Ω_01 | 0 | 1.01 (0.05); 94.3 | 1.05 (0.05); 94.7 |
| Asc. K=0.02 h=0.36 Q=2 N=16k M=512 (1000) | Ω_00 | 0.36 | 0.99 (0.04); 94.5 | 0.99 (0.04); 94.5 |
|  | Ω_11 | 0.12 | 1.02 (0.05); 93.3 | 1.05 (0.05); 93.8 |
|  | Ω_01 | 0 | 0.86 (0.04); 92.4 | 1.02 (0.05); 94.9 |
| Asc. K=0.02 h=0.36 Q=2 N=16k M=512; LD, 32 blocks, 2 overlapping annot., fitted risk (1000) | A0 Ω_00 | 0.216 | 1.14 (0.05); 92.6 | 1.14 (0.05); 92.6 |
|  | A0 Ω_11 | 0.072 | 1.06 (0.05); 91.5 | 1.06 (0.05); 91.5 |
|  | A0 Ω_01 | 0 | 0.99 (0.05); 93.4 | 0.99 (0.05); 93.3 |
|  | A1 Ω_00 | 0.144 | 1.07 (0.05); 92.1 | 1.07 (0.05); 92.1 |
|  | A1 Ω_11 | 0.048 | 1.10 (0.05); 90.9 | 1.10 (0.05); 90.8 |
|  | A1 Ω_01 | 0 | 1.00 (0.05); 93.6 | 0.99 (0.05); 93.5 |

Main's experimental sampling SEs on the same replicates (128 partners, 32
architecture probes per family; 600 replicates). "Sampling" targets a fixed
genome and omits SNP-effect variation by design:

| Setting (replicates) | Component | Jackknife | Sampling | Sampling + architecture | Cross-block |
|---|---|---|---|---|---|
| Fixed K=0.02 h=0.36 Q=2 N=8k M=256 (600) | Ω_00 | 0.89 (0.05); 91.8 | 0.72 (0.04); 88.2 | 0.98 (0.05); 92.8 | 0.99 (0.05); 92.8 |
|  | Ω_11 | 0.91 (0.06); 91.3 | 0.95 (0.06); 92.7 | 0.98 (0.06); 92.8 | 0.95 (0.06); 91.5 |
|  | Ω_01 | 0.50 (0.03); 83.0 | 0.95 (0.06); 92.0 | 0.99 (0.06); 92.8 | 1.00 (0.06); 92.7 |
| Asc. K=0.02 h=0.36 Q=2 N=8k M=256 (600) | Ω_00 | 0.89 (0.05); 92.5 | 0.10 (0.01); 45.8 | 0.94 (0.05); 93.3 | 0.89 (0.05); 92.5 |
|  | Ω_11 | 0.97 (0.06); 92.7 | 0.49 (0.03); 82.7 | 1.03 (0.06); 93.7 | 1.00 (0.06); 93.2 |
|  | Ω_01 | 0.84 (0.05); 93.0 | 0.43 (0.02); 77.0 | 1.03 (0.06); 96.2 | 0.99 (0.05); 95.3 |

### Findings

- **The deficit is real, not small-sample noise.** At N/M = 32 the jackknife
  variance ratio of the null Ω_01 is 0.50, 0.48–0.50 and 0.50 at N = 8,192,
  32,768 and 65,536. It shrinks with N/M (0.65 at N/M = 8) and with weaker
  signal (0.89 at h = 0.1), and it is larger with stronger risk
  heterogeneity (intercept 0.61 with risk slope 0.6, close to the review's
  0.49). `pcgc-inverse` behaves like `pcgc`.
- **Its sign varies.** The jackknife is conservative at K = 0.5 (Ω_01 1.10),
  for the intercept-only model (1.26) and with fitted risks (1.13–1.19). With
  fitted risks the omitted risk-fit term is negative and, for Ω_01, larger
  than the omitted between-block term (−1.36 and +1.19 times the empirical
  variance).
- **The corrected variance is calibrated** in every setting that respects the
  jackknife's LD assumption (ratios 0.89–1.14, mean 1.00; 70 of 73 cells
  within 2 MCSE of 1). Null components have type-I error 4.0–7.3%
  (mean 5.7%).
- **Coverage of variance components stays 1–3 points low** where the SE is
  itself noisy and correlated with the estimate: small M (M = 256: 91.5–92.9%),
  few blocks (32 blocks: 90.8–92.6%, identical for the jackknife) and the
  rare-disease intercept at M = 1,024 (92.9–93.2%). At M = 4,096 and at
  N = 65,536 coverage is 94.2–95.8%. Student-t quantiles with B − 1 degrees of
  freedom recover only about 0.2 points.
- **With LD crossing block boundaries** (8-SNP blocks in 16-SNP LD blocks)
  both estimators undercover (corrected 91.4–93.9%); aligning blocks to LD
  restores the variance (0.95–1.05).
- Population heritability SEs (metric held fixed), recorded in the 11 later
  settings: corrected variance ratio 0.90–1.08, coverage 90.7–95.2%
  (jackknife 0.68–1.09, 87.0–95.7%).

### Default decision

The rule below was written after the first four settings (K = 0.02, 0.1,
0.5 at h = 0.36 and K = 0.02 at h = 0.1) had been seen and before the other
19 result sets were run (`decision_rule.txt` in the output directory):

```
Declared 2026-10-13 20:22 PDT, after viewing 5 of the planned grid settings
(F_K02_h36_q2, F_K10_h36_q2, F_K50_h36_q2, F_K02_h10_q2 with JK and the
cross-block estimator; the superseded ratio-row runs for K02/K10).

The cross-block estimator becomes the CLI default only if, over every
completed grid setting and every component (1,000 replicates unless noted):
 (a) variance ratio within max(0.10, 3 MCSE) of 1;
 (b) coverage >= 0.95 - 2.58 MCSE in every cell, and mean coverage over all
     cells within 1 percentage point of 0.95;
 (c) null components: rejection <= 0.05 + 2.58 MCSE;
 (d) never below the jackknife's coverage by more than 2.58 MCSE of the
     difference in any cell.
Otherwise it is opt-in, with the failing cells listed.
```

It fails in 21 of 79 cells. Rule (b) fails in 16 cells: variance
components (including the 2,000-replicate extension, 93.2%), the
LD-misaligned design and the 32-block design; in each of them the jackknife's
coverage is lower or within one replicate. Rule (d) fails in 5 cells where
the correction removed a conservative jackknife bias (K = 0.5 Ω_01, Q = 1,
fitted risks, K = 0.1 intercept) and coverage dropped to 93.4–94.7%. Rules (a) and (c) fail only in
the LD-misaligned design. The estimator is therefore opt-in. For rare
diseases with large ĥ·N/M and supplied risks, where the jackknife's null
type-I error reached 17–19%, the opt-in correction is recommended in the
documentation.

## 6. Implementation

Code is in `src/summit/pcgc/crossblock.py`, with hooks in `gxe.py`,
`gxe_io.py`, `gxe_cli.py` and `cli.py`.

- **Preparation.** `prepare_gxe_moments` and `prepare_gxe_external` accept
  `crossblock_blocks` (an inference block count, giving the same equal-count
  contiguous partition as binary `--njack`, or explicit SNP block IDs).
  `CrossBlockOperator` wraps the genotype operator and passes each block of
  the first existing traversal to `CrossBlockAccumulator`. It uses SUMMIT's
  protected NN/TN products (NumPy for `native=False`). Fitted probit risks
  require their covariates (`risk_covariates`), as for the sampling path;
  the influence rows come from `sampling.nuisance_derivatives`. The workspace
  is admitted before any genotype I/O, and `plan_gxe_reference` accepts
  `crossblock_blocks`.
- **Artifacts.** `GxEMoments.crossblock_moments` holds the statistics. Combined
  and split files carry them as checksummed features
  `person_crossblock_covariance` and `person_crossblock_risk_fit`, in the
  summary-statistics file (they are trait-specific).
- **Fit.** `fit_gxe` always computes the SNP-block jackknife. If the moments
  carry cross-block statistics and the inference blocks are the same
  partition (labels may differ), it adds the two terms in (3). Otherwise it
  keeps the jackknife and says why in `crossblock_status`. If the corrected
  covariance has a nonpositive variance (a very small sample), it keeps the
  jackknife and reports `not_applied_nonpositive_calibrated_variance`.
- **Outputs.** `standard_errors`/`covariance` are the primary result;
  `snp_block_standard_errors`/`snp_block_covariance` are always the jackknife;
  `person_crossblock_covariance` and `risk_fit_covariance` are the added
  terms; `uncertainty_method`, `uncertainty_calibration_status`,
  `uncertainty_calibration_note` and `crossblock_status` state what was done.
  Every jackknife-only binary G×E result is labelled
  `not_calibrated_snp_block_jackknife_omits_between_block_person_covariance` (until
  17 October 2026; since then `calibrated_in_simulation_for_N_over_M_up_to_1`, see the
  update at the top).
  If main's experimental sampling moments are also present, the cross-block
  result stays primary and the sampling result moves to `individual_sampling`;
  without the cross-block statistics the sampling path behaves as before.
- **CLI.** `--binary-crossblock-blocks N` at preparation records the rows; the
  fit then uses the prepared blocks unless `--njack` is given. The option is
  off by default (see §5). It is rejected for additive (non-contextual) runs.

The added work is one N × (block width) product for scores, one for the
block rows and one for genotype squares per genotype block, all in the first
pass, plus O(N C P²) per closed SNP block. No participant-pair matrix,
partner sampling or extra genotype traversal is used. In a paired benchmark
(`scripts/pcgc/benchmark_crossblock.py`: N = 20,000, M = 4,000, Q = 3, two
annotations, 64 probes, two native threads, alternating order, host shared
with the simulation workers) median preparation time rose from 24.7 to 31.0 s
and the planned workspace from 677 to 759 MB. With the default 256 probes
the relative cost is smaller. The fit adds milliseconds.

## 7. Residual risks

- **Wald coverage of variance components.** The corrected variance is close
  to the empirical variance across the grid, but 95% normal intervals for
  diagonal Ω entries covered as little as 91.5% (90.8% with 32 blocks). The
  cause is that the
  SE estimate is strongly correlated with the estimate itself (correlation
  0.4–0.8; misses concentrate in the lower tail). Coverage with the empirical
  SD in place of the estimated SE was 94.2–96.1% in the same runs. This affects
  the jackknife equally wherever the jackknife variance is unbiased. It is
  strongest where SNP-effect (architecture) variance dominates, i.e. small M
  relative to N; real arrays have far larger M. Test-inversion (score-type)
  intervals that let the variance depend on the candidate value were not
  implemented.
- **Fitted risks.** With a probit risk fit on the context, the jackknife was
  conservative for the null covariance (the omitted risk-fit term is negative
  and larger than the omitted between-block term), and the corrected intervals
  covered 93.4–93.9%. The two added terms nearly cancel for that component,
  which adds noise to its SE.
- **LD across block boundaries.** Like the jackknife, the correction assumes
  LD does not cross blocks. With 8-SNP blocks inside 16-SNP AR(1) LD blocks,
  both undercovered (corrected 91.4–93.9%); with blocks aligned to LD both
  recovered (see the table).
- **Small samples and few blocks.** The between-block estimate uses θ̂ in the
  participant rows and has O(1/√N) relative noise. In tiny examples (N≈150)
  it can make a variance nonpositive; the fit then falls back to the jackknife
  and says so.
- **Untested regimes.** Real-array M (≈450,000, long-range LD), relatedness,
  population structure with genotype-PC adjustment, heterogeneous liability
  SDs, external-LD (`pcgc-ld`) calibration and `pcgc-basis` were not simulated;
  their code paths are covered by unit tests only. `pcgc-inverse` was run in
  one setting. Population-heritability SEs hold the population metric and
  liability variance fixed.
- **Additive-only path.** `--binary-method pcgc` without contexts uses separate
  code (`moments.py`) and its SNP-block jackknife has the same omission. Through
  the Python G×E API with an intercept-only context, the correction applies
  (Q = 1 row of the table); the additive CLI path was not changed.
- **Scale drift.** If the declared liability scale differs from the realized
  total liability variance (the review's E1 generator), a global term appears
  that neither the jackknife nor the participant rows contain.

## Reproduction

All drivers refuse to overwrite outputs. Runs used the summit Conda Python,
the portable OpenBLAS extension built in this worktree
(`build/portable`), `taskset -c 32-39`, eight fork workers with one BLAS
thread each (`threadpoolctl`), and `scripts/epistasis/checkout_python.py`:

```bash
W=/home/bronsonj/SUMMIT-pcgc-binary-intervals-20261013
taskset -c 32-39 env LD_LIBRARY_PATH=$CONDA_PREFIX/lib SUMMIT_NATIVE_DIR=$W/build/portable \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python $W/scripts/epistasis/checkout_python.py \
  scripts.pcgc.calibrate_gxe_intervals --out NEW.json --K 0.02 --h 0.36 --q 2 \
  --n 32768 --m 1024 --blocks 128 --replicates 1000 --workers 8
# main's sampling SEs on the same replicates:
#   --sampling-partners 128 --architecture-probes 32
# design variants: --design random --case-fraction .5, --ld ar1, --annotations overlap,
#   --risk fitted, --method pcgc-inverse, --risk-slope, --replicate-offset
python $W/scripts/epistasis/checkout_python.py scripts.pcgc.oracle_gxe_crossblock --out NEW_ORACLE.json
python scripts/pcgc/report_gxe_intervals.py RESULT_DIRECTORY
```

Each run writes a JSON summary and an NPZ of per-replicate estimates and
variance terms. Outputs: `grid/` (this report), `superseded_ratio_rows/` (two
early runs with the unstable per-entry ratio rows; their jackknife columns
are valid and identical to the reruns). The default rule in §5 is kept as
`decision_rule.txt` in the output directory. The grid, sampling comparison
and oracle took about five hours of wall time on eight cores.
