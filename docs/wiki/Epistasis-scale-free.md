# Scale-free epistasis

Whether a genotype-by-genotype interaction exists can depend on the phenotype
scale. A locus that only shifts and stretches a latent liability produces
interaction on most measured scales, and a log or rank transform can create
or remove it. `summit epistasis scale-free` tests a focal context (a SNP, or
any genotype-defined classes such as a set-score tertile) against the
polygenic background. Its statistics are null under scale-only mechanisms,
either exactly or to first order. It also reports the FAME-equivalent
statistic on the conventional pooled rank scale for comparison.

## Scale nulls

Let K be the focal class and U the background liability.

| Null | Mechanism | Detectable departure |
|---|---|---|
| Affine modifier | Y = h(a_K + s_K U), U independent of K | specific epistasis: the background's share or pattern changes with K |
| Proportional background | background effects scale by a common factor in each class | pattern change (ρ < 1) or selective change across annotations |

Under the affine-modifier null, the phenotype ranked within each focal class
(and stratum, e.g. sex) has the same joint law with the background in every
class. Any statistic computed on the **within-class rank** scale is then
null, whatever the unknown h, a_K and s_K. The only departures are of order
1/n_k: rank-normal scores of a small class are slightly attenuated (their
correlation with the latent value is 0.98 at n_k = 150 and 0.9995 at 7,350),
and centring within the sample. On the **pooled rank**
scale (FAME's convention) the same null produces apparent marginal
epistasis. The difference between the two scales is the global (scale-type)
part.

## Statistics

Class-score U-statistics use only pairs of distinct individuals. They need no
LD model, trace or noise model. For background SNP j, let
S_j(c, g) be the sum of x_ij y_i over individuals i in class c and group g,
normalized by the class size in that group. For a statistic m = (c1, c2),

```math
\theta_m=\frac{1}{G(G-1)}\sum_{g\ne h}\frac{1}{W}\sum_j w_j S_j(c_1,g)S_j(c_2,h),
```

which estimates the background mean of E[x_j y | c1] E[x_j y | c2].

| Statistic | Definition | Null | Scale behaviour |
|---|---|---|---|
| `dosage` | θ(gy, gy), gy = standardized class value × y | 0, one-sided | equals FAME's marginal-epistasis target; scale-dependent except on within-class ranks |
| `rho` | θ(lo, hi) / √(θ(lo, lo) θ(hi, hi)) | 1 | pattern; first-order scale-free |
| `log_amp` | log θ(hi, hi) / θ(lo, lo) | 0 | genetic-share ratio; null on within-class ranks |
| `delta2` | θ(lo, lo) + θ(hi, hi) − 2θ(lo, hi) | 0, one-sided | FAME-like contrast |
| `sel_lam[A]` | log λ_A − log λ_rest, λ = θ(lo, hi)/θ(lo, lo) | 0 | selective across annotations; first-order scale-free |
| `sel_amp[A]` | log α_A − log α_rest, α = θ(hi, hi)/θ(lo, lo) | 0 | selective genetic share |
| `dosage[A]`, `dosage_diff[A]` | dosage in A, and A minus the complement | 0 | annotation-specific marginal epistasis |
| `additive`, `additive[A]` | θ(y, y) (with `additive_scales`) | none | additive background signal θ_y, descriptive |
| `relative_dosage`, `relative_dosage[A]` | θ(gy, gy) / θ(y, y) | 0, one-sided | dosage relative to the additive signal; same null as `dosage` |
| `global_excess` | θ(gy, gy) − R θ(y, y) | none | dosage beyond the first-order prediction R θ_y of an affine focal action (R from class spreads); descriptive |

Scores are class sums of x_ij y_i divided by the effective class size
Σ(1 − h_ii), with h_ii the leverage of the covariate projection, so that their
expectation is E[x_j y | c] whatever the number of fitted covariate parameters
per class.

Each statistic has two standard errors valid under the null and under
alternatives:

- a group-level Hoeffding SE, which treats the G random individual groups as
  i.i.d. units;
- a delete-one SNP-block jackknife SE (chromosome-respecting blocks).

The degenerate statistics (`dosage`, `dosage[A]`, `delta2`) also have a
null-restricted SE, and their z score uses it. Under the null their group
kernel T_ab is a product of two group scores with mean zero, so the T_ab
(a ≠ b) are uncorrelated with mean zero and

    Var0(θ̂) = 2 E[T_ab²] / (G (G − 1))

exactly, whatever the LD, the phenotype distribution or the group sizes.
E[T_ab²] is estimated by the mean of T_ab² over a ≠ b. The full Hoeffding SE
also estimates a first-order term that is zero under the null, and that noise
makes it heavy-tailed (with G = 30 its variance estimate was negative in 15-20%
of simulated null replicates). On 666 real UK Biobank dosage columns the
null-restricted SE agreed with the analytic value 2v²τ₂/(n(n − 1)) to 0.4%
(interquartile range of the ratio 1.000 to 1.008), while the ratio of the full
Hoeffding SE to it had interquartile range 0.91 to 1.07.
Under alternatives it omits the first-order term, so intervals should use the
Hoeffding or jackknife SE. The other statistics are not degenerate at their
null, and their z score uses the larger of the Hoeffding and jackknife SEs.
`rho` = 1 lies on the Cauchy-Schwarz boundary, where the first-order term of
the delta method vanishes, so its Hoeffding SE alone is unreliable; the
max-SE z for `rho` was nominal in every null scenario we simulated.
`relative_dosage` uses the null-restricted SE, like `dosage`. G = 200 is the
qualified default.

**p-values of the dosage-type tests come from sign flips** (`"null_p":
"signflip"`, default). Under the null the score vectors of different groups
are independent with mean zero and, as sums over about n/G people, symmetric,
so θ(ε) = Σ_{a≠b} ε_a ε_b T_ab / (G(G − 1)) with random signs ε has the null
distribution of θ̂ given the scores. Its variance is the null-restricted
variance, but the distribution is skewed, with skewness
2√2 Σν³ / (Σν²)^{3/2} over the eigenvalues ν of T. Over all trans variants
the skewness is about 0.02 and the normal tail is accurate. For an annotation
set confined to a few hundred loci it is 0.1-0.3 in UK Biobank, and near z = 4
the normal tail was 3-20 times too small (for example normal p 3.0e-5 against
sign-flip p 1.0e-4). The flips are adaptive (20,000, then 2,000,000 when
p < 0.002) and seeded from the kernel, so results are reproducible. `p_normal`
keeps the normal-tail value and `p_method` says which one `p` is; z is
unchanged. `"null_p": "normal"` skips the flips.

The background excludes the chromosome of a variant context and the
chromosomes listed for a file context (validated against the genotype
source). U-statistics need the context to be independent of the background
SNPs, which fails for SNPs in LD with the context. An analysis whose
background is empty is omitted from `results.tsv`; an annotation with no
weight left for an analysis is dropped from that analysis only.

### What each statistic detects

| Change across focal classes | `dosage` | `rho` | `log_amp` | `delta2` |
|---|---|---|---|---|
| Background effects change in a variant-specific way (FAME model) | yes | yes | weak | yes |
| All background effects scale by the same factor (amplification) | weak (quadratic in the change) | no | yes | weak |
| The focal variant changes only the noise variance | second order, small | no | yes | second order, small |
| Shift and stretch of the whole liability (affine modifier) | no on within-class ranks | no | no on within-class ranks | no on within-class ranks |

When every standardized background effect in class k equals a_k c_j (a
change common to all variants), the dosage statistic equals Cov(g, a)² θ_y,
which is quadratic in the change, whereas `log_amp` responds to it at first
order. `rho` responds only to changes that differ between variants.

## Covariate adjustment

Outcomes are residualized on the covariates on every scale.
`covariate_slopes` sets how:

- `common` (default): one set of slopes per stratum (for example per sex),
  with an intercept for every focal class within the stratum;
- `class`: separate slopes within every focal class × stratum.

Class-specific slopes also absorb an interaction between the focal context
and a covariate, for example a focal variant whose effect differs along an
ancestry axis. Their cost is a bias. For a covariate direction u along which
the background relatedness has u'Au = λ > 1 (principal components, assessment
centre), the in-sample fit makes the residuals of related people in the same
class slightly negatively correlated, and same-class pairs all carry weight
g_k² > 0. The null z of the dosage and contrast statistics then shifts by
about −Σ_k (λ_k − 1) / (n √(2τ₂)) whatever the trait or focal variant. With
common slopes the pair weights g_i g_i' take both signs and the shift is of
order 1/n. In a structured simulation (two subpopulations, five in-sample
principal components) the predicted shift was −0.42, the observed shift −0.43
with class slopes and −0.07 and +0.01 with common slopes. In UK Biobank, with
20 principal components and assessment centre, class slopes gave a null mean
z of −0.135.

Use `common` for discovery and repeat significant findings with `class`,
which is conservative by the known shift. A finding that disappears under
`class` may reflect a context × covariate interaction rather than epistasis.
`run.json` records the setting used.

Residualization projects onto the centred, unit-length covariates of each
stratum through the eigendecomposition of their correlation matrix, keeping
eigenvalues above 10⁻¹⁰ of the largest; exactly collinear and constant
columns drop out.

## Calibration and power in UK Biobank

Checked with the default settings (within-class ranks, common slopes,
null-restricted SE, G = 200) on 270,510 unrelated white British participants
and 448,562 array SNPs:

- **Null calibration.** 6,400 frequency-matched array variants without
  association to their trait (800 for each of height, BMI, triglycerides, LDL
  cholesterol, bilirubin, Lp(a), testosterone and platelet count): z mean
  +0.023, SD 0.999, 5.3% below p = 0.05, 60 below 0.01 (64 expected) and 9 below
  0.001 (6.4 expected). The full Hoeffding SE gave a maximum z of 4.6; the
  null-restricted SE 3.6. These are dosage tests over the whole trans
  background, where the sign-flip and normal p-values agree; annotation-set
  tests need the sign-flip p-values (see above).
- **Power.** Simulated phenotypes on the same genotypes follow
  z = κ² σ²_γ n √τ₂ / (√2 v), with σ²_γ FAME's interaction variance and κ ≤ 1
  a within-class standardization factor (κ² ≈ 0.82-0.99), to within 4%. At the
  Bonferroni threshold for 333 tests, power was 0.70-0.80 for σ²_γ = 0.008 and
  1.00 for 0.016, for focal allele frequencies from 0.03 to 0.35.

## One genotype pass

Each block of background SNPs is decoded once. It is standardized per
individual group with the shared native standardizer, and group scores are
formed by SUMMIT's protected TN product; the arithmetic equals that of a
single ungrouped product. The G × G kernels are many small products. They use
batched BLAS on scores augmented with a sum-over-groups row, so every product
carries its own row and column checksums. Mismatching kernels are recomputed,
and a repeated mismatch stops the run. Each kernel is far too small to share
across BLAS threads, so the kernel step runs one-thread BLAS in a pool of
`--num-threads` workers; each task owns one batch of columns across all
jackknife segments of a block. Nothing of size M × G × q is stored.

Main-product cost is about 2 N M q flops, where q is the number of outcome
columns: one dosage column per analysis and scale, two pattern columns per
analysis and pattern scale, and one additive column per analysis and additive
scale. The plan holds 8 N q bytes and is checked against `--memory-gib`
before it is built. Kernel cost is about 2 G² M times the number
of statistics and annotations. Kernel memory is 8 G² q bytes per annotation.

For planning: with N = 270,510, M = 448,562 and G = 200, building the plan
takes about 0.5 s per analysis, and the pass took about 2.2 h for 3,200
columns on 16 threads (veena) and 5 h on 8 threads (tabla); the score
product dominates. Split very large scans into passes with the same rows,
groups and seed, and concatenate their `results.tsv`.

## Manifest

```json
{
  "kind": "summit.epistasis.scale_free",
  "schema_version": 1,
  "genotypes": {"geno": "cohort.bed"},
  "samples": "analysis_samples.tsv",
  "phenotypes": {"file": "phenotypes.tsv", "columns": ["TG", "ALT"], "positive": ["TG", "ALT"]},
  "covariates": {"file": "covariates.tsv", "columns": ["age", "sex", "PC1"], "categorical": ["centre"]},
  "strata": {"file": "phenotypes.tsv", "column": "sex"},
  "contexts": [
    {"name": "rs964184", "variant": "rs964184", "min_minor_homozygotes": 300},
    {"name": "liver_score", "file": "score_tertiles.tsv", "column": "tertile",
     "values": {"0": 0, "1": 1, "2": 2}, "low": [0], "high": [2], "exclude_chromosomes": ["2", "19", "22"]}
  ],
  "analyses": [["rs964184", "TG"], ["liver_score", "ALT"]],
  "background": {"annotations": {"file": "annotations.tsv", "columns": ["liver_H3K27ac"]},
                 "jackknife_blocks": 200},
  "scales": ["pooled_rank", "within_rank"],
  "pattern_scales": ["within_rank"],
  "groups": 200,
  "seed": 20261011
}
```

- **Variant contexts** code minor-allele counts; the minor allele and the
  number of minor homozygotes are taken over all pass rows. With fewer than
  `min_minor_homozygotes` (default 300), minor homozygotes join the carriers;
  `"coding": "carrier"` forces carrier coding. Genotypes must be hard calls.
- **File contexts** supply non-negative integer classes, their numeric values
  for the dosage statistic, disjoint low and high classes for the pattern
  contrast, and the chromosomes to exclude.
- **Annotation tables** have a `SNP` column and weights in [0, 1]; the
  complement is derived.
- **Covariates** are partialled on every scale, with slopes common to the
  focal classes of a stratum (`"covariate_slopes": "common"`, default) or
  specific to each class × stratum (`"class"`); see
  [Covariate adjustment](#covariate-adjustment). Laboratory traits listed
  under `positive` drop non-positive values. A missing stratum value (blank,
  NA, NaN, N/A) removes the person from every analysis.
- **Other keys:** `additive_scales` adds the plain residual column (for
  `additive`, `relative_dosage` and `global_excess`); `genotype_scale`
  `"empirical"` (default) or `"hwe"` standardizes background genotypes with
  observed or Hardy-Weinberg variances; `save_kernels` keeps the kernels;
  `null_p` `"signflip"` (default) or `"normal"` chooses the p-values of the
  dosage-type tests.

Run with explicit threads and memory:

```bash
summit epistasis scale-free scale_free.json --out results/ --num-threads 8 --memory-gib 32
```

Outputs:

| File | Contents |
|---|---|
| `results.tsv` | one row per analysis, scale and statistic: estimate, Hoeffding, jackknife and null-restricted SEs, the test SE (`se_rule` null_restricted or max), z, p (sign-flip for dosage-type tests, `p_method`), `p_normal` and the number of flips |
| `run.json` | manifest, contexts, class sizes, covariate slopes, genotype source and scale identities, timings, checksum repairs |
| `kernels.npz` | optional (`"save_kernels": true`): G × G kernels and jackknife sums, for recomputing statistics without genotypes |

## Pairwise rank tests

`summit.epistasis.scale_free.rank_pairs.pair_battery` tests one pair of
minor-allele counts. It applies the rank-based and scale-exact tests
summarized in the module docstring:

- cumulative-probability models with four links;
- a location-scale CPM;
- stratified rank-normal tests in both directions;
- Mann-Whitney contrasts;
- the robustness radius R* with a sup-t lower bound.

Fixed-scale HC3 product tests are included for comparison. An interaction
that survives every link and the stratified tests, with R*_lo well above 1,
cannot be removed by any plausible monotone rescaling.

The stratified test also rejects when the stratifying locus rescales only
part of the liability, for example only the noise, because every genetic
effect then changes by the same factor. `relative_stratified_test(y, x, z,
background, ...)` separates the two. Within each stratum of z it estimates the
coefficient of x (b_s) and of an independent polygenic score P (c_s), and
tests b_s = κ c_s for one unknown κ: a ratio form (both coefficients clearly
non-zero) and a Fieller-type linear form that needs only c_s to be clearly
non-zero. P must not contain x, z or their chromosomes. Rejection means x's
effect changes across strata of z relative to the polygenic background. In UK
Biobank, frequency-matched null partners gave λ_GC 1.045 for the stratified
test and 1.07-1.08 (tails nominal) for the relative test.

## Recommended workflow

1. Discovery: within-class ranks, common covariate slopes, the trans
   background, G = 200; test `dosage` and `rho` (and `log_amp` for
   amplification), Bonferroni within each statistic.
2. Confirmation of each finding: repeat with `"covariate_slopes": "class"`;
   check the pooled-rank statistics only as a comparison with FAME's scale;
   localize the background signal (annotations, jackknife blocks) and test the
   implicated pairs with the stratified and background-relative tests.
3. Report intervals from the Hoeffding or jackknife SE; the null-restricted SE
   is for testing.
4. Use the sign-flip p-values (the default) for thresholds, in particular for
   annotation-set statistics such as `dosage[A]`.
