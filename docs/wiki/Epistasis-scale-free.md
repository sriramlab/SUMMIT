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
exactly null, whatever the unknown h, a_K and s_K. On the **pooled rank**
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
| `log_amp` | log θ(hi, hi) / θ(lo, lo) | 0 | genetic-share ratio; exact on within-class ranks |
| `delta2` | θ(lo, lo) + θ(hi, hi) − 2θ(lo, hi) | 0, one-sided | FAME-like contrast |
| `sel_lam[A]` | log λ_A − log λ_rest, λ = θ(lo, hi)/θ(lo, lo) | 0 | selective across annotations; first-order scale-free |
| `sel_amp[A]` | log α_A − log α_rest, α = θ(hi, hi)/θ(lo, lo) | 0 | selective genetic share |
| `dosage[A]`, `dosage_diff[A]` | dosage in A, and A minus the complement | 0 | annotation-specific marginal epistasis |

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
G = 200 is the qualified default.

The background excludes every chromosome of the context. U-statistics need
the context to be independent of the background SNPs, which fails for SNPs in
LD with the context.

## One genotype pass

Each block of background SNPs is decoded once. It is standardized per
individual group with the shared native standardizer, and group scores are
formed by SUMMIT's protected TN product; the arithmetic equals that of a
single ungrouped product. The G × G kernels are many small products. They use
batched BLAS on scores augmented with a sum-over-groups row, so every product
carries its own row and column checksums. Mismatching kernels are recomputed,
and a repeated mismatch stops the run. Nothing of size M × G × q is stored.

Main-product cost is about 2 N M q flops, where q is the number of outcome
columns: one dosage column per analysis and scale, plus two pattern columns
per analysis and pattern scale. Kernel cost is about 2 G² M times the number
of statistics.

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

- **Variant contexts** code minor-allele counts on the analysed rows. With
  fewer than `min_minor_homozygotes` (default 300), minor homozygotes join the
  carriers.
- **File contexts** supply integer classes, their numeric values for the
  dosage statistic, the low and high classes for the pattern contrast, and the
  chromosomes to exclude.
- **Annotation tables** have a `SNP` column and weights in [0, 1]; the
  complement is derived.
- **Covariates** are partialled within class × stratum on every scale.
  Laboratory traits listed under `positive` drop non-positive values.

Run with explicit threads and memory:

```bash
summit epistasis scale-free scale_free.json --out results/ --num-threads 8 --memory-gib 32
```

Outputs:

| File | Contents |
|---|---|
| `results.tsv` | one row per analysis, scale and statistic: estimate, Hoeffding, jackknife and null-restricted SEs, the test SE (`se_rule` null or max), z and p |
| `run.json` | manifest, contexts, class sizes, genotype source and scale identities, timings, checksum repairs |
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
