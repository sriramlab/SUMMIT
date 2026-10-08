# Quantitative epistasis

SUMMIT tests interactions between supplied SNP pairs or between a target SNP
and a genetic score. It also estimates target-by-background variance components.
Choose targets, backgrounds, and test families before examining confirmation
outcomes. These workflows use quantitative traits; binary G×E estimation is
described in [Generalized G×E PCGC](../pcgc_gxe.md).

## Choose a model

| Analysis | Model | Interpretation |
|---|---|---|
| Supplied pairs or a fixed genetic score | `robust_mean` | Conditional mean effects with HC3 heteroskedasticity-robust covariance |
| Independently trained trans score with polygenic adjustment | `conditional_polygenic_mean` | Conditional mean effect under a declared Gaussian polygenic covariance model; experimental |
| Target-by-background variance | Default moment estimator | Variance attributed to a supplied interaction kernel, conditional on the other kernels |

For pair features H or a target-by-score feature, the finite mean model is

```math
Y=C\alpha+H\gamma+\epsilon,\qquad
H_i=X_{it}\sum_j w_jX_{ij}.
```

C contains the declared covariates and genetic main effects. The coefficient
γ measures association conditional on that design. A detected interaction
does not establish a causal mechanism. HC3 allows unequal residual variances
across independent people. It does not account for omitted mean effects or
residual dependence between people.

## Supplied pairs

Use BED or biallelic diploid PGEN with companion files, a selected sample table
with `FID IID`, and phenotype/covariate tables with those identifiers. Selected
phenotypes and covariates must be complete. Dominance terms require observed
hard calls, because dosage alone does not identify heterozygosity.

Save this as `pairs.json`, replacing the filenames and SNP IDs:

```json
{
  "kind": "summit.epistasis.prepare",
  "schema_version": 1,
  "genotypes": {"geno": "cohort.bed"},
  "samples": "samples.tsv",
  "phenotypes": {"file": "phenotypes.tsv", "columns": ["trait"]},
  "covariates": {"file": "covariates.tsv", "columns": ["age", "sex", "PC1"]},
  "genotype_scale": "hwe",
  "phenotype_scale": "raw",
  "annotations": {},
  "jobs": [{
    "id": "supplied_pairs",
    "additive_annotations": ["all"],
    "local_variants": ["rsTarget", "rsPartner1", "rsPartner2"],
    "dominance_variants": ["rsTarget", "rsPartner1", "rsPartner2"],
    "pairs": [["rsTarget", "rsPartner1"], ["rsTarget", "rsPartner2"]],
    "inference": {"method": "robust_mean", "save_reference": true}
  }]
}
```

Paths are relative to the manifest. Pair loci enter the main-effect design;
`local_variants` and `dominance_variants` allow additional adjustment. Remove
monomorphic and all-missing markers before preparation.

```bash
summit epistasis prepare pairs.json --out results/pairs \
  --num-threads 4 --block-size 128 --memory-gib 4
summit epistasis fit results/pairs/supplied_pairs.robust-score.npz \
  --out results/pairs.fit.json
```

Preparation saves scores, information, covariance, and feature definitions.
Fitting needs only the saved summary. Results include effect estimates, SEs,
their full covariance, a joint test, and supported group tests. Inspect the
rank and leverage diagnostics. Redundant features can leave individual
coefficients unidentified even when a joint effect is estimable.

Current support checks require at least 1,000 confirmation samples, fitted
rank at most 5% of N, maximum leverage at most 0.1, feature effective support
at least 100, and normalized condition number at most 10⁶. Passing these
checks does not establish calibration for an arbitrary study.

## Independently trained trans scores

Training and confirmation samples must be disjoint. Training estimates a
target-specific score using a prespecified trans background; confirmation
uses its frozen weights. All tuning and target selection must respect that
separation.

The conditional polygenic procedure includes additive, dominance, and supplied
covariate-dependent genetic covariance. For training outcomes Y₀ and
confirmation outcomes Y₁, the model uses their joint covariance V:

```math
E[Y_1\mid Y_0]=\mu_1+V_{10}V_{00}^{-1}(Y_0-\mu_0),\qquad
V_{1\mid0}=V_{11}-V_{10}V_{00}^{-1}V_{01}.
```

The workflow fits the declared nuisance mean and covariance and carries the
trained score into the confirmation model. This procedure remains
experimental. Coverage under covariance misspecification and calibration at
genome-wide significance thresholds have not been established.

For a target on a chromosome other than chromosome 5, save this recipe as
`trans.json`:

```json
{
  "kind": "summit.epistasis.trans_inputs",
  "schema_version": 1,
  "genotypes": {"geno": "cohort.bed", "genome_build": "GRCh37"},
  "target": "rsTarget",
  "background_chromosome": "5",
  "training_samples": "training.tsv",
  "confirmation_samples": "confirmation.tsv",
  "phenotype": {"file": "phenotypes.tsv", "column": "trait", "unit": "cm"},
  "covariates": {
    "file": "covariates.tsv",
    "columns": ["age", "sex", "PC1"],
    "varying_effects": ["PC1"]
  },
  "inference": {"method": "conditional_polygenic_mean"}
}
```

`make-inputs` selects complete cases and checks training genotype support.
It writes training and preparation manifests, sample lists, and selected local
main effects. Use `interaction_variants` with a SNP-list file in place of
`background_chromosome` for a supplied trans set. `varying_effects` declares
covariates whose genetic slopes enter the model.

```bash
summit epistasis make-inputs trans.json --out results/trans --num-threads 4
summit epistasis train-direction results/trans/train.json \
  --out results/trans/trained --num-threads 4 --memory-gib 8
summit epistasis prepare results/trans/prepare.json \
  --out results/trans/prepared --num-threads 4 --memory-gib 8
summit epistasis fit results/trans/prepared/trait.robust-score.npz \
  --out results/trans.fit.json
```

`train-direction --resume` and `prepare --resume` validate compatible completed
work or resume solver checkpoints. `prepare-traits` can reuse a cohort
reference for a new phenotype with matched samples and definitions.
Conditional polygenic preparation still needs genotypes and a matched
training fit for that phenotype.

## Variance components and group tests

For target genotype x, background genotypes X, nonnegative annotation weights
A, and covariate projection P, the interaction kernel is

```math
K=\frac{1}{\sum_j A_j}
P\mathrm{diag}(x)X\mathrm{diag}(A)X^T\mathrm{diag}(x)P.
```

The target is excluded from its background. A moment fit estimates the
coefficient of K jointly with additive and residual kernels. Replace `pairs`
and `inference` in a preparation job with a `components` list, for example
`[{"name":"interaction","target":"rsTarget","background":"region"}]`.
Define `region` in the manifest's `annotations` mapping from SNP IDs to
nonnegative weights, and include relevant additive annotations in
`additive_annotations`. Saved variance summaries use `.epistasis.npz`.

Moment estimates remain signed. Approximate Wald probabilities from a
variance fit need particular care at the zero-variance boundary. Gaussian
score and fitted-null procedures are also available through
`summit.epistasis.score` and `summit.epistasis.inference_workflow`, with their
stated covariance assumptions.

`combine` harmonizes compatible signed pair summaries across cohorts;
`followup` performs conditional tests within a supplied pair family. Preserve
allele coding, effect units, feature definitions, and sample-overlap
information. Correct multiplicity across all tested targets, backgrounds,
traits, and follow-up families.

## Computation

Preparation uses SUMMIT's genotype readers and matrix operations. Larger
backgrounds use streamed products and randomized reference summaries.
Increase `--nvecs` to assess reference precision; `--exact` is limited to small
numerical examples. Match process thread settings to `--num-threads` as
described in [Installation](Installation.md).

Use new output paths. Cohort references and training artifacts can contain
sample-aligned information and belong in protected storage. Reuse requires
matching samples, genotypes, scaling, features, and nuisance design.
