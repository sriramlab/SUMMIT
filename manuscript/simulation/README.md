# Phenotype simulation

Two scripts generate phenotypes from PLINK BED genotypes using the simulation
models in the manuscript. Genotypes, annotations, and covariates are supplied
by the user. Use the SUMMIT environment from the
[installation guide](../../docs/wiki/Installation.md); it includes NumPy,
pandas, and bed-reader. PLINK 2 is needed for GWAS.

| Script | Models |
| --- | --- |
| `simulate_h2.py` | Continuous traits, binary traits, two-bin enrichment, and MAF–LD benchmarks |
| `simulate_rg.py` | Paired traits with GCTA or LDAK effects, including signed covariance and overlapping annotations |
| `format_sumstats.py` | Convert biallelic PLINK 2 linear GWAS results to SUMMIT inputs |

Run the commands below from `manuscript/`. Each simulation requires a new or
empty output directory. Memory and runtime depend on the genotype matrix and
number of replicates. The separate [synthetic demo](../../example/README.md)
provides a small, self-contained example of SUMMIT estimation.

## Inputs

For each population, use this relative layout (shown for EUR):

```text
inputs/genotypes/EUR/genotypes.bed
inputs/genotypes/EUR/genotypes.bim
inputs/genotypes/EUR/genotypes.fam
inputs/annotations/EUR/maf_ld.tsv
inputs/annotations/EUR/maf_ld_features.tsv
inputs/covariates/EUR.tsv
```

`maf_ld.tsv` is a headered 0/1 annotation matrix in BIM order, with exactly
one active bin per row. `maf_ld_features.tsv` has columns `maf ldscore` in the
same variant order. Compute these quantities on the intended cohort and
variant set using the annotation definitions in the manuscript. Covariates
have `FID`, `IID`, and named covariate columns.

For the continuous, binary, and enrichment models, a three-column feature
table (`maf_feat ld_feat maf`) can supply separate effect-weighting features
and MAF values for filtering. The equivalent headerless format is also accepted.
LDAK MAF–LD and rg simulations use `[maf * (1 - maf)]**0.75 / ldscore` weights.

Overlapping simulations use `inputs/annotations/EUR/baseline.annot.gz`.
This table contains the annotation names selected by `annot_cols` in the
settings CSV. `CHR/BP/SNP/CM` metadata columns are excluded. With `add_base=1`,
an all-ones base annotation is placed first, followed by the selected columns
in their specified order. Contributions from overlapping annotations are added.

## Heritability

For a two-bin continuous trait with total h² = 0.25:

```bash
python simulation/simulate_h2.py --model continuous \
  --bed inputs/genotypes/EUR/genotypes.bed \
  --annot inputs/annotations/EUR/maf_ld.tsv \
  --mafld inputs/annotations/EUR/maf_ld_features.tsv \
  --sigma 0.125,0.125 --p-causal 0.1 --num-reps 100 --seed 42 \
  --out-dir output/h2
```

Supply one `--sigma` value per annotation column. Their sum is the total
genetic variance; environmental variance is one minus this sum. Causal
variants are sampled within each bin, with at least one selected in each
bin with a positive variance target. `--maf-ex` and `--ld-ex` specify powers
of the two effect-weighting features. The genome-wide benchmarks varied h²
over 0.10, 0.25, and 0.40, and causal proportions over 1, 0.1, and 0.01.

Use `--model binary` with the same inputs and variance settings for a
liability-threshold trait. Add `--prevalence 0.1` for population prevalence
and `--sampling cohort` or `--sampling case-control --case-frac 0.5` for
ascertained sampling. The default coding is 1/2; `--coding 01` uses 0/1.
Binary simulations use all input variants; restrict the input data first
if a MAF filter is needed.

For two-bin enrichment, the annotation columns must be ordered `target, rest`:

```bash
python simulation/simulate_h2.py --model enrichment \
  --bed inputs/genotypes/EUR/genotypes.bed \
  --annot inputs/annotations/EUR/target_rest.tsv \
  --mafld inputs/annotations/EUR/maf_ld_features.tsv \
  --scenario strong --total-h2 0.25 --architecture gcta \
  --p-causal 0.1 --num-reps 100 --seed 42 --out-dir output/enrichment
```

The `weak`, `moderate`, and `strong` scenarios multiply the two bins' total
effect weights by (1.5, 0.75), (2, 0.5), and (3, 0.3), respectively, then
normalize to `--total-h2`. GCTA uses uniform weights; LDAK enrichment uses
`maf_feat**0.75 / ld_feat` (or `maf**0.75 / ldscore` with two-column inputs).

The MAF–LD benchmark parameter files are in `settings/maf_ld/<population>/`:

```bash
python simulation/simulate_h2.py --model maf-ld \
  --bed inputs/genotypes/EUR/genotypes.bed \
  --annot inputs/annotations/EUR/maf_ld.tsv \
  --mafld inputs/annotations/EUR/maf_ld_features.tsv \
  --params simulation/settings/maf_ld/EUR/h2_0.5_causal_0.01_low_maf_ldak.txt \
  --architecture ldak --num-reps 100 --seed 42 --out-dir output/maf_ld
```

These files require eight bins, ordered by MAF stratum and then LD quartile.
`--num-reps` overrides the file's `num_simul` value. For GCTA, select
`--architecture gcta` and the corresponding file without `_ldak`.
GCTA allocates the specified variance within each bin. LDAK allocates the
total variance across causal variants in bins with positive targets, using
the MAF/LD weights above. The parameter files contain cohort-specific
allocations; do not transfer them to a different genotype dataset unchanged.

All h² models write `sim_<replicate>.phen` with `FID IID pheno` columns,
`sim.oracle_h2.tsv` with the generating per-bin variance and causal counts,
and `settings.json`. Replicate numbering starts at zero, including single
replicate runs. Case-control runs also write the selected samples to `.keep`
files; `--keep-all` retains unselected samples with missing phenotypes.

## Genetic correlation

The GCTA covariance settings are supplied with the plotted results:

```bash
python simulation/simulate_rg.py \
  --settings data/sim_rg/partitioned/truth.csv \
  --input-dir inputs --out-dir output/rg
```

Each CSV row defines a population and simulation setting. `sig1` and `sig2`
give the per-bin genetic variances; `rho_g` gives the per-bin genetic
correlations. A scalar is repeated across bins. `gamma_e` is the environmental
covariance. The remaining columns specify the annotation file, output name,
replicate count, seed, and memory mode (`max_mem=1` loads all genotypes).
Use a CSV containing only the desired rows to run a subset of populations
or settings.

Use the same command with these settings files for the other rg simulations:

| Settings | Simulation |
| --- | --- |
| `simulation/settings/rg_signed.csv` | Signed covariance architectures |
| `simulation/settings/rg_ldak.csv` | LDAK effect variances |
| `simulation/settings/overlap.csv` | Focal overlapping annotation |

`architecture` selects `gcta` or `ldak`; `annotation_model` selects
`partition` or `overlap_additive`. The overlap settings specify the generating
components; the fitted model can include additional annotations as described
in the manuscript. Each population/setting directory contains
`sim_<replicate>_1.phen`, `sim_<replicate>_2.phen`, and `settings.json`.

## Numerical settings

The scripts retain the effect sampling, genotype standardization, and random
draw order of the manuscript simulations. Keep the input order, seed, chunk
size, replicate batch size, memory mode, and numerical environment fixed for
reruns. The defaults are:

| Model | `--chunk-size` | `--rep-batch` |
| --- | --- | --- |
| Continuous h² | 10000 | min(128, number of replicates) |
| Binary h² | 10000 | 64 |
| Enrichment | 10000 | 32 |
| MAF–LD GCTA | 8192 | 25 |
| MAF–LD LDAK | 8192 | 100 if fully polygenic, otherwise 10 |
| Partitioned rg | 8192 | 10 |
| Overlapping rg | 8192 | 100 |

MAF–LD and rg simulations use mean imputation; the other h² models use
Hardy–Weinberg draws for missing genotypes. `--max-mem` is available for
binary and enrichment h² simulations. MAF–LD and rg simulations default
to float64 accumulation; `--acc-dtype float32` selects float32.

Run the small synthetic regression checks with
`python simulation/test_simulation.py` in the SUMMIT environment.

## GWAS and SUMMIT

For a continuous phenotype, run a covariate-adjusted linear GWAS, retaining
the per-variant sample counts, effect alleles, BETA, and SE. For example:

```bash
mkdir -p output/gwas
plink2 --bfile inputs/genotypes/EUR/genotypes \
  --pheno output/h2/sim_0.phen --pheno-name pheno --no-psam-pheno \
  --variance-standardize --glm hide-covar \
  --covar inputs/covariates/EUR.tsv --threads 2 --out output/gwas/trait
python simulation/format_sumstats.py \
  --glm output/gwas/trait.pheno.glm.linear \
  --cov-rank 10 --out output/gwas/trait.sumstats.tsv
```

Replace `10` with the rank of the non-intercept covariates. The converter is
for linear GWAS results; logistic-regression estimates are not interchangeable
with them. Preserve the analysis scale and ascertainment treatment used for
the binary-trait benchmark.

Compute LD scores on the matching reference genotypes, covariates, and
annotations, then run SUMMIT using the
[LD-score](../../docs/wiki/LD-scores.md) and
[heritability/correlation](../../docs/wiki/Heritability-and-genetic-correlation.md)
instructions. For correlated traits, supply the appropriate sample-overlap
covariance; the environmental covariance used by the simulator is not in
general that quantity.

Exact original estimates also depend on the original genotypes, annotations,
covariates, and estimator versions. Figure reproduction uses the aggregate
tables supplied in `data/` and does not require generating the full simulation
summary statistics.
