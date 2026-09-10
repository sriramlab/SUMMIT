# SUMMIT

**S**ummary-statistics-based **U**nified **M**ethod for **M**ultivariate **I**nference of **T**raits

SUMMIT estimates heritability and genetic correlation from GWAS summary
statistics, computes LD scores from reference genotypes, and fits
gene–environment interaction models and polygenic scores.

## Main features

- Genome-wide and windowed LD-score estimation.
- SNP heritability and genetic correlation, including annotation partitions.
- Batch analysis of many traits and annotation models.
- Gene–environment interaction (G×E) and environment-dependent residual variance.
- Joint models of multiple continuous or categorical environments.
- G×E polygenic score fitting, scoring, and calibration.

Genotype input can be PLINK BED hard calls or biallelic diploid PGEN dosages.
See the [user guide](docs/wiki/Home.md) for the inputs and availability of each method.

## Install

You need Conda, Python 3.10 or newer, and a C++17 compiler with OpenMP support.
The environment file supplies the Python and numerical-library dependencies.
Linux is recommended; the direct generalized G×E backend requires Linux.

```bash
git clone https://github.com/bronsonj98/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
python -m pip install .
summit --help
summit-pgs --help
```

See [Installation](docs/wiki/Installation.md) for compiler setup, development
installs, and the additional build needed for generalized G×E reference estimation.

## Try it

Generate a small synthetic dataset and estimate partitioned LD scores:

```bash
python example/prepare_example_inputs.py
bash example/estimate_partitioned_gwldscore.sh
bash example/h2_ldscore.sh
```

Examples write to `example/out/`. They illustrate the commands; their small
sample sizes are unsuitable for evaluating statistical performance.

## Documentation

- [Input files](docs/wiki/Input-files.md)
- [LD scores](docs/wiki/LD-scores.md)
- [Heritability and genetic correlation](docs/wiki/Heritability-and-genetic-correlation.md)
- [Batch analyses](docs/wiki/Batch-analyses.md)
- [G×E models](docs/wiki/GxE-models.md)
- [Multiple environments](docs/wiki/Multiple-environments.md)
- [G×E polygenic scores](docs/wiki/Polygenic-scores.md)
- [Benchmarks](docs/wiki/Benchmarks.md)

## Citation

Jeong, M., Pazokitoroudi, A., Liu, Z., & Sankararaman, S. (2024).
Scalable summary statistics-based heritability estimation method with individual
genotype level accuracy. *Genome Research*, gr.279207.124.
[Paper](https://doi.org/10.1101/gr.279207.124).
