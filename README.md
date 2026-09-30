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
- Cross-trait genetic response covariance through the research Python API.
- Ascertainment-aware binary-trait regression with PCGC.
- G×E polygenic score fitting, scoring, and calibration.

Genotype input can be PLINK BED hard calls or biallelic diploid PGEN dosages.
See the [user guide](https://github.com/sriramlab/SUMMIT/wiki) for the inputs and availability of each method.

## Install

You need Conda, Python 3.10 or newer, and a C++17 compiler with OpenMP support.
The environment file supplies the Python and numerical-library dependencies,
including OpenBLAS headers. Linux and macOS (Apple Silicon and Intel) support
the optimized native G×E backend with BED input. On macOS, first install the
Xcode Command Line Tools if needed with `xcode-select --install`.

```bash
git clone https://github.com/sriramlab/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
```

On macOS, add OpenMP before building:

```bash
conda install -c conda-forge llvm-openmp
```

Then install SUMMIT on either platform:

```bash
CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS python -m pip install .
summit --help
summit pgs --help
summit reference --help
```

See [Installation](https://github.com/sriramlab/SUMMIT/wiki/Installation) for
upgrading an existing environment, development installs, and platform limits.
If a build reports missing OpenBLAS headers, install the `openblas` development
package as described in [Troubleshooting](https://github.com/sriramlab/SUMMIT/wiki/Troubleshooting).

## Try it

Generate a small synthetic dataset and estimate partitioned LD scores:

```bash
python example/prepare_example_inputs.py
bash example/estimate_partitioned_gwldscore.sh
bash example/h2_ldscore.sh
```

Examples write to `example/out/`. They illustrate the commands; their small
sample sizes are unsuitable for evaluating statistical performance.
See the [example guide](example/README.md) for G×E examples, including an
explicit native-backend command that also runs on macOS.

## Documentation

[Commands and options](docs/wiki/Commands-and-options.md) lists the commands and shared analysis options.

- [Input files](https://github.com/sriramlab/SUMMIT/wiki/Input-files)
- [LD scores](https://github.com/sriramlab/SUMMIT/wiki/LD-scores)
- [Heritability and genetic correlation](https://github.com/sriramlab/SUMMIT/wiki/Heritability-and-genetic-correlation)
- [Batch analyses](https://github.com/sriramlab/SUMMIT/wiki/Batch-analyses)
- [G×E models](https://github.com/sriramlab/SUMMIT/wiki/GxE-models)
- [Multiple environments](https://github.com/sriramlab/SUMMIT/wiki/Multiple-environments)
- [Cross-trait response models](https://github.com/sriramlab/SUMMIT/wiki/Cross-trait-analysis)
- [Binary traits and PCGC](https://github.com/sriramlab/SUMMIT/wiki/Binary-traits-and-PCGC)
- [G×E polygenic scores](https://github.com/sriramlab/SUMMIT/wiki/Polygenic-scores)
- [Real-data results](https://github.com/sriramlab/SUMMIT/wiki/Real-data-results)
- [Benchmarks](https://github.com/sriramlab/SUMMIT/wiki/Benchmarks)

## Citation

Jeong, M., Pazokitoroudi, A., Liu, Z., & Sankararaman, S. (2024).
Scalable summary statistics-based heritability estimation method with individual
genotype level accuracy. *Genome Research*, gr.279207.124.
[Paper](https://doi.org/10.1101/gr.279207.124).
