# SUMMIT manuscript reproduction

Source data and code for *Genome-wide LD scores enable robust heritability and
genetic correlation estimation from summary statistics across genetic ancestries*.

This folder reproduces Figures 1–8 and S1–S44 from the supplied aggregate
results. It includes simulation estimates, real-trait estimates, plotting
summaries, and supplementary metadata. Individual-level genotypes and
phenotypes, and the full simulation GWAS summary statistics, are not included.

## Reproduce the figures

From this folder:

```bash
conda env create -f environment.yml
conda activate summit-manuscript
python verify_package.py
python reproduce.py --list
python reproduce.py
```

Results are written to `output/figs/main/` and `output/figs/supplementary/`.
The runner also writes figure-specific statistical summaries and recalculates
the curve and calibration inputs for S14 and S15. Supplied inputs are preserved.
Use a new output directory for another run:

```bash
python reproduce.py --figure 1 4 S12 --out-dir output/selected
```

`python reproduce.py --check` checks input availability without generating
figures. Paths in the manifest are relative to this folder; the runner also
works when invoked from another working directory. A CPU is sufficient.
Rendering all figures takes several minutes, including bootstrap calculations;
the high-resolution output can occupy substantially more space than the inputs.

## Files

| Location | Contents |
| --- | --- |
| [FIGURES.md](FIGURES.md) | Script and input mapping for each of the 52 figures |
| [figures.json](figures.json) | Exact figure commands and output names |
| `scripts/` | Plotting and statistical-summary code |
| `data/` | Aggregate source tables |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Table columns and row counts |
| [metadata/supplementary_data.xlsx](metadata/supplementary_data.xlsx) | Trait sample sizes, cohort characteristics, and external GWAS sources |
| [simulation/](simulation/README.md) | Phenotype simulators, parameter files, and GWAS-to-SUMMIT instructions |

The scripts retain the plotting functions, statistical filters, axis limits,
and resampling seeds used in the manuscript. They also export the associated
statistical summaries where the original analysis did so. This package does
not rebuild the manuscript LaTeX or every typeset table.

The plotting environment was checked on Debian 12 with Python 3.11.5,
NumPy 1.24.3, pandas 2.0.3, SciPy 1.11.1, Matplotlib 3.7.2, and seaborn 0.12.2.
Font rendering and image metadata can vary between platforms.

## Starting from genotypes or summary statistics

The [simulation guide](simulation/README.md) explains how to generate new
phenotypes on an available genotype dataset. Exact reruns of the UK Biobank
analyses require authorized access to the same genotype, covariate, and
annotation inputs described in the manuscript. The supplied aggregate tables
allow figure reproduction without that access.

External GWAS source links and release information are in the workbook's
`External GWAS` sheet. SUMMIT input formats and estimation commands are in the
[user guide](../docs/wiki/Input-files.md) and the
[heritability/correlation guide](../docs/wiki/Heritability-and-genetic-correlation.md).
The separate [synthetic software demo](../example/README.md) runs without
external data and provides expected numerical results.

Record the source commit (`git rev-parse HEAD`) and the environment when citing
or rerunning this package. Software uses the repository's [MIT License](../LICENSE);
external data retain their original access and use terms.
