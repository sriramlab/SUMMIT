# SUMMIT

**S**ummary-statistics-based **U**nified **M**ethod for **M**ultivariate **I**nference of **T**raits

```summit``` is an efficient tool for accurately estimating:
1. genome-wide LD scores from biobank data
2. heritability of phenotypes from summary statistics
3. genetic correlation from summary statistics

## Quick Start (Conda recommended)

### 0. Requirements
- Linux or macOS
- Anaconda3 / Miniconda
- C++17 compiler (e.g., `g++` on Linux; Xcode CLT on macOS)

### 1. Clone the repository
```bash
git clone https://github.com/bronsonj98/SUMMIT.git
cd SUMMIT
```

### 2. Create a virtual environment
```bash
conda env create -f environment.yml
conda activate summit
python -m pip install -U pip
```

### 3. Build & install
```bash
pip install -v .
```

You can check whether the ```gwldcore``` module has been successfully built or not by running:
```bash
python -c "import gwldcore, numpy; print('gwldcore OK; numpy', numpy.__version__)"
```

## How to use ```summit```
### 1. Estimating partitioned heritability from summary statistics

```summit``` can accurately (i.e., comparable to methods that use individual-level data) estimate heritability from summary-level data.
To estimate (partitioned) heritability, you need the following: 
1. LD scores (either genome-wide or fixed-window) **OR** trace summaries calculated from in-sample or reference population genotype.
2. GWAS summary statistics for the trait of interest

Here is an example run command (```h2_ldscore.sh``` in the [example](example) directory):
```
python3 ../src/summit.py --h2 ./sim_50k_h2_0.25_p_0.01.sumstat \
                  --ldscores ./double_uniform_10k_stoc_k100.gw.ldscore.gz \
                  --annot ./double_uniform_0.2.txt \
                  --verbose \
                  --njack 1000
```
Running the script should result in a total heritability estimate ```h^2``` of 0.27439 and ```SE``` of 0.01578. Here, the partitioned genome-wide LD scores (```gw.ldscore.gz```) estimated using ```k=100``` random vectors and ```N=10,000``` reference sample have been used.
Currently, to estimate partitioned heritability, the "thin" annotation file used for calculating the LD scores must also be provided separately (to be updated).

Similarly, you may estimate heritability using the trace summary statistics (.tr), which is mathematically equivalent to genome-wide sum of the LD scores (```h2_tr.sh``` in the [example](example) directory):
```
python3 ../src/summit.py --h2 ./sim_50k_h2_0.25_p_0.01.sumstat \
                  --trace ./double_uniform_50k_k100 \
                  --annot ./double_uniform_0.2.txt \
                  --verbose
```
You should get an heritability estimate of 0.26851 and ```SE``` of 0.01512. Note, the ```h^2``` estimates are different, as the trace summaries (```.tr```) are estimated from the ```N=50,000``` in-sample genotype, compared to the stochastic genome-wide LD scores, calculated from the ```N=10,000``` reference genotype. Still, both estimates overlap well within their error ranges.

If you'd like to create your own trace summaries, please refer to the ```PyRHE``` program from our lab: https://github.com/sriramlab/PyRHE. You may also use the C++ version of ```GENIE```.
You would need your own individual-level genotype for this (running with ```-tr``` option will save the trace summaries). While less flexible than using SNP-level LD scores, the trace summaries are directly estimated with ```GENIE``` or ```pyRHE```, and the files are a lot smaller in size (less than 0.1 MB).

### 2. Estimating genome-wide LD scores

Other methods use sliding fixed-sized windows (typically < 2Mb) to estimate the LD scores of the SNPs. This results in under-estimation of LD scores, as the long-range LD (> 2Mb) is not captured. Often times, this results in upward-biased estimates of heritability. One of the advantages of Randomized Haseman-Elston regression is that it can efficiently estimate genome-wide correlations between SNPs through random projection. In the original RHE papers, this is used to estimate the trace of squared kinship matrix. ```summit``` extends this idea further by estimating the genome-wide (partitioned) LD scores.

Here is an example run command (```estimate_gwldscore.sh``` in the [example](example) directory):
```
python3 ../src/summit.py --geno ./small \
                  --annot ./small.2bins_annot.txt \
                  --covar ./small.cov \
                  --out ./small.2bins \
                  --nvecs 100 \
                  --num-threads 8 \
                  --step-size 1000
```
This script should run within a few seconds and create a gzip file named ```small.2bins.gw.ldscore.gz``` and ```small.2bins.gw.log```. The file format of ```small.2bins.gw.ldscore.gz``` is identical to the traditional LDSC LD scores, where the first three columns are metadata ('CHR', 'SNP', 'BP'), and the remaining columns the (partitioned) LD scores.

## Parameters

```
--trace : File path for trace summary statistics (.tr) and corresponding metadata (.MN). If the path is a directory, all trace summaires (ending with .tr) will be used by aggregating them.
--save-trace : File path for saving (aggregated) trace summaries (.tr) and corresponding metadata (.MN)
--bim : File path for the reference .bim file used for trace calculation (required for trace summaries)
--max-chisq : Filter out SNPs with chi-sq statistic above the threshold.
--ldscores : File path for LD scores of the reference SNPs. You may use either the traditional (truncated) LD scores (.l2.ldscore.gz) or genome-wide stochastic LD scores (.gw.ldscore.gz)
--out : Output file path to save the analysis log and result (.log) or the genome-wide LD scores (.gw.ldscore.gz)
--verbose : Verbose mode: print out the normal equations
--suppress : Suppress mode: do not print out the outputs to stdout (log file only)
--njack : Number of jackknife blocks (only if using LD scores as input). Default is 100
--annot : Path of the annotation file (if using partitioned heritability). You may either use the "thin annot" (annotation matrix only) or the LDSC-style annotation files (.annot.gz).
--geno : Path of the genotype file to calculate the genome-wide LD scores. Calculates partitioned scores if --annot is also specified.
--num-threads: Cap the number of threads for BLAS to limit CPU usage. Default is 4.
--nvecs : Number of random vectors to use for estimating stochastic genome-wide LD scores. Default is 10000.
--step-size : Number of SNPs to process in each step of estimating stochastic genome-wide LD scores. Default is 1000.
--seed : Seed for estimating stochastic genome-wide LD scores. If not specified, the default numpy (pseudo) random number generator will be used.
--h2 : File path for phenotype-specific summary statistics (.sumstat[.gz]) to estimate heritability. If the path is a directory, all summary statistics (ending with .sumstat[.gz]) will be used.
--rg : Comma-separated file path for a pair of phenotype-specific summary statistics (.sumstat[.gz]) to estimate genetic correlation (rg).
--covar : Path of the covariate file to adjust for when calculating the genome-wide LD scores. If not specified, no covariates adjustments are made.
--rand-dist : Specify which distribution to use to generate random vectors ('normal', 'rademacher', 'spherical'). Default is spherical distribution.
--dtype : Specify the dtype to use for calculations (either float32 or float64). Default is float32.
--rand-samp : Select a random subset of the samples for LD score calculation. Pass a value between (0, 1] for a ratio, and an integer greater than 100 for the number of samples.
--ddof : Specify the delta degrees of freedom (ddof) for estimating genome-wide LD scores. Default is 1 (empirical SD).
```

## TODO's
✅ genetic correlation

✅ both-side filtering of outlier SNPs

☑️ partitioned genetic correlation

## References
```SUM-RHE``` is now published on Genome Research as an open access article. For references, please cite
```
Jeong, M., Pazokitoroudi, A., Liu, Z., & Sankararaman, S. (2024). Scalable summary statistics-based heritability estimation method with individual genotype level accuracy. Genome research, gr.279207.124. Advance online publication. https://doi.org/10.1101/gr.279207.124
```


