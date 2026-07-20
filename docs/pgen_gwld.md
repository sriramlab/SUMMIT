# PGEN LD-score estimation

## Mathematical conclusion

PGEN support does not require a new genome-wide estimator. Once a genotype
backend supplies the same standardized sample-by-variant matrix, the existing
randomized algorithm and its normalization are independent of the on-disk
format. The substantive questions are instead:

1. which numeric matrix is decoded (hard calls or dosages),
2. whether its missing values, allele orientation, sample order, and scaling
   match the BED path, and
3. whether that matrix can be streamed without adding a full-genotype copy.

The PGEN path therefore shares the existing phase-1 and phase-2 native kernels
with BED after decoding and standardization.

## Exact finite-sample estimand

Let `D` be the decoded `N x M` dosage matrix after sample selection. For variant
`j`, SUMMIT computes its mean and sample variance over the `n_j` nonmissing
values,

```text
mean_j = sum_observed D_ij / n_j
var_j  = sum_observed (D_ij - mean_j)^2 / (n_j - 1),
```

and forms `G_j` by standardizing observed values and setting missing values to
zero. Zero is mean imputation on this standardized scale. Monomorphic and
all-missing columns remain zero.

If `C` is the orthonormal basis of the retained, centered covariates, define

```text
P = I - C C',       p = rank(C),       d = N - p - 1.
```

Without covariates, `P = I`, `p = 0`, and `d = N - 1`. For each variant,

```text
q_j = 1 / sqrt(max(G_j' P G_j / d, eps)),
X_j = P G_j q_j,
r_ij = X_i' X_j / d.
```

Except for an epsilon-clipped column, `X_j' X_j = d`, so `r_ij` is the
covariate-residualized sample correlation. A zero column contributes zero.

For nonnegative annotation value `a_jk`, probe `v`, and a random vector with
`E[z_v z_v'] = I`, phase 1 constructs

```text
W_kv = sum_j sqrt(a_jk) X_j z_jv,
```

and phase 2 accumulates

```text
Lhat_ik = (1 / V) sum_v (X_i' W_kv / d)^2.
```

Conditional on the realized decoded matrix,

```text
E_z[Lhat_ik | D] = sum_j a_jk r_ij^2.
```

This follows directly from `E[z_jv z_lv] = 1(j=l)`. Rademacher, Gaussian, and
the blockwise spherical probes used by SUMMIT all satisfy this identity. The
reported score applies the same finite-sample null subtraction as the BED
implementation,

```text
reported_L_ik = Lhat_ik - (sum_j a_jk) / d.
```

Consequently, changing BED decoding to PGEN decoding changes neither the
conditional probe expectation nor the null correction. It can change the
answer only by changing `D` (or through ordinary floating-point/probe error).

## Why dosage LD is not generally hard-call LD “in expectation”

An imputation dosage is usually a conditional mean, not a realized genotype.
If `H_j` is a latent hard-call genotype, `O` is the imputation information, and
`D_j = E[H_j | O]`, the law of total covariance gives

```text
Cov(H_j, H_k)
  = Cov(D_j, D_k) + E[Cov(H_j, H_k | O)].
```

Similarly, `Var(H_j) = Var(D_j) + E[Var(H_j | O)]`. Thus equality of the first
moments `E[D_j] = E[H_j]` does not imply equality of covariance, correlation,
or squared correlation. Sample centering, variance normalization, and squaring
make an equality claim still less tenable.

The rigorous interpretation is therefore:

- a hard-call PGEN targets hard-call-matrix LD and should agree with a BED that
  decodes to the same matrix;
- a dosage PGEN targets dosage-matrix LD; and
- agreement between dosage and hard-call LD is an empirical property of a
  particular panel and imputation quality, not a general unbiasedness result.

## Allele orientation

The established SUMMIT BED decoder counts the BIM A2 allele. The PGEN reader
therefore requests `allele_idx=0` (PVAR REF dosage), while output metadata maps
PVAR ALT to A1 and REF to A2. This mapping does not imply that A2 in an
arbitrary PLINK 1 BIM is necessarily the biological REF allele; REF/ALT status
must come from trustworthy metadata or a controlled conversion.

An allele flip changes the sign of a standardized column, so exact `r^2` is
unchanged. It does not, however, preserve a particular finite set of seeded
random-probe realizations unless the corresponding probe signs are also
flipped. Symmetric probes make the distributions equal, but not generally the
same-seed values. Matching REF/A2 orientation is therefore required for a
bitwise BED/PGEN backend test.

## Streaming implementation

The PGEN backend is designed around pgenlib's variant-major range API:

1. Open one persistent `PgenReader` for the analysis.
2. Pass the final sorted sample indexes to `sample_subset`, so excluded samples
   are not materialized in Python.
3. Allocate one reusable C-contiguous `(step_size, N)` dosage buffer.
4. Decode a variant range with `read_dosages_range(..., allele_idx=0,
   sample_maj=0)`.
5. Transpose it as a zero-copy Fortran-contiguous `(N, block_size)` view.
6. Mean-impute and standardize that view in place in the native extension.
7. Pass it to the same phase-1 or phase-2 kernel used by BED.

No full `N x M` matrix and no per-block transpose copy are created. The decode
buffer costs approximately

```text
step_size * N * sizeof(dtype)
```

bytes. `--step_size` therefore trades range-call overhead against buffer size
and native matrix-multiplication granularity. The estimator normally scans the
PGEN once in phase 1 and once in phase 2 per random-vector tile. Residual
variance computation is fused into the first phase-1 scan, avoiding a third
PGEN pass.

The default random-probe noise diagnostic also avoids a probe-by-SNP array.
Phase 2 accumulates one fourth-moment sufficient statistic per annotation;
see [Genome-wide LD-score Monte Carlo noise](gwld_mc_noise.md). The optional
`--write-ld-mc-var` output uses one additional `M x K` float64 array.

This design follows the official
[pgenlib Python API](https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt),
which specifies variant-major dosage range reads, sorted `uint32` sample
subsets, REF dosage at `allele_idx=0`, and `-9` as the missing-dosage sentinel.

## Supported scope

SUMMIT supports plain-text PVAR metadata and biallelic diploid PGEN data in all
three LD-score modes:

SUMMIT does not read BGEN directly. Convert BGEN input to a biallelic diploid
PGEN/PVAR/PSAM trio before running these estimators.

- The randomized genome-wide estimator uses the dense CPU kernels, dosage mean
  imputation, and `ddof=1`. It supports the default annotation-level random-
  probe noise diagnostic and optional per-SNP MC intervals.
- The deterministic windowed estimator streams standardized dosage panels into
  a NumPy implementation of the same prepared-panel matrix expression used by
  the native BED path. It supports
  annotations, sample subsets, and covariate residualization, and uses the
  established windowed-LD `ddof=0` genotype-standardization convention.
- The randomized GxE estimator streams standardized dosage blocks through its
  additive-interaction and interaction-interaction calculations. It supports
  annotations, sample subsets, environments, and covariates.

PGEN input is currently CPU-only and uses dosage mean imputation. Genome-wide
PGEN rejects HWE imputation, CUDA, finite-sample skew correction, and K-moment
output; the windowed and GxE paths likewise do not add hard-call-only HWE
behavior. Multiallelic and non-diploid variants remain explicitly out of scope.
Compressed `.pvar.zst` metadata are not yet read; provide a plain-text `.pvar`.

Genome-wide `ddof != 1` is rejected for BED as well as PGEN: the estimator's
residual normalization, phase-2 divisor, and final null are all defined with
`d = N - 1` or `N - p - 1`.

### Windowed PGEN I/O

The windowed path keeps one persistent `PgenReader`, decodes consecutive
variant panels, and retains prepared panels in a bounded least-recently-used
cache. It never materializes the full sample-by-variant matrix. The defaults
choose a panel width and cache budget from the sample count, chunk size, and
available memory. Automatic caching is capped at 4 GiB and one eighth of
available memory; when `--target-mem` is set, it is also capped at one quarter
of that budget. `--win-panel-cols` overrides the decoded panel width and
`--win-cache-mb` sets the cache budget (`0` disables caching, `-1` selects the
automatic value). The `SUMMIT_WIN_CACHE_MB` environment variable can also set
the automatic cache budget.

This panel/cache design avoids repeatedly decoding a left panel when a physical
window crosses several panel boundaries. Candidate panel pairs are selected
coarsely for efficient matrix multiplication, then every corrected squared
correlation outside the exact inclusive base-pair window is masked to zero.
Thus panel width and chunk boundaries affect I/O and memory use, not which
variant pairs contribute. A PGEN and BED matrix decoded to identical values
therefore use the same deterministic matrix expression.

### Metadata validation

PVAR/BIM variant IDs and PSAM/FAM sample IDs must be unique. Full annotation
files are aligned by SNP ID and then required to have exactly matching
canonical chromosome and integer base-pair coordinates; thin annotation files
must have exactly one row per genotype variant. Variant order must be
chromosome-contiguous, and windowed LD additionally requires nondecreasing BP
within each chromosome. These checks catch ordering and coordinate mismatches,
but SUMMIT cannot infer a named genome build from CHR/BP alone. Users must
supply genotype, annotations, and LD scores from the same build.

## Command examples

An explicit `.pgen` path is recommended when BED and PGEN trios share a prefix.
For example:

```bash
# Randomized genome-wide dosage LD
summit --geno ref.pgen --annot annot.gz --nvecs 1000 \
  --impute-method mean --out ref.dosage

# Deterministic 20 Mb windowed dosage LD
summit --geno ref.pgen --annot annot.gz --ld-wind-kb 20000 \
  --impute-method mean --out ref.dosage.20mb

# Genome-wide dosage GxE LD
summit --geno ref.pgen --annot annot.gz --env environment.txt \
  --impute-method mean --nvecs 1000 --out ref.dosage.env
```

## Validation requirements

The automated tests use three distinct checks:

1. A hard-call PGEN and matching BED must produce identical selected rows,
   standardized blocks, fused residual scales, seeded phase-1 panels,
   phase-2 accumulators, and final scores in the same build.
2. Fractional PGEN dosages are compared with an exact LD score computed from
   the decoded dosage matrix, not with hard calls.
3. For `V` Rademacher probes and `b_j = r_ij`, the Monte Carlo standard error
   used for the second check is

```text
SE_i = sqrt(2 * ((sum_j b_j^2)^2 - sum_j b_j^4) / V).
```

The stochastic test requires error within five such standard errors plus a
small numerical allowance. Monomorphic, all-missing, fractional, subsetted,
and short-final-block cases are included explicitly.

## UK Biobank chr21 development validation

The implementation was also checked against a July 2026 chr21 fixture derived
from UK Biobank BGEN dosages: 16,562 variants shared with the existing EUR
unrelated hard-call panel, with nested 1,000- and 10,000-sample subsets. No UK
Biobank data are included in this repository.

- At both sample sizes, a hard-call-only PGEN, its exported BED, and the
  production BED produced bitwise-identical LD-score arrays in the same build
  and probe realization. The 1k run used 256 probes; the 10k run used 128.
- For the first 512 dosage variants at 1k samples, an independently decoded
  exact dosage-matrix oracle had zero five-standard-error violations with
  16,384 Rademacher probes; the largest absolute error was 2.59 probe standard
  errors. The oracle used the same `1e-10` residual-variance floor, and no
  variant was floor-clipped.
- Dosage-versus-production-hard-call LD scores had Pearson correlations
  0.999307 at 1k and 0.999827 at 10k. These are descriptive comparisons between
  different matrices, not equality tests.
- A two-pass float32 range-read benchmark at 10k samples was fastest at block
  size 512 among the tested sizes 128, 512, 1024, 2048, and 4096: approximately
  52,025 variant records/second on the warm pass with a 20.48 MB dosage buffer.
  The 4096-variant buffer was slower and eight times larger.

The production and PGEN-exported BED payloads were byte-identical, but 5,086
of 16,562 production BIM rows had their two allele labels reversed relative to
PVAR. This does not affect SUMMIT's label-blind squared-LD calculation or the
observed numeric equality, but it is a metadata inconsistency and reinforces
why arbitrary BIM A2 labels must not be assumed to be biological REF alleles.
