# Troubleshooting

## The package does not build

Activate the Conda environment before installing. Check that a C++17 compiler,
OpenMP, CMake, and BLAS/LAPACK are available. On a cluster, load the compiler
module recommended by the site. If CMake selects the wrong BLAS library, use:

```bash
CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS python -m pip install .
```

Avoid mixing the base Conda environment's shared libraries with those in the
SUMMIT environment. Rebuild after changing the compiler or numerical libraries.

## The installed command uses old code

An editable installation can point at another checkout. Reinstall from the
intended repository with `python -m pip install --no-build-isolation -e .`.
For a separate PGS worktree, see the checkout launcher in [Installation](Installation.md).

## Genotype input is ambiguous or unsupported

Use an explicit `.bed` or `.pgen` filename. Check that its two companion files
exist. PGEN needs a plain-text PVAR and biallelic diploid variants. The direct
generalized G×E reference executor currently accepts BED only.

## Samples or variants do not match

Check `FID IID` strings, SNP order, genome build, and allele definitions. Do not
repair a mismatch by sorting only one file. For one-environment G×E trait batches,
make a shared complete sample set. PGS supports separate trait masks but still
requires each trait's saved scaling and feature definitions.

## An output already exists

Most G×E and PGS operations require a new prefix or directory. Choose a new
location after a failed run. A PGS directory without its completion marker is
incomplete and cannot be loaded as a model.

## A model is rank-deficient

Look for redundant fixed effects, duplicate annotations, empty bins, or context
columns with no variation. More random vectors cannot resolve structural
redundancy. Simplify the model or define an appropriate retained context basis.

## PGS does not converge

Read the candidate's reported true residual and termination reason. Check that
residual variances are positive and that priors use the original genotype and
phenotype scales. More iterations can help a valid slowly converging system;
loosening the tolerance changes numerical accuracy and should be recorded.

## Memory use is too high

Reduce SNP blocks, model/RHS tiles, or batch size. PGS `stream` mode avoids
retaining the genotype panel. Cached PGEN uses more bytes than cached BED.
A planner cannot guarantee an operating-system RSS ceiling; allow room for
libraries, parsers, and output arrays.
