# Installation

## Requirements

Use Linux for the widest feature support. SUMMIT requires Python 3.10 or newer,
a C++17 compiler, OpenMP, and BLAS/LAPACK. Conda installs the numerical libraries
and Python packages listed in `environment.yml`; install a compiler separately
if one is unavailable on your system.

On macOS, a compiler with OpenMP support is needed. The Linux direct G×E
backend is unavailable there.

## Standard installation

```bash
git clone https://github.com/bronsonj98/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
python -m pip install .
```

Verify the commands:

```bash
summit --help
summit-pgs --help
summit-generalized-gxe-variant-ldscore --help
```

The distribution is currently named `gwldcore`; its Python package and main
command are named `summit`. A normal install compiles the native modules.

## Generalized G×E reference estimation

The generalized reference executor currently requires a Linux build linked to
pthread BLIS. The standard OpenBLAS installation supports the other analysis
paths, but does not satisfy this executor's build check.

For a site-specific BLIS build, configure `GXELDCORE_USE_PRIVATE_BLIS` and the
archive/include locations in `CMakeLists.txt`. The accompanying source metadata
is supplied by the person building that library. This setup is needed only for
the direct generalized reference executor; it is not an extra input to an analysis.

## Development installation

From an activated environment:

```bash
python -m pip install --no-build-isolation -e .
```

Reinstall after changing native code. If you use a separate worktree with an
older editable installation in the same environment, the PGS checkout launcher
selects the current source tree for that process:

```bash
python scripts/prediction/checkout.py test -q
```

That launcher requires native modules built for the checkout. It does not
change the shared environment's installation.

See [Troubleshooting](Troubleshooting.md) for compiler and library errors.
