# Installation

## Requirements

Use Linux for the widest feature support. SUMMIT requires Python 3.10 or newer,
a C++17 compiler, OpenMP, and BLAS/LAPACK. Conda installs the numerical libraries
and Python packages listed in `environment.yml`; install a compiler separately
if one is unavailable on your system.

macOS supports BED-based G×E reference estimation and scoring, including the
optimized native backend and generalized per-SNP executor. Use OpenBLAS from
the same Conda environment as NumPy. Private BLIS, explicit CPU placement,
and NUMA controls remain Linux features. Pinned PGEN G×E input currently
requires Linux; use BED on macOS.

## Standard installation

```bash
git clone https://github.com/sriramlab/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
python -m pip install .
```

On macOS, install the Xcode Command Line Tools if needed (`xcode-select --install`)
and add OpenMP before the pip installation:

```bash
conda install -c conda-forge llvm-openmp
CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS python -m pip install .
```

To rebuild an existing installation after updating SUMMIT, use that same pip
command with `--no-cache-dir`. The default one-environment example uses NumPy;
add `--gxe-native-backend direct --rand-dist rademacher` to its `summit` command
to select native G×E execution. Use a new `--out` prefix when comparing the two
runs, and use the same probe distribution in both runs for numerical comparisons.

Verify the commands:

```bash
summit --help
summit pgs --help
summit reference --help
```

The distribution is currently named `gwldcore`; its Python package and main
command are named `summit`. A normal install compiles the native modules.

## Generalized G×E reference estimation

The generalized reference executor supports the standard protected OpenBLAS
build on Linux and macOS. The native implementation retains its two complete
genotype traversals and uses the same scientific definitions on both platforms.

For a site-specific BLIS build, configure `GXELDCORE_USE_PRIVATE_BLIS` and the
archive/include locations in `CMakeLists.txt`. The accompanying source metadata
is supplied by the person building that library. This optional Linux build
retains its process-local thread ownership and placement checks; it is not an
extra input to an analysis.

## Development installation

From an activated environment:

```bash
python -m pip install --no-build-isolation -e .
```

Reinstall after changing native code or console entry points, using the same
native build configuration. `python -m summit` also runs the command-line
interface.

See [Troubleshooting](Troubleshooting.md) for compiler and library errors.
