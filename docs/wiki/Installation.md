# Installation

## Requirements

Use Linux for the widest feature support. SUMMIT requires Python 3.10 or newer,
a C++17 compiler, OpenMP, and BLAS/LAPACK. Conda installs the numerical libraries
and Python packages listed in `environment.yml`; install a compiler separately
if one is unavailable on your system.

macOS on Apple Silicon and Intel supports BED-based G×E reference estimation
and scoring, including the optimized native backend and generalized per-SNP
executor. Use OpenBLAS from the same Conda environment as NumPy. Private BLIS,
explicit CPU placement, and NUMA controls remain Linux features. Pinned PGEN
G×E input currently requires Linux; use BED on macOS.

The environment file installs OpenBLAS 0.3.31 or newer, including the `openblas`
development package that supplies its headers. Keep NumPy and SUMMIT linked to
the OpenBLAS in that environment; the native G×E backend checks this at runtime.

## Standard installation

Start with Conda installed, then follow the steps below to create a new SUMMIT
environment.

On macOS, first install Apple's Xcode Command Line Tools if they are not already
available, then wait for the installer to finish:

```bash
xcode-select --install
```

On either Linux or macOS, create the environment from the repository:

```bash
git clone https://github.com/sriramlab/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
```

On macOS, add OpenMP to that environment before building:

```bash
conda install -c conda-forge llvm-openmp
```

Then build and install on either platform:

```bash
CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS python -m pip install .
```

Verify the commands:

```bash
summit --help
summit pgs --help
summit reference --help
```

The distribution is currently named `gwldcore`; its Python package and main
command are named `summit`. A normal install compiles the native modules.

## Run the examples

From the repository root, run the small synthetic examples:

```bash
bash example/estimate_gwldscore.sh
bash example/estimate_gxe_ldscore.sh
```

These generate their inputs and write results under `example/out/`. Choose a
new output prefix for additional G×E runs. The one-environment shell example
uses the NumPy backend; see [G×E models](GxE-models.md#native-backend-on-linux-and-macos)
for a complete native command using `--gxe-native-backend direct` and
`--rand-dist rademacher`. Use the same probe distribution when comparing backends.

## Generalized G×E reference estimation

The generalized reference executor supports the standard protected OpenBLAS
build on Linux and macOS. The native implementation retains its two complete
genotype traversals and uses the same scientific definitions on both platforms.
No additional build is needed with the OpenBLAS installation above. See
[Multiple environments](Multiple-environments.md#synthetic-reference-example)
for the synthetic native example.

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
