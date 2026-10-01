# Installation

## Requirements

SUMMIT supports Linux and macOS. It requires Python 3.10 or newer,
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

## Tested environments and hardware

The manuscript LD-score, heritability, and genetic-correlation workflows run
on ordinary CPU hardware. No GPU or other non-standard hardware is required.
Memory and runtime depend on the number of samples, variants, annotations,
and random vectors; the bundled synthetic demo is suitable for a desktop.

| System | Verification |
| --- | --- |
| Debian GNU/Linux 12, x86-64 | Native OpenBLAS build and the complete synthetic LD-score/h²/rg demo |
| macOS 15, Apple Silicon and Intel | Automated tests in GitHub Actions |
| macOS Tahoe 26.6.2 | Author-reported installation and functional testing |

The Debian demo was checked with Python 3.12.12, NumPy 2.3.5, pandas 2.3.3,
SciPy 1.16.3, bed-reader 1.0.0, pgenlib 0.94.1, psutil 7.1.3,
threadpoolctl 3.6.0, tqdm 4.67.1, and OpenBLAS 0.3.34. The build used
GCC 12.2.0, CMake 4.1.2, Ninja 1.13.1, nanobind 2.12.0, and
scikit-build-core 0.11.6. These are tested versions; the supported dependency
ranges are in `environment.yml` and `pyproject.toml`.

Allow a few minutes for the SUMMIT build after the compiler and dependencies
are installed. A native build took about 100 seconds with two build workers
on an AMD EPYC 7501 Linux system. Conda environment creation and dependency
downloads are additional and depend on the network and package cache.

## Run the examples

From the repository root, run the small synthetic examples:

```bash
python example/prepare_example_inputs.py
bash example/estimate_partitioned_gwldscore.sh
bash example/h2_ldscore.sh
bash example/rg_supplied_overlap_covariance.sh
```

These generate their inputs and write results under `example/out/`.
The [example README](../../example/README.md) lists expected outputs and timing.

For G×E, run `bash example/estimate_gxe_ldscore.sh`. Choose a
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
