# Synthetic examples

Run `python prepare_example_inputs.py` to generate the inputs under
`out/synthetic/`. Every genotype, identifier, covariate, and phenotype is
created from a fixed random seed. No participant dataset is read.

The shell scripts generate missing inputs before running SUMMIT. h²/rg scripts
also create their LD-score input when needed. The two example traits are
identical, so the supplied-overlap example uses a value of one.

Use `--out` to generate inputs in another new directory. The generator reuses
its own completed example directory and refuses other existing directories.
Generated files are excluded from Git. See the [user guide](../docs/wiki/Home.md)
for real-analysis input requirements.

## Manuscript methods demo

After installing SUMMIT, run these commands from the repository root:

```bash
python example/prepare_example_inputs.py
bash example/estimate_partitioned_gwldscore.sh
bash example/h2_ldscore.sh
bash example/rg_supplied_overlap_covariance.sh
```

The demo generates 512 synthetic samples and 2,048 variants. Expected files
under `example/out/` include:

| File | Expected result |
| --- | --- |
| `small.2bins.gw.ldscore.gz` | LD scores for 2,048 SNPs and two annotation bins |
| `small.2bins.gw.mc.tsv` | Monte Carlo diagnostics with numerical status `ok` for both bins |
| `h2_ldscore.results.tsv` | Total h² approximately 0.40217; standard error approximately 0.09840 |
| `rg_supplied_overlap_covariance.log` | Total genetic correlation approximately 1.00372 |

The two traits are identical. The finite-sample, unconstrained correlation
estimate is close to one and can slightly exceed it. Last digits can differ
across numerical libraries and platforms.

The complete four-command sequence took about 18 seconds on Debian 12 with
an AMD EPYC 7501 CPU, using two numerical threads. This includes Python
startup, input generation, and file I/O. Individual estimator timings printed
inside the logs exclude some of that work. Allow roughly a minute on a typical
desktop; actual runtime varies. See the
[tested environment](../docs/wiki/Installation.md#tested-environments-and-hardware).

## Linux and macOS

Follow [Installation](../docs/wiki/Installation.md) first. On macOS, use the
Conda OpenBLAS build with the `openblas` development headers and `llvm-openmp`.
Both Apple Silicon and Intel Macs support the BED-based examples, including
native G×E execution.

From the repository root:

```bash
bash example/estimate_gwldscore.sh
bash example/estimate_gxe_ldscore.sh
```

The G×E shell script uses the NumPy backend. To try the optimized native backend,
use the complete command in [G×E models](../docs/wiki/GxE-models.md#native-backend-on-linux-and-macos),
with `--gxe-native-backend direct --rand-dist rademacher` and a new output prefix.
The [multiple-environment example](../docs/wiki/Multiple-environments.md#synthetic-reference-example)
also runs with the standard OpenBLAS installation on macOS.
