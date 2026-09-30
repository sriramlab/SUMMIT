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
