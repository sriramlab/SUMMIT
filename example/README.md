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
