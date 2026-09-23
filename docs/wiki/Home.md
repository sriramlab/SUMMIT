# SUMMIT user guide

SUMMIT estimates heritability and genetic correlation, computes reference LD
scores, and fits gene–environment interaction models and polygenic scores.

Start with [Installation](Installation.md), try the synthetic
example below, then choose an analysis guide. [Input files](Input-files.md)
describes the formats needed for your own data.

```bash
python example/prepare_example_inputs.py
bash example/estimate_partitioned_gwldscore.sh
bash example/h2_ldscore.sh
```

Run these commands from an installed SUMMIT checkout. Generated inputs and
results go to `example/out/`.

## Choose an analysis

| Analysis | Guide |
|---|---|
| Genome-wide or windowed LD scores | [LD scores](LD-scores.md) |
| Total or partitioned h² and rg | [Heritability and genetic correlation](Heritability-and-genetic-correlation.md) |
| Many traits or annotation models | [Batch analyses](Batch-analyses.md) |
| One environment, including heterogeneous residual variance | [G×E models](GxE-models.md) |
| Joint continuous and categorical environments | [Multiple environments](Multiple-environments.md) |
| Context-dependent genetic prediction | [Polygenic scores](Polygenic-scores.md) |
| Python model construction and mathematical details | [PGS API](PGS-API.md), [Methods](Methods.md) |
| Research extensions | [Contextual Python API](Contextual-Python-API.md) |

## Results and background

- [Real-data results](Real-data-results.md): genetic-variance estimates from
  UK Biobank, with aggregate figures and downloadable plot data.
- [Benchmarks](Benchmarks.md): runtime, memory, and reproduction commands.
- [Methods](Methods.md): model definitions and estimating equations.
- [Troubleshooting](Troubleshooting.md): input, installation, and fitting errors.

The main `summit` command supports LD scores, h²/rg, and one-environment G×E.
Joint generalized G×E estimation is available through Python, with a separate
planning and inspection command. PGS uses `summit-pgs`. Research functions in
`summit.context` are identified in their guide.
