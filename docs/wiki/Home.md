# SUMMIT user guide

Start with [Installation](Installation.md) and [Input files](Input-files.md).
Each analysis guide explains what to supply, how to run it, and how to read the results.

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

[Benchmarks](Benchmarks.md) describes measured performance and reproduction
commands. [Troubleshooting](Troubleshooting.md) covers input, build, and fitting
errors. [Development](Development.md) describes the source layout and tests.

The main `summit` command supports LD scores, h²/rg, and one-environment G×E.
Joint generalized G×E estimation is available through Python, with a separate
planning and inspection command. PGS uses `summit-pgs`. Research functions in
`summit.context` are identified in their guide.
