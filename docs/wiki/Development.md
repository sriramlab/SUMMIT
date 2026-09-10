# Development

## Source layout

| Directory | Contents |
|---|---|
| `src/summit/ldscore` | Genome-wide, windowed, and G×E reference calculations |
| `src/summit/context` | Context coding, covariance models, and research APIs |
| `src/summit/prediction` | PGS fitting, model files, scoring, and calibration |
| `src/native` | Genotype decoding and numerical kernels |
| `example` | Synthetic input generator and runnable commands |
| `tests` | Unit, numerical, and integration tests |
| `docs/wiki` | User documentation, also suitable for GitHub Wiki |
| `reports` | Historical machine-readable benchmark records |

The generalized per-SNP and sample-probe contextual estimators have different
randomizations and summary formats. Their distinction and fixed SNP-block
deletion definitions are in [Methods](Methods.md).

## Tests

From a development installation:

```bash
python -m pytest -q
```

Some tests require particular native build variants or optional plotting/R
software. The workflow in `.github/workflows/ldscore-tests.yml` builds against
OpenBLAS before running the core suite.

For a separate PGS checkout with compiled modules:

```bash
python scripts/prediction/checkout.py test -q
```

The focused prediction/context/genotype regression selection is:

```bash
python scripts/prediction/checkout.py test \
  tests/test_prediction_core.py tests/test_prediction_io.py \
  tests/test_prediction_cli.py tests/test_prediction_selection.py \
  tests/test_prediction_edges.py tests/test_context_multienvironment.py \
  tests/test_context_fixed.py tests/test_context_fit.py \
  tests/test_generalized_gxe_native.py tests/test_pgen_gwld.py \
  -q -k 'not cli_dispatch and not end_to_end'
```

Build `gxeldcore`, `gwldcore`, and `winldcore` for that selection. Dense numerical
comparisons check independent equations; stored fixtures are synthetic.

## Documentation

Keep the README short: purpose, main features, installation, and a first example.
Put commands, inputs, interpretation, and practical limits in the relevant guide.
Add mathematical details to [Methods](Methods.md) when needed to distinguish
estimators. Benchmark pages should report dimensions, settings, measured results,
and limits without extrapolating unmeasured workloads.

Edit wiki sources in `docs/wiki`. Old documentation paths link to the relevant
current guide so existing links remain useful. Historical build logs belong
with benchmark records rather than installation instructions.

Before publishing, run `python scripts/check_repository_content.py`. The
`--history` option also checks files reachable from local Git refs. The check
flags dataset files, participant-table rows, and credentials without displaying
matched values. Review the provenance of numerical fixtures and benchmark
records separately; a pattern scan alone cannot establish that data are synthetic.
