# Development

Install from an activated SUMMIT environment:

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -q
```

Rebuild after changing native code. Use the same BLAS configuration for the
build and runtime. Some tests require a particular native build or optional R
and plotting packages. GitHub CI builds against OpenBLAS.

Source modules are grouped by analysis in `src/summit`; genotype decoding and
numerical kernels are in `src/native`. Read [AGENTS.md](AGENTS.md) and the
[methods](docs/wiki/Methods.md) before changing reference estimators.

Edit user guides in `docs/wiki`; these are also the GitHub Wiki sources.
Document inputs, commands, outputs, and scientific assumptions in the relevant
guide. Keep development logs and machine-specific job histories out of it.

Before publishing, run `python scripts/check_repository_content.py`.
Tests and examples must use synthetic or published aggregate data.
