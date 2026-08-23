# GxE schema and workflow migration

## Additive-compatible metadata changes

Newly written references, phenotype moments, and score bundles carry:

```json
{
  "feature_convention_version": 1,
  "feature_convention": "standardized_projected"
}
```

The other allowed value is `raw_projected`. Existing schema-v3 bundles without
these fields remain readable: `kernel_mode=standardized` maps to
`standardized_projected`, and `kernel_mode=genie` maps to `raw_projected`.
When both legacy and canonical fields exist, disagreement is an error. No
numeric conversion between conventions is attempted.

Fit JSON gains additive diagnostic fields: `solve_method`,
`normal_symmetry_error`, `cauchy_schwarz_max_violation`, `component_influence`,
`identifiable`, and `normal_equation_diagnostics`. Existing core coefficient,
trace, contribution, singular-value, rank, condition, and jackknife fields are
unchanged.

Population-transfer diagnostics identify the two scaling factors, reference
and study `N`/rank, extrapolation, and finite-probe same-person symmetry/eigenvalue
state. These additions do not change stored reference matrix values.

The schema-v1 multi-environment batch manifest now also records a bounded GEMM
shape histogram, total modeled FLOPs, process-lifetime peak RSS at publication,
and the RSS measurement scope. Existing batch fields remain unchanged.

Direct-native compile provenance now includes `gemm_execution_mode`,
`gemm_integrity_enabled`, `blas_runtime_isolation`, the fixed runtime thread
count/layer, and the private archive SHA-256. Private builds emit
`serialized_fixed_private_openblas`; shared builds retain the checked
`serialized_fixed_shared_openblas` fallback. These are additive metadata inside
the existing `compile_options` object and do not change scientific identity or
matrix values. The direct backend rejects OpenBLAS versions older than 0.3.31.

## CLI names

Preferred CLI names are:

```text
--gxe-kernel-mode standardized_projected
--gxe-kernel-mode raw_projected
```

The legacy values `standardized` and `genie` remain aliases. Output metadata is
canonical regardless of which accepted spelling was supplied.

## Retired persisted construction

The hidden new-write controls `_gxe-build-cache`, `_gxe-feature-cache`,
`_gxe-reference-shard`, and `_gxe-merge-shards` have been removed from the CLI.
The historical Hoffman cache/cache-attestation/shard/merge worker tasks now fail
with an explicit retirement error before executing their plan. Legacy readers
and semantic validators remain for auditing already sealed cache-bound bundles.

New construction must use either the streaming one-environment path or the
fused in-memory multi-environment path. Neither writes a feature matrix or
randomized source/target sketch to disk. This is a workflow migration, not an
automatic conversion of old shard manifests.
