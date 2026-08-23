# SUMMIT coding-agent guardrails

## Generalized G×E LD scores

Before editing generalized contextual/G×E reference code, read:

- `docs/generalized_gxe_variant_ldscore_contract.md`
- `docs/architecture/ADR-generalized-gxe-variant-ldscore.md`
- `docs/context_native/scientific_contract_v1.md`

Two different randomized estimators exist and must not be conflated.

### Production generalized per-variant LD-score estimator

- uses **variant-axis** probes `Xi in R^{M x B}`;
- completes global source sketches in pass 1;
- scores every target variant against those completed sources in pass 2;
- emits or reduces fixed full-genome per-variant directional LD scores;
- uses exactly two complete reference-genotype traversals; and
- constructs SNP-block jackknife replicates by deleting target SNP rows from
  fixed full-genome LD-score sums. It does not recompute retained SNP LD scores.

Canonical jackknife method:

```text
frozen_full_genome_variant_ldscore_delete_block_v1
```

### Separate sample-probe contextual covariance estimator

The stable contextual reference under `src/summit/context/reference_v1.py` and
`src/native/contextual_streamed_reference_v1.inc` uses **sample-axis** probes to
estimate an aggregate kernel Gram and grouped action numerators. It is not the
per-variant generalized LD-score estimator and must retain a distinct artifact
identity and command path.

### Non-negotiable feature convention

```math
F_q=P\operatorname{diag}(\phi_q)G
```

uses one sealed genotype scale shared across contexts. Do not import the mature
non-general estimator's separate post-projection X/W column normalization.

### Reuse policy

Reuse the mature non-general descriptor/decode/imputation/genotype-scale,
protected NN/TN, packed/vendor-BLAS, threading, NUMA, telemetry, and artifact
machinery. Do not copy its hard-coded X/W scientific layout or create a second
genotype decoder.
