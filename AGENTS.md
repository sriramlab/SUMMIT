# SUMMIT coding-agent guardrails

## Generalized G×E LD scores

Before editing generalized contextual/G×E reference code, read:

- `docs/generalized_gxe_variant_ldscore_contract.md`
- `docs/architecture/ADR-generalized-gxe-variant-ldscore.md`

Two different randomized estimators exist and must not be conflated.

### Production generalized per-variant LD-score estimator

- uses **variant-axis** probes `Xi in R^{M x B}`;
- completes global source sketches in pass 1;
- scores every target variant against those completed sources in pass 2;
- emits fixed full-genome per-variant directional LD scores;
- uses exactly two complete reference-genotype traversals; and
- accepts no block IDs, block count, or jackknife argument.

### Definitive jackknife boundary

"No jackknife" applies only to estimation of generalized per-SNP reference LD
scores and per-SNP trait statistics. After the genome-wide per-SNP rows exist,
`--njack` assigns target SNPs to blocks and reduces those fixed rows into full
and delete-block normal equations. Each replicate drops the deleted target-SNP
contribution, rescales by retained annotation mass, and reuses all other
quantities. It does not recompute source sketches or any retained SNP's LD
score. Standard errors and Wald statistics come from these post-hoc
normal-equation replicates.

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

Reuse the mature non-general descriptor/decode/imputation/genotype-scale and
numerical kernels where useful. Do not import its hard-coded X/W scientific
layout, artifact identity checks, or create a second genotype decoder.
