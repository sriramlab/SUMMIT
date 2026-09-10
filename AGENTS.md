# SUMMIT development rules

Before changing generalized contextual/G×E reference code, read
[Methods](docs/wiki/Methods.md) and the relevant implementation and tests.

## Generalized per-SNP reference estimator

- Use variant-axis probes with shape M×B.
- Complete all global source sketches in pass 1, then score targets in pass 2.
- Estimate fixed full-genome per-SNP directional LD scores in exactly two
  complete reference-genotype traversals.
- Keep block IDs and jackknife counts out of reference estimation.

After the per-SNP reference and trait rows exist, `--njack` groups target rows
and forms full/delete-block normal equations. Each replicate subtracts the
block contribution, rescales retained annotation mass, and reuses the other
quantities. It never recomputes retained-SNP LD scores or source sketches.
The method name is `frozen_full_genome_variant_ldscore_delete_block_v1`.

The sample-probe contextual estimator in `src/summit/context/reference_v1.py`
and `src/native/contextual_streamed_reference_v1.inc` estimates aggregate
kernel actions and grouped numerators. Keep its formats and calculation
separate from the generalized per-SNP estimator.

## Feature definitions and reuse

Generalized features are `F_q = P diag(phi_q) G`, using one genotype scale
across contexts. Preserve the order of multiplication and projection. Do not
add context-specific post-projection column normalization.

Reuse existing genotype descriptors, decoding, imputation, scaling, and
numerical kernels. Keep the scientific layout specific to the estimator;
do not add an independent genotype decoder.
