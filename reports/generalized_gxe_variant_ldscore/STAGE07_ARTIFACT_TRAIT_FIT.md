# Stage 07 artifact, trait, fit, and jackknife integration

## Decision

**PASS.** The generalized variant-probe reference now has a distinct closed V1
artifact, atomic publication and strict loading, an adapter to the unchanged
contextual trait artifact, compact full/delete-block fit assembly, and a
standalone planning/inspection CLI. It cannot be cross-loaded as the contextual
sample-probe reference. Every tested jackknife replicate subtracts fixed target
rows from the full-genome directed numerator, reuses the full same-person
matrix, and performs no genotype or per-variant-panel work.

The isolated authoritative-package mismatch in `CODEX_LAUNCH_PROMPT.md` remains
the non-scientific, non-executable packaging discrepancy documented and
explicitly waived in Stage 00. The package mathematical oracle, scientific
documents, executable scripts, source archive, and all other manifest entries
passed. It is not a Stage 07 estimator or qualification failure and was not
silently repaired.

## Settled candidate

- schema/publication commit:
  `58744beab9ee6aeb801458c2778bba2d50ff54d1`;
- trait/fit/CLI commit:
  `2f670b3c304ca5160e4330ede727cc5c17346c23`;
- frozen legacy CLI-metadata compatibility commit:
  `4e31beb1b421df2391170de73c78bf5fa5b0052b`;
- final Release root:
  `/tmp/summit-generalized-stage07-final.u5rTiP`;
- embedded source-tree SHA-256:
  `fa69e9f884af12526584da1b7ab17e870ca03ae50db73242cd60a070f280284f`;
- final Release `gxeldcore` SHA-256:
  `eac81ffb83bc48039fac4aa267db943cc148e00fac01ea38379c048ed655a9a6`.

The clean build reports GNU 12.2, `Release`, OpenBLAS,
`-march=native`, sanitizer mode `none`, protected GEMM integrity and checksums
enabled, and global variant-probe support enabled.

## Artifact boundary

The new artifact has kind
`summit.generalized_gxe.variant_ldscore_reference`, schema version 1, and
suffix `.generalized-gxe-variant-ldscore-v1.npz`. For example, publishing to
`reference` resolves to
`reference.generalized-gxe-variant-ldscore-v1.npz`. Its manifest declares:

```text
scientific contract  generalized_gxe_variant_ldscore_v1
estimator family     variant_probe_two_pass_per_variant_ldscore
jackknife method     frozen_full_genome_variant_ldscore_delete_block_v1
probe algorithm      counter_global_variant_global_probe_v1
```

`build_generalized_gxe_variant_reference_v1` accepts only complete validated
native outputs. `write_generalized_gxe_variant_reference_v1` uses the mature
atomic no-replace publication primitives; the final path does not expose a
partial artifact. `load_generalized_gxe_variant_reference_v1` validates the
closed manifest and every stored array before returning owned, C-contiguous,
read-only arrays.

The closed payload binds the variant and retained-variant order, allele order,
sample identity, basis specification and calibration, fixed-effect
specification/rank, pair/component tables, annotations, block partition, common
genotype scale, global probe stream, exact two-pass ledger, integrity and
performance evidence, provenance, and terminal status. It stores full `DNUM`,
block `BDNUM`, block annotation masses, the symmetric numerator, full genetic
Gram, and full same-person matrix. The directional per-variant panel and cached
delete-block Grams are optional; omitting either does not affect fit.

The contextual loader rejects this kind and the generalized loader rejects the
contextual kind. Corrupted payloads, a mismatched suffix, malformed or
noncanonical metadata, hash changes, incomplete pass ledgers, and partial or
pre-existing publication targets fail closed.

## Trait and fit adapter

The existing contextual trait executor and artifact are reused unchanged. The
new adapter requires exact equality of:

- the sealed genotype scale plan;
- variant/allele order and retained-variant order;
- basis specification and calibration;
- fixed-effect specification;
- pair/component map and residual-component order;
- annotation names, map, masses, and per-block masses;
- jackknife group map, group sequence, and group variant counts; and
- dimensions `M`, `Q`, `K`, `C`, and `J`.

The shared normal-equation assembly was extracted into
`assemble_contextual_normal_equations_from_moments_v1`; the prior contextual
entry point delegates to it. The generalized adapter supplies its compact
reference moments and the existing trait moments to this shared assembly. It
reuses the established population transfer and raw symmetric rank-checked
solver. There is no LD-score correction of trait moments, ridge,
pseudoinverse, coefficient clipping, or PSD replacement.

For deletion block `g`, the reference adapter computes
`DNUM - BDNUM[g]`, subtracts the block annotation masses, symmetrizes and
normalizes once, and reuses the full same-person matrix byte-for-byte. It never
recomputes retained-SNP LD scores. Every fit and jackknife replicate consumes
compact artifact summaries only; the optional per-variant panel is not
accessed. Changing `J` therefore changes stored compact reductions and fit
postprocessing, not the two-pass traversal or genotype GEMM plan.

## CLI/API boundary

The separate command surface is available as:

```bash
python -m summit.ldscore.generalized_gxe_variant_cli plan ...
python -m summit.ldscore.generalized_gxe_variant_cli inspect ...
```

Help names the command `summit-generalized-gxe-variant-ldscore`, identifies it
as the variant-probe exactly-two-pass estimator, and states that it is not the
sample-probe contextual action estimator. `plan` performs dry-run work/memory
planning; `inspect` fully validates a closed artifact and reports its contract,
jackknife method, axes, ledger, and panel disposition.

A `pyproject.toml` console-script entry was deliberately not retained. The
first full-suite run showed that changing package entry-point metadata violates
the repository's frozen legacy surface guard. Removing only that metadata
change restored compatibility while preserving the distinct module CLI and all
new APIs.

## End-to-end numerical evidence

The fit fixture binds a loaded existing contextual trait artifact to the new
reference kind and exercises three unequal delete blocks. Comparisons used the
direct dense symmetric solve and independent frozen-row assembly:

| Comparison | Maximum absolute error |
|---|---:|
| Full fitted coefficients vs direct dense solve | `5.5511151231257827e-17` |
| All delete-block coefficients vs direct compact solves | `5.5511151231257827e-17` |
| All delete-block reference Grams vs direct `DNUM-BDNUM` assembly | `0.0` |
| Delete-block same-person matrix vs full same-person matrix | `0.0` |

Tests also verify exact `sum_g BDNUM[g] == DNUM` and reconstruction of full
annotation masses from block masses, native-result-to-artifact construction,
round-trip equality and immutability, optional panel omission, and explicit
rank-deficiency failure. Compatibility mutations covering scale, basis,
calibration, component order, fixed effects, variant identities, annotation
map/mass, and group map/order/counts are all rejected.

## Qualification

| Scope | Result |
|---|---:|
| Artifact schema/publication focused scope | **37 passed, 1 skipped** |
| New native/artifact/fit/CLI common scope | **54 passed** |
| Generalized Stage 02–07 scope | **112 passed** |
| Mature contextual/non-general artifact regressions | **82 passed** |
| Existing contextual fit regression | **16 passed** |
| Final clean full repository Release suite | **1699 passed, 6 skipped, 1 XPASS in 146.23 s** |

The initial full run before commit `4e31beb` had one failure in the frozen
legacy-surface hash and no estimator failure. The final clean build and suite
above include the compatibility correction. No new Stage 07 test skipped.
The six skips and one XPASS are existing repository-level outcomes.

## Stop-gate conclusion

The new generalized variant-LD-score artifact can be generated from the native
two-pass result, atomically published, strictly loaded, bound to a compatible
trait artifact, fit, jackknifed from frozen target-row summaries, and inspected
without ambiguity with the contextual sample-probe family. Full and deleted
fits match their direct dense oracles, the same-person term is reused exactly,
cross-family and compatibility failures are closed, mature regressions pass,
and the full clean Release suite passes. Stage 08 may begin from this settled
candidate and report.
