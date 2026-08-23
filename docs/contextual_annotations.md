# Disjoint annotation contextual covariance

The private `summit.context` development API supports annotation-specific
contextual covariance for an ordered, exactly-one partition of retained
variants. The first supported mode is deliberately strict: every annotation
entry is binary, every retained variant belongs to exactly one bin, every bin
has positive mass, and every frozen delete-group replicate retains positive
mass in every bin. Overlapping or fractional annotations are not given
standalone covariance-matrix interpretations.

## Partition identity

`build_disjoint_annotation_partition` binds all of the following into one
partition identity:

- ordered bin names and definitions;
- explicit lower-inclusive, upper-exclusive boundaries with the final upper
  boundary closed;
- the ordered, allele-aware variant identity supplied by the caller;
- exact binary membership;
- the annotation source declaration;
- the complete frozen LOO-group sequence.

The manifest records annotation masses, group-by-annotation masses, group
variant counts, deletion leverage, effective group counts, and whether every
deletion remains estimable. Equal-weight jackknife groups must differ in total
size by at most one. Reference and study artifacts must have identical
partition, component, variant, basis, scaling, and grouping identities.

`build_maf_ld_partition` applies fixed numeric MAF and LD-score edges. It does
not estimate quantiles separately in the study cohort. Values outside the
frozen range, empty bins, uncovered variants, multiple membership, nonfinite
values, and reordered definitions fail closed. The example configuration is
[`example/context/maf_ld_partition.json`](../example/context/maf_ld_partition.json).

## Losslessly grouped approximate deletion

The summary-only jackknife remains the declared SNP-moment approximation. For
the frozen groups, Stage 08 changes its storage rather than its algebra. During
reference construction it accumulates

```text
group annotation masses       J x K
group Gram numerators         J x Pg x Pg
```

and during trait construction it accumulates

```text
group RHS numerators          J x Pg
group trace numerators        J x Pg
group genetic-residual terms  J x Pg x H.
```

Here `Pg = K Q(Q+1)/2`. A deletion subtracts exactly one group aggregate and
renormalizes by the retained annotation masses. The same full-reference
same-person matrix is reused, matching the existing approximate-deletion
contract. Full and every declared deleted-group equation are numerically
identical to retaining the corresponding SNP-level numerators. The grouped
artifacts contain no variant-length scientific arrays and inference requires
all frozen groups; selecting only a subset is not labeled an equal-group
jackknife.

Grouping only bounds the retained deletion payload. The current NumPy
reference builder still materializes genotype-dependent feature and probe
intermediates, so it remains a small-fixture development backend rather than a
biobank-scale implementation.

## Coefficients, totals, and contrasts

Each genetic kernel is normalized by its own annotation mass `M_k`. Therefore
the fitted `Omega_k` is the total covariance contribution assigned to bin `k`.
It is not automatically a per-SNP enrichment. The explicit
`per_annotation_mass` scale reports `Omega_k / M_k`; each deleted-group
replicate uses its own retained mass.

`fit_annotation_context_model` reports raw annotation matrices, the joint
coefficient jackknife covariance, rank and condition diagnostics, and optional
per-annotation PSD projections. Raw estimates are never replaced by their PSD
versions. `derive_annotation_contrast` computes a named left-minus-right
contrast from the full joint deleted-group replicates. Its trace uncertainty
uses deletion-specific traces and masses. `derive_annotation_total` sums the
bin contributions without an additional mass factor. A sum on the per-mass
scale is labeled as such and is not called the total genome contribution.

## Resource planning

`estimate_annotation_resources` is a dimension-only planner. Without a pilot
normal matrix, condition is reported as unknown; increasing the probe count is
never presented as a remedy for structural collinearity. The planned streamed
base-Gram schedule retains `Q` source sketches of shape `M x b`, streams one
source through `K` target sketches of shape `N x b`, and retains the component
action tile of shape `N x b x Pg`. Losslessly grouped Gram numerators also
require per-component source contractions and group reductions; Stage 09 must
profile and implement that pass before this is treated as a complete
production operator. The estimator separately reports:

- the planned block-streamed operator peak under a hard memory cap;
- the much larger current Python development peak;
- SNP-level and grouped retained sufficient-statistic bytes;
- source, annotation-target, same-person, and trait product counts.

For reference deletion contributions alone, storage falls from
`8 M Pg^2` bytes to `8 J Pg^2` bytes. This does not remove the current Python
builder's transient `Q N M` feature panel or its exact/randomized development
intermediates. Use `scripts/context/dry_run_context_resources.py` before any
large run and treat a dimension-only result as a resource estimate, not a
conditioning certificate.

## Development example

```python
partition = build_maf_ld_partition(
    maf,
    ld_score,
    maf_edges=(0.0, 0.05, 0.20, 0.50),
    ld_edges=(0.0, 5.0, 15.0, 50.0, 1_000_000.0),
    variant_hash=ordered_allele_aware_variant_hash,
    loo_groups=balanced_block_labels,
    source="external_reference_panel_maf_ld_v1",
)

reference = build_grouped_context_reference(
    partition=partition,
    genotype=reference_genotype,
    basis=reference_basis,
    projector=reference_projector,
    # remaining generic reference arguments, including raw genotype scaling
)
summary = build_grouped_context_trait_summary(
    partition=partition,
    genotype=study_genotype,
    phenotype=study_phenotype,
    basis=study_basis,
    projector=study_projector,
    # remaining generic trait-summary arguments
)
fit = fit_annotation_context_model(reference, summary, partition=partition)
contrast = derive_annotation_contrast(
    fit,
    "maf0__ld0",
    "maf1__ld0",
    reference=reference,
    summary=summary,
    scale="total_component",
)
```

## Scope

This stage is a correctness-first Python implementation for quantitative
traits, a fixed context basis, unrelated individuals, common complete samples,
and strict disjoint partitions. The practical development ceiling is chosen
from the supplied K/Q/B benchmark and memory dry-run rather than silently
allocating beyond the requested cap. Missingness, overlapping-annotation
interpretation, production file adapters, protected native operators, and
public CLI integration remain outside this module and are Stage 09 work.
