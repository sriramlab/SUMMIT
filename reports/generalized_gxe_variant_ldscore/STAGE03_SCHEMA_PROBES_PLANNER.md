# Stage 03: schema, probes, planner, and pass counters

Date: 2026-08-22

Base commit: `eb1b78381959a62f4efe18fd82fe9ba025c115e4`

Scope: identity/schema, global probe stream, work/memory planner, and ledger;
no genotype or LD-score kernel

## Result

**PASS.** The generalized estimator now has a distinct fail-closed V1 schema,
globally addressed variant probes with exact Python/native agreement, a planner
that charges every required resident object, and a first-class ledger that
admits only two ordered complete descriptor passes.

All 28 focused tests and the full 1,630-test repository suite pass.

## Files changed

- `src/summit/ldscore/generalized_gxe_variant.py`
  - canonical estimator, contract, jackknife, and probe constants;
  - pair/component axis serialization through `ContextPairIndex` and
    `ContextComponentIndex`;
  - full manifest and numeric-array validation;
  - global counter probe specification, generation, fingerprint, and native
    wrapper;
  - corrected leading-work and resident-memory planner; and
  - mutable two-pass execution ledger.
- `src/native/gxeldcore.cpp`
  - one globally addressed Rademacher primitive and build-info declarations;
  - no genotype decode, source, target, or scoring kernel.
- `tests/test_generalized_gxe_variant_contracts.py`
  - 28 focused schema, probe, planner, overflow, and ledger tests.
- this report.

The mature physical-block-keyed Philox function and its existing callers were
not changed.

## Canonical identities

```text
artifact kind       summit.generalized_gxe.variant_ldscore_reference
schema version      1
scientific contract generalized_gxe_variant_ldscore_v1
jackknife method    frozen_full_genome_variant_ldscore_delete_block_v1
probe algorithm     counter_global_variant_global_probe_v1
```

New writers and validators accept only those canonical values. The new loader
explicitly rejects `summit.context.reference`,
`summit.context.reference.v1`, and `summit.gxe.reference`. It also rejects the
old `block_local_ldscore_deletion` identifier. The old non-general loader and
its legacy constant remain unchanged, so legacy compatibility is confined to
the old artifact family.

## Schema coverage

The V1 manifest validator binds and reconstructs:

- ordered variant and sample counts/digests;
- ordered basis names/digest;
- fixed-effect digest, rank, and residual rank;
- diagonal-first pair table, full contextual serialization, and digest;
- annotation-major component table, serialization, and digest;
- ordered annotation names, digest, and positive FP64 masses;
- one contiguous block ID per retained variant and ordered unique labels;
- residual component order;
- global randomization identity and recomputed fingerprint;
- frozen row-deletion semantics and full same-person reuse;
- clean two-pass performance counters;
- source/native provenance; and
- required scientific diagnostics.

Loaded numeric arrays are required to be finite FP64 with exact shapes and
bound hashes. The validator checks:

```text
directed_numerator            [C,C]
symmetric_numerator           [C,C]
genetic_gram                  [C,C]
block_directed_numerator      [J,C,C]
block_annotation_mass         [J,K]
same_person                   [C,C]
deleted_genetic_gram          [J,C,C] optional
directional_ldscores          [M,P,C] optional
```

It independently recomputes symmetrization, residual-rank/mass Gram
normalization, full numerator and mass reconstruction from blocks, every cached
deleted Gram, and same-person symmetry. Shape, dtype, digest, pair order,
component order, probe fingerprint, estimator kind, and pass-count tampering
all fail closed in tests. Canonical JSON round-trip validation passes.

## Global probe identity

Each sign is a fixed uint64 SplitMix composition of:

```text
root seed
SHA-256-derived named stream key
global retained-variant index
global probe index
```

The function does not receive a physical genotype-block start, probe tile
start, or thread index. Both implementations return a Fortran-order FP64
Rademacher matrix in the caller's requested row/probe order.

Frozen example for root seed `20260822`, variant indices `[0,1,7,19]`, and
probe indices `[3,4,11]`:

```text
[[ 1, -1,  1],
 [-1, -1, -1],
 [ 1,  1,  1],
 [ 1,  1, -1]]
```

The metadata fingerprint for root seed `20260822`, offset 5, and count 11 is:

```text
variants [0,1,2,7,31]
probes   [5,6,8,12]
shape    [5,4]
dtype    int8
sha256   03dcce2fc63cf2d85ca84f8eabe9fb9cd9f1f53ba55d13ccf802fa55742224ad
```

Tests reconstruct the same full matrix from four variant blocks, four probe
chunks, duplicate/permuted requested rows, and native thread counts 1, 2, and
4. All comparisons are bit-exact. Native build info records:

```text
global_variant_probe_supported    true
global_variant_probe_algorithm    counter_global_variant_global_probe_v1
global_variant_probe_output_dtype float64
```

## Work formulas

The planner implements:

```text
P = Q(Q+1)/2
C = KP

pass 1 FLOPs = 2NMKB
pass 2 FLOPs = 2NMKQ^2B
total         = 2NMKB(1+Q^2)
reduction product terms = MKP^2B
```

At `N=300,000`, `M=1,000,000`, and `J=200`:

| Q | K | B | P | C | Pass 1 | Pass 2 | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 128 | 6 | 6 | `76,800,000,000,000` | `691,200,000,000,000` | `768,000,000,000,000` |
| 3 | 1 | 1024 | 6 | 6 | `614,400,000,000,000` | `5,529,600,000,000,000` | `6,144,000,000,000,000` |
| 4 | 8 | 128 | 10 | 80 | `614,400,000,000,000` | `9,830,400,000,000,000` | `10,444,800,000,000,000` |
| 4 | 8 | 1024 | 10 | 80 | `4,915,200,000,000,000` | `78,643,200,000,000,000` | `83,558,400,000,000,000` |

All arithmetic uses checked signed-64-bit products; invalid, negative, zero,
and overflowing configurations are rejected.

## Resident-array plans

The 1 TiB fixture plans admit precomputed RHS panels and `V=4096`. Sizes below
are exact bytes; peak includes 15% allocator headroom, 8 MiB per thread, and a
64 MiB telemetry/publication reserve.

| Q/K/B | Base sources | Context sources | Pass-2 RHS | Decoded block | Cross block | Same-person `a[c,i]` | Block DNUM | Peak bytes | Output bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3/1/128 | 307,200,000 | 921,600,000 | 2,764,800,000 | 9,830,400,000 | 37,748,736 | 14,400,000 | 57,600 | 16,564,334,080 | 288,000,000 |
| 3/1/1024 | 2,457,600,000 | 7,372,800,000 | 22,118,400,000 | 9,830,400,000 | 301,989,888 | 14,400,000 | 57,600 | 49,016,691,405 | 288,000,000 |
| 4/8/128 | 2,457,600,000 | 9,830,400,000 | 39,321,600,000 | 9,830,400,000 | 536,870,912 | 192,000,000 | 10,240,000 | 72,144,240,205 | 6,400,000,000 |
| 4/8/1024 | 19,660,800,000 | 78,643,200,000 | 314,572,800,000 | 9,830,400,000 | 4,294,967,296 | 192,000,000 | 10,240,000 | 491,923,331,047 | 6,400,000,000 |

The planner also charges pair-reduction scratch `8VP^2`, three same-person
small matrices, three aggregate matrices, block masses, and one output buffer.
The full directional output is reported separately and is never counted as a
resident in-memory requirement.

Equivalent peak/output sizes in GiB:

| Q/K/B | Peak GiB | Directional output GiB |
|---|---:|---:|
| 3/1/128 | 15.426738 | 0.268221 |
| 3/1/1024 | 45.650351 | 0.268221 |
| 4/8/128 | 67.189560 | 5.960464 |
| 4/8/1024 | 458.139303 | 5.960464 |

## Constrained-memory plan

For the primary `Q=3,K=1,B=128` fixture with an 8 GiB limit, PGEN metadata,
and eight threads, precomputed RHS is rejected. The admitted plan is:

```text
variant block width             2,048
RHS tile columns                128 of 1,152
pass-1 decoded blocks           489
pass-2 decoded blocks           489
peak resident bytes             7,573,496,116
peak resident GiB               7.053368
planned descriptor passes       2
planned variant visits          2,000,000
```

The main resident bytes are 307,200,000 base sources, 921,600,000 contextual
sources, 307,200,000 RHS tile, 4,915,200,000 decoded block, 2,097,152 cross
output, and 970,340,660 allocator headroom. Tile reduction changes decoded
block counts, not physical descriptor passes. A 1 GiB fixture cannot hold the
fixed source state plus minimum tiles and is rejected rather than assigned
extra passes.

BED and PGEN plans have identical work, memory, and two-pass declarations.
Only descriptor metadata differs: BED records `ceil(N/4)*M` estimated bytes per
pass, while PGEN leaves compressed descriptor bytes unknown until file
inspection.

## Two-pass ledger

`TwoPassLedger` records:

```text
planned/observed descriptor passes
planned/observed retained-variant visits
duplicate visits
pass-1/pass-2 decoded blocks
retries, repairs, fallbacks, and integrity failures
```

It requires pass 1 then pass 2, contiguous full-axis block coverage, no active
pass at publication, exactly `2M` visits, and zero duplicate/retry/fallback/
integrity-failure counts for clean completion. A fixture with an overlapping
decoded block records the duplicate visit and is rejected.

## Validation

Focused source-only run before rebuilding the extension:

```text
27 passed, 1 skipped in 0.92s
```

The skip was specifically the absent new native binding. Fresh Release build
and final focused run:

```bash
PYTHONPATH=/tmp/summit-generalized-stage03-release.nkbjxd/install \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q \
  tests/test_generalized_gxe_variant_contracts.py
```

```text
28 passed in 0.91s
```

Full regression:

```bash
PYTHONPATH=/tmp/summit-generalized-stage03-release.nkbjxd/install \
SUMMIT_STAGE6_TEST_INSTALL=/tmp/summit-generalized-stage03-release.nkbjxd/install \
  /home/bronsonj/anaconda3/envs/summit/bin/python -m pytest -q tests
```

```text
1630 passed, 5 skipped, 1 xpassed in 131.72s
```

Static checks passed:

```text
Python py_compile
100-character Python line scan
git diff --check
```

The qualified native module SHA-256 is
`ffbd17e3260a6e047f96081fd6c332c35b360d140bb85d901cc5c6be287e3b11`.
It embeds source commit `eb1b78381959a62f4efe18fd82fe9ba025c115e4` and
staged-tree SHA-256
`35a387fe19fa21312c51f4c03e96329b8d2d3870c2f4c9cd82f84159f5741fdf`.

## Stop gate

Stage 03 passes. Global probes are fixed and native-exact; schemas reject the
other estimator families; planner arithmetic and overflow checks pass; and
every admitted normal plan declares exactly two physical passes. Genotype
kernels may begin only in Stage 04 after this focused commit.
