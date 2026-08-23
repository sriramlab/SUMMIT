# Stage 00: baseline and live-repository audit

Date: 2026-08-21  
Repository: `/home/bronsonj/SUMMIT`  
Authoritative package: `/home/bronsonj/SUMMIT_generalized_GxE_ldscore`  
Scope: audit and qualification only; no estimator implementation changes

## Stop-gate result

**Original Stage 00 result: FAIL / HOLD because of package integrity.**

**Post-audit disposition: PASS BY EXPLICIT USER WAIVER; Stage 01 may begin.**

The live SUMMIT baseline is buildable and its tested behavior is understood, but
the required authoritative-package integrity check does not pass byte-for-byte. The only
reported manifest failure is `CODEX_LAUNCH_PROMPT.md`:

```text
expected  f0fcec724f5eb9f363327b169200e89c6de586713799d16348cc84b16f8a91fc
observed  a3000ed8d9abda0eda92a81c208a29ad15697ce1366128ee09ac77391c4be160
```

`scripts/check_package.py` otherwise passed its 16 snapshot anchors, required
43-file inventory, guard-token checks, source-archive check, Python compilation,
and patch apply/check exercise before raising the manifest-hash assertion. A
direct `sha256sum -c MANIFEST.sha256` check also identified only this file as a
mismatch. The package has not been edited.

### User waiver recorded after the original audit

On 2026-08-21, the user explicitly directed that this isolated discrepancy be
treated as non-blocking and that staged implementation continue. A brief
follow-up check established:

- `sha256sum -c MANIFEST.sha256` still reports exactly one mismatch;
- every executable script, mathematical test, scientific document, patch, and
  source archive listed in the manifest passes;
- the mismatching object is the Markdown-only launch prompt, not executable
  estimator code or a mathematical-contract document;
- the observed launch prompt restates the same decisive contract and staged
  workflow used for this audit; and
- its modification timestamp (`2026-08-21 23:03:10 -0700`) is later than the
  manifest and package files stamped at `2026-08-21 00:00:00 -0700`.

These observations do not prove what byte-level edit produced the hash change.
They do constrain the discrepancy to one orchestration document, while all
scientific and executable integrity checks remain valid. Under the user's
explicit risk acceptance, the packaging hash is waived as a Stage 01 blocker.
It remains recorded here rather than silently normalized or repaired.

The mathematical oracle passed when run independently. Thus the hold concerns
package integrity, not a mathematical-oracle failure.

## Scope and governing contract

The requested estimator is a new, per-variant generalized G×E LD-score path. Its
governing properties are:

- variant-axis probes;
- directional full-genome per-variant output with logical shape `[M, P, C]`;
- `F_q = P diag(phi_q) G` under one common sealed genotype scale;
- one complete global-source genotype pass, a hard barrier, and one complete
  target-scoring genotype pass;
- exactly two complete reference-genotype traversals;
- global probe counters invariant to block and tile partitioning;
- SNP-block jackknife deletion by subtracting target SNP rows from fixed
  full-genome LD-score sums;
- no recomputation of retained-SNP LD scores and no exact delete-kernel
  requirement;
- reuse of the full same-person matrix across all jackknife replicates; and
- no clipping, ridge substitution, or pseudoinverse fallback in the stated
  scientific contract.

The existing contextual covariance/action implementation uses sample-axis
probes and approximate fixed-probe group actions. It is a separate estimator and
is not a valid extension point for relabeling as the requested variant LD-score
estimator.

The mature non-general G×E estimator is the systems template. Its genotype
descriptor/decode/imputation/common-scale machinery, protected NN/TN kernels,
packed and vendor BLAS support, threading, NUMA, telemetry, and publication
machinery are reusable. Its hard-coded X/W feature layout and separate
post-projection X/W scales are not part of the new scientific contract.

## Authoritative material reviewed

The following files were read in the prescribed order and in full:

1. `README.md`
2. `SUPERSEDING_DECISIONS.md`
3. `SCIENTIFIC_CONTRACT_GENERALIZED_VARIANT_LDSCORE_V1.md`
4. `MATHEMATICAL_DERIVATION_AND_PROOFS.md`
5. `NON_GENERAL_GXE_AUDIT.md`
6. `TWO_PASS_PRODUCTION_ARCHITECTURE.md`
7. `IMPLEMENTATION_PSEUDOCODE.md`
8. `REUSE_MAP_AND_PATCH_PLAN.md`
9. `ARTIFACT_SCHEMA_V1.md`
10. `PERFORMANCE_MODEL.md`
11. `VALIDATION_AND_QUALIFICATION.md`
12. `IMPLEMENTATION_ACCEPTANCE_CHECKLIST.md`
13. `REPOSITORY_RECORDING_PLAN.md`
14. `PACKAGE_BUILD_REPORT.md`
15. `CODEX_PROMPT_INDEX.md`
16. `CODEX_MASTER_PROMPT.md`

`CODEX_LAUNCH_PROMPT.md`, `codex_prompts/00_BASELINE_AND_LIVE_AUDIT.md`,
and both package-check scripts were also inspected. No later-stage prompt was
executed.

## Repository identity and live state

Initial repository state was clean.

```text
branch       codex/contextual-stage7-release-audit
HEAD         81762da159d240d71a2d72f5f1f54a1c50865601
HEAD tree    f262e87f51b4401789f54723550869b30914ac42
```

The package-recorded implementation snapshot is commit
`69203355d558792981993e2ac8fceb6851e41418`. The diff from that commit to live HEAD contains nine Stage 7
report/evidence files and 1,213 inserted lines. A scoped diff over `src`,
`tests`, `pyproject.toml`, and `CMakeLists.txt` is empty. Therefore, as observed,
the live estimator source and test baseline match that implementation snapshot;
the later commit records audit material only.

No new branch was created because the repository was already on a dedicated
feature/audit branch. No commit was made in Stage 00.

## Package integrity and mathematical oracle

Requested entry points:

```bash
cd /home/bronsonj/SUMMIT_generalized_GxE_ldscore
python scripts/check_package.py
python scripts/run_math_checks.py
```

Observed package-check result:

```text
PASS: 16 snapshot anchors
PASS: required 43-file inventory and guard checks
PASS: source archive SHA-256
PASS: Python compilation
PASS: patch applies/checks
FAIL: manifest hash mismatch: CODEX_LAUNCH_PROMPT.md
```

The source archive SHA-256 reported by the check was:

```text
18c77effc86e152d758e6b9ad20127c93d44c43b064f61f776ff9a05ca9c6a3c
```

Independent mathematical-oracle result:

```text
10 passed in 4.99s
exact identity maximum absolute error      7.816e-14
randomized relative Frobenius error         0.00493398
same-person relative Frobenius error        0.0103367
directional pairs                           6
directional components                      6
```

## Build and runtime environment

Observed host and toolchain:

| Item | Observed value |
|---|---|
| Host | `Tabla` |
| Kernel | Linux 4.19 |
| CPU | AMD EPYC 7501, 2 sockets, 64 physical cores, 128 logical CPUs |
| NUMA | 8 nodes, CPUs 0-127 available; `numactl` command absent |
| Memory | approximately 1 TiB RAM, 8 GiB swap |
| Python | 3.12.12 |
| CMake | 3.25.1 |
| C++ compiler | GNU 12.2.0 |
| NumPy | 2.3.5 |
| SciPy | 1.16.3 |
| pandas | 2.3.3 |
| pytest | 9.1.1 |
| pgenlib | 0.94.1 |
| nanobind | 2.12.0 |
| scikit-build-core | 0.11.6 |
| psutil | 7.1.3 |
| threadpoolctl | 3.6.0 |
| BLAS | conda OpenBLAS 0.3.34, Zen, pthreads, maximum 128 threads |
| OpenMP | conda `libgomp`, maximum 128 threads |

No relevant `OMP_*`, `OPENBLAS_*`, `BLIS_*`, or `MKL_*` setting was inherited
for the baseline runs. The fresh Release module resolved to conda OpenBLAS and
conda `libgomp`; `ldd` reported no missing dependency. Its RUNPATH was
`$ORIGIN:/home/bronsonj/anaconda3/envs/summit/lib`.

## Baseline qualification

All builds below were fresh out-of-tree builds. Their temporary roots were
retained through evidence capture and removed after this report was verified.

### Release build

Configuration, compilation, installation, and import passed. Build metadata
reported API 9, backend `gxeldcore_direct` 1.9, Release `-O3`, `-march=native`,
OpenMP enabled, protected GEMM/checksum enabled, source commit equal to live
HEAD, and source-tree SHA-256
`e1538be652e66ed2cba0c6b3beadd434dcd96c6873ddfbff51063a5f7c045b02`.

Installed module hashes:

```text
gxeldcore  f17d61bdfa9f208fc9737f2c2b23ab3b478c6d656614867c39a86ccdcc8984ad
gwld       5bdfa12c583424e89f6d0a1003ece4480c7911732ec92fe134f3f0b9495859c1
win        27a6f6ac9131bc274f39da30fa9bb03e0980876e5671ac50ddcd82919d1647e9
```

Full Python/native test suite:

```text
1587 passed, 6 skipped, 1 xpassed in 167.65s
```

One skip was an opt-in Stage 6 installed-artifact smoke test. Rerunning it with
the required `SUMMIT_STAGE6_TEST_INSTALL` setting passed:

```text
1 passed in 6.43s
```

Thus 1,588 distinct tests passed across the full run plus the opt-in rerun. The
five remaining skips were environment/backend qualifications: a non-BLIS
candidate, retired block-jackknife coverage, an absent private OpenBLAS archive,
and placement tests requiring pthread-BLIS/OpenMP-BLAS conditions not present in
this build. The XPASS was
`test_diagnostic_source_tt_vendor_telemetry_and_dense_oracle`, whose historical
xfail reason expected the older API 8 source-TT rejection; the current API 9
behavior passed its assertions.

`ctest` reported no registered tests. This repository's native-extension
qualification is driven through pytest.

Focused scientific and integration audit:

```text
15 passed in 2.50s
```

This covered global and dense oracles, native source/target paths,
determinism, multi-environment no-jackknife pass planning, same-person and
directional identities, deletion logic, CLI workflow, contextual action
oracles, and compact artifact handling.

The explicit reference-transfer fixture passed:

```text
1 passed in 0.22s
transfer median relative errors: 2.1716%, 3.1353%, 2.2305%
naive median relative errors:    9.2679%, 86.4641%, 23.5139%
```

### ASAN plus UBSAN build

Fresh portable `-O1` build with AddressSanitizer and UndefinedBehaviorSanitizer
enabled passed its supported scope. The installed extension SHA-256 was
`4126d78923f2c819b6de745b860c56d12db4ea04fcbec483230677970d920ee4`.
Runs used the GCC 12 ASAN runtime preloaded ahead of conda
`libstdc++`, `halt_on_error`, `abort_on_error`, and `detect_leaks=0`.

```text
903 passed, 1 skipped in 138.85s
opt-in installed-artifact smoke: 1 passed in 11.35s
```

All 904 distinct supported tests therefore passed. There was no ASAN or UBSAN
diagnostic. Leak checking was explicitly disabled, so this run makes no leak
qualification claim.

### UBSAN-only build

Fresh portable `-O1` UBSAN build passed all 904 tests in its supported scope:

```text
904 passed in 96.17s
```

The installed extension SHA-256 was
`8f79f78acdc534e2ba4d98b3d1181f45e0b145271164f990b112fe8e6d76dfc4`.
No undefined-behavior diagnostic was observed.

## Mature non-general G×E estimator: live audit

The mature implementation is in `src/summit/gwe_ldscore.py` and
`src/summit/_native/gxe_multi.py`, with native implementation in
`src/native/gxeldcore.cpp`.

Observed scientific flow in `gwe_ldscore.py`:

- estimator and per-variant output definitions: lines 1576-1604;
- effective-rank bookkeeping (`p_eff`, `N_eff`, `df`): lines 2011-2018;
- covariate projection and centering: lines 2916-2920;
- genotype-scale policy selection: lines 1786-1810;
- separate post-projection X/W scaling: lines 2922-2942 and 3075-3234;
- variant-probe generation: lines 3236-3257;
- source action construction: lines 3315-3333;
- target left-score construction: lines 3335-3351;
- same-person construction: lines 3366-3468;
- global source loop, U-statistic assembly, barrier-like finalization, target
  loop, and four directional outputs: lines 5115-5209;
- score publication: lines 5273-5365;
- no-jackknife policy: lines 1964-1968;
- reference manifest and random-algorithm record: lines 4292-4398.

The probe generator has logical shape `L x v`, hence uses variant-axis probes.
Its counter seed is `_make_seed(root_seed, blk_start, probe_id)`. Probe-column
tiling preserves a logical probe column, but changing physical variant-block
starts changes random signs. This is incompatible with the new contract's
global-counter invariance and must be replaced at the scientific seam.

The mature implementation emits the four directional products XX, XW, WX, and
WW with `M x K` panels. It computes effective covariate rank and uses a common
projector, but applies distinct feature-specific post-projection normalizations.
Those X/W scales cannot be copied into the common-scale generalized estimator.

Observed multi-environment/native flow:

- random-key table keyed by physical block start: `_native/gxe_multi.py` lines
  2575-2588;
- pass planner: lines 2665-2670, 2758-2762, and 6972-6986;
- direct-context descriptor/decode integration: native lines 7075 and 8564;
- source dense and packed kernels: native lines 10760 and 10994;
- target dense and packed kernels: native lines 11690 and 11913;
- multi-environment direct context and run entry: native lines 12950 and 13487;
- source/features, U-statistic finalization, and target/output phases: native
  lines 13697-14064.

The native planner uses two fused passes only for one environment and one probe
chunk. Otherwise it reports and performs `1 + 2 * (environment products) *
(probe chunks)` genotype traversals. The Python path similarly performs one
feature pass plus two traversals for each probe tile. Therefore neither path
currently satisfies exactly two complete traversals for arbitrary generalized
`Q`, `P`, and tiling.

The mature same-person quantity is formed from completed source actions and
does not require an additional genotype traversal. This is a reusable
architectural property.

## Contextual covariance/action estimator identity

The contextual estimator is identified in `src/summit/context/reference_v1.py`:

- scientific policy and estimator identity: lines 400-410;
- fixed-probe approximate group deletion: lines 412-422;
- compact artifact statement: lines 1141-1148;
- persisted compact arrays: lines 91-98;
- schema and artifact-family constants: lines 3278 onward;
- sample-primary and variant same-person probe declarations: lines 3304 onward.

Its primary contextual actions use sample-axis probes (`N x K`), as shown in
`src/summit/context/oracle.py` lines 422-479. The native pass planner in
`src/native/gxeldcore.cpp` lines 3068-3234 allocates source/action passes by
sample-resident group and same-person passes by variant tile. Its phases begin
at source 4966, action 5134, Gram 5492, group 6284, same-person 6403, and execute
7473.

The contextual artifact stores compact Gram, same-person, grouped-numerator,
mass, and count arrays; it does not store a per-target-variant LD-score axis.
Its CLI exposure is not a normal public command in the current command table.
Consequently this path is scientifically and operationally separate from the
requested full-genome per-variant directional estimator.

## Existing jackknife inference path

Observed inference code is in `src/summit/gxe_score.py` and
`src/summit/gxe.py`:

- loader and exact/local reference modes: `gxe_score.py` lines 387-447;
- jackknife preparation: `gxe.py` lines 993 onward;
- local-mode within-score rejection: lines 1044-1048;
- full/group panel construction: lines 1078-1101;
- retained-panel subtraction: lines 1107-1129;
- inference test summary: lines 613-677.

Local deletion already subtracts target-block SNP rows from fixed full panels;
it does not recompute retained-SNP LD scores. Exact mode adds reverse/within
terms. The current mature generator, however, declares no jackknife and emits
no `BLOCK` column or top-level jackknife metadata, so there is no ordinary
generator-to-fit end-to-end local-jackknife workflow. The loader can consume
legacy or manually declared compatible artifacts, but the standard generator
does not produce one.

## Required Stage 00 questions

| # | Audit question | Answer and evidence |
|---:|---|---|
| 1 | Are mature probes variant-axis? | **Yes.** `gwe_ldscore.py:3236-3257` constructs `L x v` probes. |
| 2 | Are all global sources complete before their target scoring? | **Yes within each current probe tile.** `gwe_ldscore.py:5115-5209` completes source construction/finalization before target scoring, but repeats that flow across tiles. |
| 3 | Are outputs per-variant and directional? | **Yes.** Four XX/XW/WX/WW `M x K` panels are emitted (`gwe_ldscore.py:5178-5209`, `5273-5365`). |
| 4 | Is covariate handling rank-aware and common-scale? | **Rank-aware projector: yes. Common sealed genotype scale: no.** Rank is computed at 2011-2018, while distinct X/W post-projection scales appear at 2922-2942 and 3075-3234. |
| 5 | Are probes invariant to tiling/blocking? | **Probe-column tiling: yes. Physical variant blocking: no.** The key includes `blk_start` (`gwe_ldscore.py:3236-3257`; `_native/gxe_multi.py:2575-2588`). |
| 6 | What traversal count is planned? | **Only the one-environment/one-chunk fused case is two passes.** Otherwise native planning is `1 + 2*products*chunks` (`_native/gxe_multi.py:2665-2670`, `2758-2762`, `6972-6986`). |
| 7 | Does non-fused execution decode again for each tile? | **Yes.** Native source/target tile loops occur at `gxeldcore.cpp:13943-14015`; the Python path has equivalent per-tile traversals. |
| 8 | Does mature generation emit jackknife-ready block metadata? | **No.** It explicitly declares no jackknife (`gwe_ldscore.py:1964-1968`, `4292-4398`). |
| 9 | Does inference support target-row subtraction from fixed full sums? | **Yes.** Local mode does this at `gxe.py:1078-1129`. |
| 10 | Is there a standard generation-to-fit jackknife E2E path? | **No.** The generator emits no `BLOCK`/jackknife artifact required by the loader. |
| 11 | Can the mature same-person matrix be computed without a third pass? | **Yes.** It is assembled from completed source state (`gwe_ldscore.py:3366-3468`). |
| 12 | Which systems pieces are reusable? | Descriptor/decode/imputation/common-scale infrastructure, protected NN/TN GEMM, packed/vendor BLAS, threading/NUMA/telemetry, integrity, and atomic publication. Scientific X/W layouts/scales and contextual group actions are excluded. See the reuse map below. |
| 13 | Which tests establish the baseline? | Full Release pytest, opt-in installed-artifact smoke, focused 15-test audit, explicit transfer fixture, 904-test ASAN/UBSAN scope, 904-test UBSAN scope, and the independent 10-test package math oracle, with exact results recorded above. |

## Reuse map and boundaries

Reusable systems seams observed in the live source:

| Capability | Live seam | Reuse boundary |
|---|---|---|
| Packed genotype representation | `src/native/common/genotype.hpp:31-48` | Reuse `MailmanPackedBlock` storage and access, not an X/W-specific layout. |
| Standardized read/decode | `src/native/common/genotype.hpp:53-97`, `109-136` | Reuse existing imputation/standardization and packed memory management. |
| PGEN/BED dispatch | `src/summit/genotype_source.py:38-92`, `330-455` | Extend descriptor routing only; do not create another decoder. |
| DirectContext | `src/native/gxeldcore.cpp:7075`, `8564` | Reuse descriptor/file/decode lifecycle, not current environment/product science. |
| Protected BLAS | `src/native/gxeldcore.cpp:4888`, `5094`, `6145`, `6204` | Reuse checked/partitioned NN/TN execution and fallback behavior. |
| Integrity and telemetry | `src/native/gxeldcore.cpp:1895`, `2964`, `6884` | Reuse integrity snapshots, GEMM telemetry scopes, and packed-panel telemetry. |
| Two-phase source/target pattern | `src/native/gxeldcore.cpp:13697-14064` | Reuse systems scheduling concepts after replacing repeated-tile science with a true global source pass/barrier/target pass. |
| Atomic publication | `src/summit/gwe_ldscore.py:3638` onward and `src/summit/context/_artifact_io.py` | Reuse publication mechanics; implement the authoritative new schemas rather than reusing old payload schemas. |

The new implementation must introduce global logical variant counters, generic
`phi_q`/direction descriptors, one common sealed genotype scale, bounded source
state retained across a hard barrier, and target-row jackknife metadata. These
are scientific changes and are intentionally absent from this Stage 00 report
commit.

## Disposable audit experiments

Small in-memory or temporary-directory experiments were run without persisting
tracked source or data.

### Source completion versus premature target scoring

A three-source-block, two-probe construction produced the completed global
panels:

```text
XX  [6.5, 6.5, 36.0, 30.5]
XW  [10.0, 16.0, 17.0, 18.0]
WX  [6.5, 42.5, 4.0, 10.0]
WW  [16.0, 36.0, 13.0, 29.0]
```

For the first target, completed-global XX was `[6.5, 6.5]`, whereas scoring
against partial source state yielded `[2.5, 2.5]`. This directly demonstrates
why the production global-source barrier is scientifically material. The
combined result hash was
`b07129f8066a3a6c557e68ba677a77335389924609d621d06792a7b39a21785f`.

### Probe-counter invariance

With root seed 20260821 and logical offset 7, generating an `8 x 4` logical
probe matrix as one probe panel or by probe-column tiles produced identical
bytes (SHA-256 prefix `a3f0`). Generating the same logical variants as two
physical blocks with the current block-start key produced a different hash
prefix (`ecc1`), was not equal, and changed 10 entries. This confirms the exact
live incompatibility addressed by the global-counter requirement.

### Fixed-full-sum deletion

For full directional target rows `[10, 20, 30, 40]` assigned to blocks
`[0, 0, 1, 1]`, block sums were `[30, 70]` and retained sums were `[70, 30]`.
The production-style deletion operation returned `[70, 30]` solely by
subtraction; no retained-row kernel was invoked. The deleted-matrix result
SHA-256 began `bfcd0e`.

### Current reference artifact shape

A temporary live reference artifact reported kind `summit.gxe.reference`,
schema 4, and files `diagonal`, `ww`, `wx`, `xw`, and `xx`. Its diagonal table
contained:

```text
CHR SNP BP A1 A2 NORM_X NORM_W SCALE_X SCALE_W DNXE_X DNXE_W CORR_XW ANNOT_0
```

It contained neither top-level jackknife metadata nor a `BLOCK` column. The
temporary reference SHA-256 began `c932329`. The temporary directory was
removed automatically.

## Stage 01 entry criteria

The user has explicitly waived only the isolated
`CODEX_LAUNCH_PROMPT.md` manifest mismatch and authorized continued staged
implementation. Stage 01 is therefore permitted subject to all of its own
scientific and technical stop gates. The waiver does not apply to any future
test, oracle, source-integrity, schema, traversal, or numerical failure.

No estimator change, native change, test change, build-system change, schema
change, or authoritative-package change was made during Stage 00. The only
repository output is this report.
