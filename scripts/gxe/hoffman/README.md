# Hoffman common-cohort GxE runbook

This directory contains preparation/verification code and an explicit runbook;
it does not submit jobs. The cache, wide-score, probe-shard, and shard-merge
APIs are implemented. The focused GxE/deployment suite must pass from the exact
checksummed frozen snapshot before sealing. The production trace contract is
eight contiguous B128 shards forming one B1024 reference, with sealed prefix
checkpoints at B128, B256, B512, and B1024. Earlier B10/B50/B100 runs are
historical calibration evidence, not inputs to this production tiling.

## Fixed analysis contract

The four core groups and exact expected full-cohort N/ranks are in
`panel_config.json`. Production uses post-projection standardized kernels and
sample genotype scaling. GENIE/HWE is a separately named comparator, not the
default and not merge-compatible with default artifacts.

The panel deliberately batches 32 trait-by-environment analyses into only four
fixed-design references. `age_bp` retains the medication-adjusted DBP/SBP
design, and `sex_whr` retains WHR's BMI adjustment. The common-covariate
`age_assay` and `sex_assay` groups include the strongest published comparison
traits plus ApoB, total bilirubin, and (for sex) total protein, which are absent
from the published GENIE 52-trait result table. Their full-cohort intersections
retain 273,532 and 223,433 samples, respectively.

All generated data, logs, job scripts, verification reports, caches, partials,
and results belong under:

```text
/u/scratch/b/bronsonj/summit_gxe_full_20260808
```

Use `/u/project/sriram/bronsonj` only for the small frozen code/config snapshot.
Never run from or write into collaborator source trees. The full and valid 50k
PLINK triples must each be physically copied once into a new scratch leaf.

## Staging rules

Start every staging or job shell with:

```bash
set -euo pipefail
umask 077
```

Create each scratch directory once with mode 0700. Copy files without following
symlinks, then set BED/BIM/FAM and phenotype/covariate files to mode 0600. Do not
reuse an old output leaf and do not use an overwrite flag. The verifier hashes
the entire source and staged BED, so a same-size corruption cannot pass.

Before any transfer, freeze the local phenotype/covariate inventory in one new
private manifest. The required source root and phenotype filename template are
explicit in `panel_config.json`; the command-line root must match that value.
The generator checks the full-cohort FAM hash/N, exact phenotype headers,
covariate environment headers, unique IDs, and exact FAM row order. It resolves
22 unique `{trait}.pheno` files and one named covariate record per group (four
records), hashes the bytes while validating them, never writes IDs, and refuses
symlinks or an existing output path.

```bash
set -euo pipefail
umask 077
install -d -m 0700 <NEW_PRIVATE_LOCAL_MANIFEST_DIRECTORY>
python scripts/gxe/hoffman/generate_source_manifest.py \
  --config scripts/gxe/hoffman/panel_config.json \
  --source-root /home/bronsonj/UKBB/asha/phens \
  --fam /home/bronsonj/UKBB/geno/EUR_300k/UKBB_EUR_300k_unrel_3rd.no_mhc_imp.fam \
  --output <NEW_PRIVATE_LOCAL_MANIFEST_DIRECTORY>/source_manifest.json
```

The JSON is created atomically as mode 0600 and cannot be placed inside the
read-only source root. Use its `relative_path`, `bytes`, and `sha256` records as
the transfer allowlist. Copy exactly those 22 phenotype files and four named
covariate files to a new 0700 Hoffman scratch leaf, then require every staged
file to match the corresponding manifest size/hash before building groups.
Keep the two byte-identical assay covariate filenames as separate named records
so their group provenance remains explicit.

After restrictive phenotype/covariate copies have been transferred, build each
group against the staged FAM. Example shape (paths are intentionally explicit
placeholders and this command does not copy anything):

```bash
python scripts/gxe/hoffman/build_common_group.py \
  --scratch-root /u/scratch/b/bronsonj/summit_gxe_full_20260808 \
  --fam <SCRATCH_GENOTYPE_PREFIX>.fam \
  --covar <SCRATCH_PHENOTYPE_SOURCE_DIR>/bp_diastolic.covar \
  --environment-column age \
  --phenotype bp_diastolic=<SCRATCH_PHENOTYPE_SOURCE_DIR>/bp_diastolic.pheno \
  --phenotype bp_systolic=<SCRATCH_PHENOTYPE_SOURCE_DIR>/bp_systolic.pheno \
  --label age_bp \
  --out-dir <NEW_PRIVATE_GROUP_LEAF>
```

Repeat according to `panel_config.json`. The builder requires exact FAM ID-set
equality for every input, fixes one intersection cohort, writes FAM-order tables
with every outside row marked missing, and records source/output hashes plus the
selected-ID digest. It refuses a pre-existing output directory.

For the configured valid-50k calibration only, the source phenotype tables
legitimately contain the other full-cohort IDs. Add
`--allow-source-superset`; this still requires every 50k FAM ID exactly once.
Do not use that flag for the full-cohort groups.

Verify a staged genotype and its group manifests once, in a dedicated staging
job, before any cache job:

```bash
python scripts/gxe/hoffman/verify_staged_inputs.py \
  --config scripts/gxe/hoffman/panel_config.json \
  --dataset full \
  --geno-prefix <PRIVATE_STAGED_FULL_PREFIX> \
  --group-manifest <PRIVATE_AGE_BP_GROUP>/age_bp.group.json \
  --group-manifest <PRIVATE_AGE_ASSAY_GROUP>/age_assay.group.json \
  --group-manifest <PRIVATE_SEX_WHR_GROUP>/sex_whr.group.json \
  --group-manifest <PRIVATE_SEX_ASSAY_GROUP>/sex_assay.group.json \
  --report <NEW_PRIVATE_REPORT_PATH>
```

Use `--dataset subset_50k` and group manifests built against the staged valid
50k FAM for calibration. Never use the truncated `50k_subset/plink2` files.

## Computation gate and exact CLI sequence

The cache, reference-shard, shard-merge, wide-score, single-fit, and batch-fit
APIs pass the current-source equivalence suite. Production remains gated on
repeating those checks from the checksummed frozen snapshot and on the staged
50k calibration.
The commands below are exact CLI templates; substitute only new private
staged/group/output paths. Every output prefix must be fresh. Never use
`--gxe-overwrite` in this workflow.

### Hash-bound UGE deployment layer

`deployment_config.json`, `job_spec_templates.json`, `hoffman_deploy.py`, and
the nine `uge_*.sh` wrappers implement the production launch gates. They do not
submit anything. A renderer validates all sealed inputs and writes one concrete
script plus pre-created private stdout/stderr files in a fresh 0700 job leaf;
it only prints the corresponding `qsub` command. The wrappers expose no free
SUMMIT arguments and reject arrays, wrong `NSLOTS`, wrong file modes, symlink
components, hard-linked staged inputs, stale hashes, reused output leaves, and
non-frozen imports.

After copying the final code snapshot below `/u/project/sriram/bronsonj`, make
one new 0700 provenance directory inside that snapshot and seal the exact
package Python files, deployment scripts/configs, interpreter bytes, every
recorded file in the required Python distributions, and native `gwldcore`
extension in a schema-v2 frozen manifest. This command refuses an existing
manifest:

```bash
GXE_FROZEN=/u/project/sriram/bronsonj/<FROZEN_SUMMIT_SNAPSHOT>
GXE_PY=/u/home/b/bronsonj/.conda/envs/summit/bin/python
install -d -m 0700 "${GXE_FROZEN}/private_provenance"
"${GXE_PY}" -I -B "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
  seal-code-manifest \
  --config "${GXE_FROZEN}/scripts/gxe/hoffman/deployment_config.json" \
  --root "${GXE_FROZEN}" \
  --python "${GXE_PY}" \
  --native-build-dir "${GXE_FROZEN}/build/<WHEEL_TAG>" \
  --output "${GXE_FROZEN}/private_provenance/frozen_code_manifest.json"
```

Copy one template from `job_spec_templates.json` into a new mode-0600 JSON
under the configured scratch root and replace every placeholder with an exact
record object (`path`, byte count, and lowercase SHA256). Create its `job_root`
once as an empty 0700 leaf. Render without submitting:

```bash
"${GXE_PY}" -I -B "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
  render \
  --config "${GXE_FROZEN}/scripts/gxe/hoffman/deployment_config.json" \
  --job-spec <PRIVATE_MODE_0600_JOB_SPEC>
```

Inspect the single printed `qsub <.../job.sh>` command, then submit it manually
only after the stated gate is satisfied. The job writes
`process_receipt.json` only after exact artifact postvalidation. Final UGE
accounting is unavailable to a running job, so after it exits run the separate
collector once (no polling loop), supplying the receipt values recorded after
the job closed:

```bash
"${GXE_PY}" -I -B "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
  record-qacct \
  --config "${GXE_FROZEN}/scripts/gxe/hoffman/deployment_config.json" \
  --expected-config-sha256 <FROZEN_DEPLOYMENT_CONFIG_SHA256> \
  --receipt <JOB_ROOT>/process_receipt.json \
  --receipt-sha256 <PROCESS_RECEIPT_SHA256> \
  --receipt-bytes <PROCESS_RECEIPT_BYTES>
```

This writes one no-replace `completed_qacct.json` only when the unique final
record has `failed=0`, `exit_status=0`, and the exact requested slot count; the
raw private accounting output is retained as `qacct.txt` and hash-bound by that
receipt. The first UGE invocation creates `attempt.json` before checking slots,
imports, or inputs; the shell creates a no-clobber `attempt.lock` even before
Python starts. Any failed/partial leaf is therefore permanently
quarantined and must not be retried in place.
Every downstream spec must cite the completed qacct record, not merely the
process receipt. Every dependency must bind the exact current deployment-config
record. The sole cross-snapshot exception is dependency 1 of `cache_attest`: it
must be a completed cache job under the explicitly allowlisted `0113342`
deployment-config SHA. No other task or dependency index can use that exception.

Every task spec also declares `task_args.dataset`, exactly `full` or
`subset_50k`. Only `full` may use the `production` shard lineage and produce
`production_prefix`, `production_score`, or `production_fit` receipts.
`subset_50k` remains available for calibration, but must use the separately
labeled `calibration`, `calibration_prefix`, `calibration_score`, and
`calibration_fit` lineage. Dataset equality is checked at every stage, cache,
attestation, shard, merge, score, and fit boundary. Thus a valid-50k B1024
calibration cannot satisfy a full-cohort score or reporting dependency. The
separately named eight-slot shard-00 benchmark can never enter either merge
lineage.

For the valid-50k B128 calibration gate, copy the relevant template but use
these exact fields; do not reuse the full/production values:

```json
{"task":"shard","task_args":{"dataset":"subset_50k","role":"calibration","shard_index":0}}
{"task":"merge","task_args":{"dataset":"subset_50k","shards":["<VALID_50K_SHARD_00_RECORD>"]}}
```

The shard receipt must consequently contain
`{"dataset":"subset_50k","role":"calibration"}` and the B128 merge receipt
must contain `{"dataset":"subset_50k","role":"calibration_prefix"}`. If the
calibration continues to B1024, its score and fit specs must retain
`dataset=subset_50k`; their dependencies and receipts must respectively carry
`calibration_prefix`, `calibration_score`, and `calibration_fit`. None of these
records is accepted by a `dataset=full` job.

The renderer derives every task-specific path from the validated `task_args`
records in the private job spec and writes a deterministic, hash-checked
`job.sh`. Do not edit that script or inject shell variables into it. The
following names are explanatory shorthand used only by the manual CLI
templates below; production jobs use the equivalent concrete paths assembled
by `hoffman_deploy.py`:

```bash
GXE_SUMMIT=<FROZEN_CHECKSUMMED_SUMMIT_ENTRYPOINT>
GXE_GENO=<PRIVATE_STAGED_PLINK_PREFIX>
GXE_GROUP=<PRIVATE_GROUP_DIRECTORY>
GXE_LABEL=<age_bp_OR_age_assay_OR_sex_whr_OR_sex_assay>
GXE_ENV="${GXE_GROUP}/${GXE_LABEL}.env.tsv"
GXE_COVAR="${GXE_GROUP}/${GXE_LABEL}.covar.tsv"
GXE_WIDE_PHENO="${GXE_GROUP}/${GXE_LABEL}.phenotypes.tsv"
GXE_TRAITS=<CONFIGURED_COMMA_SEPARATED_TRAITS_IN_ORDER>
GXE_CACHE_OUT=<NEW_PRIVATE_CACHE_PREFIX>
GXE_CACHE="${GXE_CACHE_OUT}.gxe.cache.npz"
GXE_PARTIAL_ROOT=<NEW_PRIVATE_SHARD_DIRECTORY>
GXE_MERGED_ROOT=<NEW_PRIVATE_MERGE_DIRECTORY>
GXE_SCORE_OUT=<NEW_PRIVATE_BATCH_SCORE_PREFIX>
GXE_FIT_ROOT=<NEW_PRIVATE_FIT_DIRECTORY>
```

The production annotation contract is deliberately unpartitioned and matches
the published GENIE one-bin comparisons: `annotation=null` and
`annotation_contract=all_variants_unit_weight` in `panel_config.json`.
Therefore, omit `--annot`. The cache must record one `L2_0` annotation with an
all-ones canonical vector and mass 454,207. Do not stage an annotation file.

Build one phenotype-free cache per dataset/group. `--write-gxe-jackknife` and
`--njack 100` seal the production SNP-block definition into the cache:

```bash
"${GXE_SUMMIT}" \
  --gxe-build-cache \
  --geno "${GXE_GENO}" \
  --env "${GXE_ENV}" \
  --covar "${GXE_COVAR}" \
  --gxe-kernel-mode standardized \
  --gxe-genotype-scale sample \
  --write-gxe-jackknife \
  --njack 100 \
  --rand-dist rademacher \
  --seed 20260808 \
  --dtype float32 \
  --ddof 1 \
  --gxe-missing-values=-9,NA,NaN,nan,.,None,null \
  --impute-method mean \
  --step_size 500 \
  --target-xz-mem 16 \
  --nvecs 1024 \
  --num-threads "${NSLOTS}" \
  --out "${GXE_CACHE_OUT}"
```

This writes `${GXE_CACHE_OUT}.gxe.cache.npz`. Validate its mode, SHA256,
genotype/design fingerprints, N/rank, one-bin annotation contract, J=100 block
labels, finite feature scales, and diagnostics before any shard job.

A schema-v2 cache produced by the frozen `0113342` deployment may be reused
only through `cache_attest`. Its dependencies are ordered: first a newly
completed `stage_verify` job whose report binds the B1024 panel SHA, then the
completed legacy cache qacct record. The task requires the exact allowlisted
legacy deployment-config SHA and a deterministic fingerprint of every
`src/summit/**/*.py` source recorded by the old frozen manifest, then reruns
schema/array, genotype, group/design/rank, environment/covariate,
analysis-fingerprint, annotation, and J-block validation. It also reads the
qacct-bound legacy cache job spec and requires its exact genotype/group records
and stage report to declare the same dataset as the new attestation. It writes
a new dataset-bound attestation; an attested-cache shard must cite that exact
artifact and its completed qacct record. Merely having schema version 2 is
insufficient.

Each B128/J100 shard uses the same cache, seed, and estimator settings, with a
disjoint global probe interval. Run one new job for each
`GXE_SHARD_INDEX=0,...,7`; do not use an array or a task-concurrency cap:

```bash
GXE_SHARD_INDEX=<INTEGER_0_THROUGH_7>
GXE_PROBE_OFFSET=$((128 * GXE_SHARD_INDEX))
printf -v GXE_SHARD_TAG '%02d' "${GXE_SHARD_INDEX}"
GXE_SHARD_OUT="${GXE_PARTIAL_ROOT}/shard_${GXE_SHARD_TAG}"

"${GXE_SUMMIT}" \
  --geno "${GXE_GENO}" \
  --env "${GXE_ENV}" \
  --covar "${GXE_COVAR}" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --gxe-reference-shard \
  --gxe-probe-offset "${GXE_PROBE_OFFSET}" \
  --nvecs 128 \
  --gxe-kernel-mode standardized \
  --gxe-genotype-scale sample \
  --write-gxe-jackknife \
  --njack 100 \
  --rand-dist rademacher \
  --seed 20260808 \
  --dtype float32 \
  --ddof 1 \
  --gxe-missing-values=-9,NA,NaN,nan,.,None,null \
  --impute-method mean \
  --step_size 500 \
  --target-xz-mem 16 \
  --num-threads "${NSLOTS}" \
  --out "${GXE_SHARD_OUT}"
```

The merge input is each `${GXE_SHARD_OUT}.gxe.shard.json`, not a directional
panel or jackknife NPZ. Shards 00 through 07 cover `[0,128)` through
`[896,1024)`. Full production `merge` specs accept only `dataset=full` shards
with `role=production`; subset calibration merges analogously require only
`dataset=subset_50k`, `role=calibration` shards. Both accept only the four exact
contiguous prefixes ending at B128, B256, B512, and B1024. All checkpoints are
at or above the merger's 100-probe fit-able threshold and therefore must not
carry the low-probe override:

```bash
GXE_B128_OUT="${GXE_MERGED_ROOT}/B128"
"${GXE_SUMMIT}" \
  --gxe-merge-shards "${GXE_PARTIAL_ROOT}/shard_00.gxe.shard.json" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --out "${GXE_B128_OUT}"
```

The production B1024 merge lists all eight shards and must not use the
low-probe override:

```bash
GXE_SHARDS=(
  "${GXE_PARTIAL_ROOT}/shard_00.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_01.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_02.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_03.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_04.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_05.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_06.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_07.gxe.shard.json"
)
GXE_B1024_OUT="${GXE_MERGED_ROOT}/B1024"
"${GXE_SUMMIT}" \
  --gxe-merge-shards "${GXE_SHARDS[@]}" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --out "${GXE_B1024_OUT}"
```

For an independent-half Monte Carlo diagnostic, use only the separately named
`merge_half` task with shards 04--07. It emits `B512_second_half` and is marked
`diagnostic_second_half`; a score task accepts qacct provenance only from the
ordinary prefix-only `merge` task, so this non-prefix artifact cannot enter the
production path. Never relax the ordinary merge prefix/checkpoint checks.

The merger rejects mixed cache hashes, sample/variant/annotation/design
fingerprints, J, kernel/genotype scales, randomization settings, or overlapping
probe identities. Schema-v3 merged references retain raw realized-sample
panels; no generic XX-like storage offset is applied to any direction.

After validating the B1024 reference, score every group phenotype in one BED
pass. Production scoring requires `dataset=full` and a completed
`production_prefix` B1024 merge. A valid-50k calibration score must instead
declare `dataset=subset_50k` and depend on a `calibration_prefix` B1024 merge:

```bash
"${GXE_SUMMIT}" \
  --gxe-score-reference "${GXE_B1024_OUT}.gxe.ref.json" \
  --geno "${GXE_GENO}" \
  --env "${GXE_ENV}" \
  --covar "${GXE_COVAR}" \
  --gxe-pheno "${GXE_WIDE_PHENO}" \
  --gxe-pheno-cols "${GXE_TRAITS}" \
  --gxe-missing-values=-9,NA,NaN,nan,.,None,null \
  --step_size 500 \
  --num-threads "${NSLOTS}" \
  --out "${GXE_SCORE_OUT}"
```

For each trait, this writes `${GXE_SCORE_OUT}.<trait>.gxe.gwas.tsv.gz`,
`${GXE_SCORE_OUT}.<trait>.gxe.gwis.tsv.gz`, and
`${GXE_SCORE_OUT}.<trait>.gxe.moments.json`. These artifacts bind the
feature-cache SHA and may be reused with any B checkpoint merged from that
exact cache; do not rescore each checkpoint.

For production, use one `fit_batch` Hoffman spec per configured group with
`dataset=full`; its completed score dependency must be a same-dataset
`production_score`. A separately labeled valid-50k calibration fit declares
`dataset=subset_50k` and requires a `calibration_score`. Its
`traits` entries must contain exactly `trait`, `moments`, `gwas`, and `gwis`,
must list every group trait in `panel_config.json` order, and must use exact
mode-0600 `path`/`bytes`/`sha256` records. The spec has exactly one dependency:
the completed qacct record for that group's wide-score job. Rendering fails
unless every triplet record is an exact member of the score receipt's outputs,
the remaining score output is its single log, and the qacct group, trait order,
cache SHA, and B1024-reference SHA all match. In addition, the batch cache and
reference records must exactly equal the `path`/`bytes`/`sha256` records in the
qacct-bound score job spec's `task_args`; matching content hashes at different
paths are deliberately insufficient.

The renderer writes `fit_batch_manifest.json` as a deterministic mode-0600
`summit.gxe.fit_batch` manifest inside the fresh job root. It chooses one fresh
common output parent (`<JOB_ROOT>/artifacts`), creates one output prefix per
trait, and launches one single-slot command equivalent to:

```bash
"${GXE_SUMMIT}" \
  --gxe-fit-batch <JOB_ROOT>/fit_batch_manifest.json \
  --gxe-max-condition 1e12 \
  --out <JOB_ROOT>/artifacts/fit_batch
```

The job receipt binds the exact rendered manifest. Completion requires the
single batch log and both result files for every configured trait, with no
extra artifact, and postvalidates every fit's rank, conditioning, jackknife,
four-component order, finite diagnostics, and JSON/TSV agreement. Publication
is transactional in SUMMIT: an incomplete multi-trait result is not accepted
as a completed Hoffman job. Because the strict CLI manifest schema contains
paths rather than file records, Hoffman provenance separately records every
cache, reference, moments, GWAS, and GWIS `path`/`bytes`/`sha256` triple in the
task details, invocation, process receipt, and final qacct receipt. The runner
rehashes all of them immediately before invoking SUMMIT, immediately after it
returns, after output postvalidation, and again before qacct collection; any
mutation permanently fails that fresh job leaf. Each per-trait fit JSON also
records the canonical path, byte count, and SHA256 accumulated from the private
SUMMIT input snapshots actually parsed for the reference manifest, feature
cache, phenotype moments, GWAS, and GWIS. Postvalidation requires those five
records to exactly equal the deployment records (with one shared
reference/cache pair across the batch), closing even a change-read-restore
mutation that leaves the live paths unchanged at the after-run rehash.

The existing `fit` form remains available for a separately named one-trait
diagnostic or sensitivity analysis. It fits one trait against the validated
B1024 reference:

```bash
GXE_TRAIT=<ONE_CONFIGURED_TRAIT>
"${GXE_SUMMIT}" \
  --gxe-fit "${GXE_B1024_OUT}.gxe.ref.json" \
  --gxe-gwas "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.gwas.tsv.gz" \
  --gwis "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.gwis.tsv.gz" \
  --gxe-moments "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.moments.json" \
  --gxe-max-condition 1e12 \
  --out "${GXE_FIT_ROOT}/${GXE_TRAIT}"
```

The single-trait Hoffman form uses the same five-record provenance gate as the
batch form. Its cache and reference must exactly match the completed score job
spec, its moments/GWAS/GWIS records must be exact score outputs, and its fit
JSON's snapshot-derived consumed-input records must exactly match all five
deployment records. The runner binds those records in task details, invocation,
process receipt, and qacct receipt and rehashes them around execution and after
postvalidation. Thus the diagnostic form does not weaken the path-alias or
change-read-restore protections of the production batch form.

Do not add `--allow-ill-conditioned-gxe` to primary fits. If a separately named
sensitivity fit is later justified, preserve the primary failure/diagnostics.

Execution order:

1. Freeze the now-green current source, then rerun the complete equivalence
   suite from that checksummed snapshot with user-site/editable imports disabled.
2. Run a new stage-verification job so its report binds the new B1024 panel
   SHA. Build new caches, or run the explicit `0113342` cache-attestation task;
   never carry an old stage receipt into this panel.
3. Gate B128 first on valid-50k age-BP, checking the rendered resource/free-space
   arithmetic and final qacct wall/CPU/maxvmem. Historical B10/B50/B100 artifacts
   cannot enter the new intervals.
4. On full N, run each group's shard 00 over `[0,128)` and validate the B128
   checkpoint before launching the remaining seven nonarray jobs. Merge only
   the exact prefixes for B256, B512, and B1024.
5. Optionally run the separately named second-half diagnostic over shards 04--07
   after all shard qacct records are complete. It cannot satisfy a score
   dependency and is not a production checkpoint.
6. Batch-score each group once after B1024, then run one group-level batch-fit
   job containing every configured trait in order. Use single-trait fit specs
   only for separately named diagnostics. Interpret sex
   GxE, but not separate sex NxE/residual estimates; binary-environment
   identifiability remains intrinsic. Use age groups for heterogeneous-noise
   validation.

## UGE resources and monitoring

The renderer derives these fixed profiles from `deployment_config.json`; the
total is always checked as `slots × h_data-per-slot`:

| Task | Slots | `h_data` per slot | Total | `h_rt` |
|---|---:|---:|---:|---:|
| staged-input verification | 1 | 4G | 4 GiB | 24:00:00 |
| cache | 4 | 6G | 24 GiB | 48:00:00 |
| legacy-cache attestation | 1 | 8G | 8 GiB | 08:00:00 |
| production B128 shard | 4 | 12G | 48 GiB | 48:00:00 |
| optional shard-00 benchmark | 8 | 6G | 48 GiB | 24:00:00 |
| merge checkpoint | 1 | 8G | 8 GiB | 08:00:00 |
| second-half diagnostic merge | 1 | 8G | 8 GiB | 08:00:00 |
| wide score | 4 | 6G | 24 GiB | 48:00:00 |
| per-trait fit | 1 | 4G | 4 GiB | 04:00:00 |
| group batch fit | 1 | 4G | 4 GiB | 04:00:00 |

At the largest sealed group (`N=290259`), B128, J=100, K=1, and float32, the
preflight model is 27.681 GiB for the two deletion-sketch scratch maps, 0.554
GiB for four resident sketches, and 3.785 GiB for the one-block decode/
projection workspace: 32.020 GiB before unmodeled runtime overhead. The 48-GiB
profile must also satisfy a configured 12-GiB modeled-memory reserve. Before
rendering and again at job start, the scratch filesystem must have at least the
27.681-GiB shard scratch allocation plus an 8-GiB free-space reserve. This is a
per-job capacity gate; operators must still account for aggregate space before
launching several shards concurrently.

B128 remains the production shard size because the explicit one-block model
fits the 48-GiB request with reserve; no B64 fallback is needed. A cache-bound
shard still makes two genotype passes, so eight B128 shards require 16 genotype
passes per group. These are initial unmeasured full-cohort requests: inspect the
first valid-50k and full-N B128 qacct records before broad launch, and change
resources only in a newly checksummed/resealed snapshot. Do not use `-tc`.

All `-o`/`-e` logs, Python temporary files, and generated job scripts must be
scratch paths. The renderer fixes `TMPDIR` to a private job subdirectory; the
wrapper runs Python with `-I -B`, and the runtime verifies effective isolated,
no-user-site, and no-bytecode flags. Python hash randomization remains enabled
under `-I`; scientific probe identities use the explicit sealed Philox seed and
do not depend on Python's process hash. The renderer uses a minimal
`PATH`/`LD_LIBRARY_PATH`, caps BLAS/OpenMP threads to `NSLOTS`, and records the
exact imported module path, sealed interpreter/distribution hashes,
code/config/cache hashes, command, UGE job ID, package versions, and thread
variables before computation.
The rendered job applies the hash-bound `/usr/bin/numactl --interleave=all`
policy once at its outer launch boundary and exports the recursion sentinel
before the runner calls SUMMIT in-process. A nested launcher would corrupt the
embedded command line and invalidate the sealed invocation provenance.

Monitor with `qstat -j <JOB_ID>` immediately after submission and after startup,
then no more frequently than about every six hours for long jobs. Inspect only
private log size, modification time, and a short tail. On completion, require
`qacct -j <JOB_ID>` to report `failed=0` and `exit_status=0`, and record
wallclock, CPU, and maxvmem before validating artifacts. Do not create polling
loops or high-frequency status files.
