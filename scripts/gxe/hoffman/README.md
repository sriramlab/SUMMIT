# Hoffman common-cohort GxE runbook

This directory contains preparation/verification code and an explicit runbook;
it does not submit jobs. The cache, wide-score, probe-shard, and shard-merge
APIs are implemented, and the current-source GxE suite is green (238/238 tests
at the deployment-layer freeze point). The remaining production gates are an
independent test run from the checksummed frozen snapshot and the staged 50k
calibration described below.

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
from the published GENIE 53-trait result table. Their full-cohort intersections
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

The cache, reference-shard, shard-merge, wide-score, and fit APIs pass the
current-source equivalence suite. Production remains gated on repeating those
checks from the checksummed frozen snapshot and on the staged 50k calibration.
The commands below are exact CLI templates; substitute only new private
staged/group/output paths. Every output prefix must be fresh. Never use
`--gxe-overwrite` in this workflow.

### Hash-bound UGE deployment layer

`deployment_config.json`, `job_spec_templates.json`, `hoffman_deploy.py`, and
the six `uge_*.sh` wrappers implement the production launch gates. They do not
submit anything. A renderer validates all sealed inputs and writes one concrete
script plus pre-created private stdout/stderr files in a fresh 0700 job leaf;
it only prints the corresponding `qsub` command. The wrappers expose no free
SUMMIT arguments and reject arrays, wrong `NSLOTS`, wrong file modes, symlink
components, hard-linked staged inputs, stale hashes, reused output leaves, and
non-frozen imports.

After copying the final code snapshot below `/u/project/sriram/bronsonj`, make
one new 0700 provenance directory inside that snapshot and seal the exact
package Python files, deployment scripts/configs, interpreter, and native
`gwldcore` extension. This command refuses an existing manifest:

```bash
GXE_FROZEN=/u/project/sriram/bronsonj/<FROZEN_SUMMIT_SNAPSHOT>
GXE_PY=/u/home/b/bronsonj/.conda/envs/summit/bin/python
install -d -m 0700 "${GXE_FROZEN}/private_provenance"
"${GXE_PY}" -I "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
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
"${GXE_PY}" -I "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
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
"${GXE_PY}" -I "${GXE_FROZEN}/scripts/gxe/hoffman/hoffman_deploy.py" \
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
process receipt. Merge specs accept only completed production shard receipts;
the separately named eight-slot shard-00 benchmark can never enter a merge.

Set these task-specific variables inside each private job script:

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
  --nvecs 100 \
  --num-threads "${NSLOTS}" \
  --out "${GXE_CACHE_OUT}"
```

This writes `${GXE_CACHE_OUT}.gxe.cache.npz`. Validate its mode, SHA256,
genotype/design fingerprints, N/rank, one-bin annotation contract, J=100 block
labels, finite feature scales, and diagnostics before any shard job.

Each B10/J100 shard uses the same cache, seed, and estimator settings, with a
disjoint global probe interval. Run one new job for each
`GXE_SHARD_INDEX=0,...,9`; do not use an array concurrency cap:

```bash
GXE_SHARD_INDEX=<INTEGER_0_THROUGH_9>
GXE_PROBE_OFFSET=$((10 * GXE_SHARD_INDEX))
printf -v GXE_SHARD_TAG '%02d' "${GXE_SHARD_INDEX}"
GXE_SHARD_OUT="${GXE_PARTIAL_ROOT}/shard_${GXE_SHARD_TAG}"

"${GXE_SUMMIT}" \
  --geno "${GXE_GENO}" \
  --env "${GXE_ENV}" \
  --covar "${GXE_COVAR}" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --gxe-reference-shard \
  --gxe-probe-offset "${GXE_PROBE_OFFSET}" \
  --nvecs 10 \
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
panel or jackknife NPZ. A one-shard B10 merge is diagnostic and must carry the
explicit low-probe override:

```bash
GXE_B10_OUT="${GXE_MERGED_ROOT}/B010"
"${GXE_SUMMIT}" \
  --gxe-merge-shards "${GXE_PARTIAL_ROOT}/shard_00.gxe.shard.json" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --allow-low-probe-gxe-jackknife \
  --out "${GXE_B10_OUT}"
```

Use the same form for cumulative B20--B90 diagnostic checkpoints, listing only
the intended disjoint shard manifests and retaining the low-probe override.
The production B100 merge lists all ten shards and must not use that override:

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
  "${GXE_PARTIAL_ROOT}/shard_08.gxe.shard.json"
  "${GXE_PARTIAL_ROOT}/shard_09.gxe.shard.json"
)
GXE_B100_OUT="${GXE_MERGED_ROOT}/B100"
"${GXE_SUMMIT}" \
  --gxe-merge-shards "${GXE_SHARDS[@]}" \
  --gxe-feature-cache "${GXE_CACHE}" \
  --out "${GXE_B100_OUT}"
```

The merger rejects mixed cache hashes, sample/variant/annotation/design
fingerprints, J, kernel/genotype scales, randomization settings, or overlapping
probe identities. Schema-v3 merged references retain raw realized-sample
panels; no generic XX-like storage offset is applied to any direction.

After validating the B100 reference, score every group phenotype in one BED
pass. Production scoring can wait for B100:

```bash
"${GXE_SUMMIT}" \
  --gxe-score-reference "${GXE_B100_OUT}.gxe.ref.json" \
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
exact cache; do not rescore each checkpoint. Fit one trait at a time against
the validated B100 reference:

```bash
GXE_TRAIT=<ONE_CONFIGURED_TRAIT>
"${GXE_SUMMIT}" \
  --gxe-fit "${GXE_B100_OUT}.gxe.ref.json" \
  --gxe-gwas "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.gwas.tsv.gz" \
  --gwis "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.gwis.tsv.gz" \
  --gxe-moments "${GXE_SCORE_OUT}.${GXE_TRAIT}.gxe.moments.json" \
  --gxe-max-condition 1e12 \
  --out "${GXE_FIT_ROOT}/${GXE_TRAIT}"
```

Do not add `--allow-ill-conditioned-gxe` to primary fits. If a separately named
sensitivity fit is later justified, preserve the primary failure/diagnostics.

Execution order:

1. Freeze the now-green current source, then rerun the complete equivalence
   suite from that checksummed snapshot with user-site/editable imports disabled.
2. On valid 50k, build all four caches, run shard 00, and make a B10 diagnostic
   merge. Benchmark the identical shard-00 probe identity at four and eight
   slots under distinct prefixes; only one copy may enter later merges.
3. Compare numerical outputs and final `qacct` wall/CPU/maxvmem. Do not score
   the full phenotype panel during this compute-only calibration gate.
4. On full N, build all four caches and run shard 00. These are the first ten
   production probes, not throwaway calibration artifacts.
5. Only after the full-B10 gates pass, run shards 01--09 in manual waves and
   merge at every B10 increment. Only B100 is a production reference.
6. Batch-score each group once after B100, then fit every trait. Interpret sex
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
| production shard | 4 | 6G | 24 GiB | 48:00:00 |
| shard-00 calibration benchmark | 8 | 4G | 32 GiB | 24:00:00 |
| merge checkpoint | 1 | 8G | 8 GiB | 08:00:00 |
| wide score | 4 | 6G | 24 GiB | 48:00:00 |
| per-trait fit | 1 | 4G | 4 GiB | 04:00:00 |

These are initial gated requests, not measured full-cohort requirements; revise
them only through a new checksummed config after the 50k qacct calibration.
Do not use `-tc` anywhere. Submit independent jobs manually; after calibration,
run one group at a time in two waves of five four-slot probe jobs (20 requested
slots per wave).

All `-o`/`-e` logs, Python temporary files, and generated job scripts must be
scratch paths. The renderer fixes `TMPDIR` to a private job subdirectory, sets
`PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1`, uses a minimal environment
`PATH`/`LD_LIBRARY_PATH`, caps BLAS/OpenMP threads to `NSLOTS`, and records the
exact imported module path, code/config/cache hashes, command, UGE job ID,
package versions, and thread variables before computation.

Monitor with `qstat -j <JOB_ID>` immediately after submission and after startup,
then no more frequently than about every six hours for long jobs. Inspect only
private log size, modification time, and a short tail. On completion, require
`qacct -j <JOB_ID>` to report `failed=0` and `exit_status=0`, and record
wallclock, CPU, and maxvmem before validating artifacts. Do not create polling
loops or high-frequency status files.
