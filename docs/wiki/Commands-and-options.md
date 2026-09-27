# Commands and options

Use `summit` for installed command-line workflows. Each command keeps its
existing estimator, input format and defaults. Older executables and option
spellings remain accepted so existing scripts continue to run.

## Commands

| Task | Command |
|---|---|
| Genome-wide or windowed LD scores | `summit --geno ...` |
| Heritability or genetic correlation | `summit --h2 ...` or `summit --rg ...` |
| Binary-trait preparation and inference | `summit --binary-method pcgc ...` |
| GxE reference, scoring and fitting | `summit --geno ... --env ...`, `--gxe-score-reference`, `--gxe-fit`, or `--gxe-fit-batch` |
| Plan, fit, score or inspect PGS models | `summit pgs {plan,fit,score,scale,inspect}` |
| Plan or inspect generalized GxE references | `summit reference {plan,inspect}` |
| Collect authenticated reference Z summaries | `summit reference zpass` |

Use `--help` after the command, for example `summit pgs fit --help`.
The main help groups inputs, analysis modes, regression, uncertainty,
randomization, memory, and runtime controls. It shows one spelling per option.
Unsupported trace inputs and the deprecated `--collapse-reg-ld` no-op no
longer appear in help; their parser compatibility is retained.

The `reference plan` command plans work and memory; it does not generate a
reference or fit a model. `reference zpass` requires an authenticated study
layout. Cross-trait research drivers and Python APIs retain their specialized
input contracts; the new launcher does not turn them into generic GWAS readers.

## Shared controls

| Option | Meaning |
|---|---|
| `--nvecs` | Number of random probes in LD/GxE/PCGC preparation or reference planning |
| `--seed` | Random seed for LD/GxE/PCGC preparation |
| `--block-size` | Genotype variants processed per block |
| `--num-threads` | Compute thread count, or the thread count used in a dry-run plan |
| `--genome-build` | Build label in binary preparation or PGS genotype-scale preparation |
| `--geno`, `--annot`, `--out` | Genotype input, annotation input and output, where the command accepts them |

The probe axis remains part of the estimator: sharing `--nvecs` does not make
ordinary LD scores and binary reference statistics interchangeable.
Windowed LD scores are deterministic and do not use probes. `--block-size auto`
is supported only for GxE reference generation. An explicit width can affect
that estimator's finite-probe realization; changing its spelling does not.

Defaults are preserved:

| Workflow | Probes | Seed | Genotype block size |
|---|---:|---|---:|
| Main LD/GxE preparation | 1,000 | Unspecified | 1,000 |
| Binary preparation | 256 | 0 | 256 |
| PGS | Not applicable | Not applicable | 512 |
| Generalized reference plan | Required | Not applicable | 4,096 |
| Reference Z pass | Not applicable | Not applicable | 128 |

Binary inference still reports SEs with 200 SNP blocks by default. `--njack`
controls these deletion groups; `--block-size` controls genotype processing.
They are different quantities. The HE/LDSC `--njack` default remains `chr`.

## Memory budgets

These budgets cover different allocations and remain separate:

| Command or option | Scope |
|---|---|
| Binary `--memory-gib` | Reference workspace; default 1 GiB |
| PGS `--memory-gib` | Prediction or scale-preparation memory plan; default 16 GiB |
| `reference plan --memory-gib` | Generalized reference work-plan limit; required |
| Main LD/GxE `--target-mem` | Sketch-panel or windowed-LD budget; default `auto` |
| `--gxe-native-workspace-gib` | Hard ceiling for each direct native GxE call; default 16 GiB |
| `--gxe-total-memory-gib` | Modeled GxE process peak; default `auto` |
| `--win-cache-mb` | Windowed-LD cache in MiB |

A workspace budget is not a limit on the process's resident memory. Passing
binary `--memory-gib` to ordinary LD/GxE fails with a message naming the relevant
budgets, rather than being ignored. `--target-mem` retains its historical
precedence over `--target-xz-mem` if an old script supplies both. The latter
remains a sketch-only legacy control; windowed LD uses `--target-mem`.

## Compatibility table

Use the right-hand spelling in new scripts. The left-hand forms still work.

| Previous spelling | Preferred spelling |
|---|---|
| `summit-pgs ...` | `summit pgs ...` |
| `summit-generalized-gxe-variant-ldscore ...` | `summit reference ...` |
| `--binary-probes` | `--nvecs` |
| `--binary-seed` | `--seed` |
| `--step_size`, `--binary-block-size` | `--block-size` |
| `--binary-memory-gib` | `--memory-gib` |
| `--binary-genome-build` | `--genome-build` |
| Reference plan `--probes` | `--nvecs` |
| Reference plan `--threads` | `--num-threads` |
| Reference plan `--variant-block-width` | `--block-size` |
| Reference plan `--memory-bytes 8589934592` | `--memory-gib 8` |
| Reference Z pass `--bed-prefix`, `--annotations`, `--output`, `--width` | `--geno`, `--annot`, `--out`, `--block-size` |
| Main LD/GxE `--target-xz-mem` | `--target-mem` |
| `--write-ld-mc-ci` | `--write-ld-mc-var` |

The new option aliases share one parsed value. Conflicting values, such as
`--nvecs 256 --binary-probes 1000`, fail explicitly. Equal values are accepted.
Reference planning accepts either GiB or legacy bytes, never both in one call.
Existing overlap-covariance/intercept aliases retain their previous behavior.

Keep scientific inputs distinct: `--covar` names a projection-covariate file,
while `--binary-covariates` selects risk columns from the binary sample table.
`--binary-scale` supplies population means and inverse standard deviations;
`--gxe-genotype-scale` chooses a scaling rule. Merging these options would
change the model.

For an existing installation, reinstall with the same build configuration to
refresh console entry points; see [Installation](Installation.md).
The old standalone commands continue to work. `python -m summit` also uses
the unified dispatcher. PGS retains its own thread/NUMA initialization before
numerical imports; it does not pass through the legacy LD runtime setup.
