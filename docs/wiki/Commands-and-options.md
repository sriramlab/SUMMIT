# Commands and options

Use `summit --help` for analysis options, or add `--help` to a command such
as `summit pgs fit`.

## Commands

| Task | Command |
|---|---|
| Genome-wide or windowed LD scores | `summit --geno ...` |
| Heritability or genetic correlation | `summit --h2 ...` or `summit --rg ...` |
| Binary-trait preparation and inference | `summit --binary-method pcgc ...` |
| One-environment G×E | `summit --geno ... --env ...`, `--gxe-score-reference`, `--gxe-fit`, or `--gxe-fit-batch` |
| Polygenic scores | `summit pgs {plan,fit,score,scale,inspect}` |
| Generalized G×E reference planning and inspection | `summit reference {plan,inspect}` |
| Cross-trait reference Z moments | `summit reference zpass` |

Generalized G×E fitting and cross-trait response models use Python APIs.
See [Multiple environments](Multiple-environments.md) and
[Cross-trait analysis](Cross-trait-analysis.md) for their workflows.

## Shared controls

| Option | Meaning |
|---|---|
| `--geno` | BED or PGEN input, with companion files |
| `--annot` | SNP annotations |
| `--out` | Output prefix or directory, as specified by the command |
| `--nvecs` | Number of random vectors for reference estimation |
| `--seed` | Random seed |
| `--block-size` | Number of genotype variants processed together |
| `--num-threads` | Compute thread count |
| `--memory-gib` | Working-memory budget in GiB |
| `--njack` | Jackknife block count or scheme |

Use complete option names. `--block-size` controls computation; `--njack`
controls uncertainty estimation. Windowed LD scores do not use random vectors.
`--block-size auto` is available for one-environment G×E reference generation.
Its resolved width is saved with the reference because it affects the
finite-vector estimate.

| Workflow | Random vectors | Seed | Block size | Memory budget |
|---|---:|---|---:|---|
| LD/G×E preparation | 1,000 | Unspecified | 1,000 | `auto` |
| Binary preparation | 256 | 0 | 256 | 1 GiB |
| PGS | — | — | 512 | 16 GiB |
| Generalized reference plan | Required | — | 4,096 | Required |
| Reference Z moments | — | — | 128 | — |

Binary inference uses 200 contiguous SNP blocks by default. HE/LDSC uses
chromosome deletion (`--njack chr`). Input formats and method-specific
settings are described in the individual analysis guides.

## Memory

For LD/G×E, `--memory-gib` budgets the sketch panels or windowed-LD workspace.
For binary preparation it budgets reference workspace; for PGS it supplies
the fit or scoring memory plan. It is not a limit on process resident memory.
Allow additional memory for inputs, libraries, and output arrays.

G×E also provides `--gxe-native-workspace-gib` for each direct native call
(default 16 GiB), and `--gxe-total-memory-gib` for the estimated process peak
(default `auto`). `--win-cache-mb` controls the windowed-LD cache in MiB.
These limits cover different allocations.

## Genome build

Provide build-matched genotypes, annotations, and reference files. SUMMIT does
not convert coordinates or infer the genome build.

`--genome-build` optionally records a label during binary or PGS scale
preparation. The PGS specification accepts the equivalent `genome_build`
field. Labeled saved scales must be reused with the same label. PGS scoring
rejects conflicting labels when both are supplied and always checks SNP
positions and allele pairs, including when labels are omitted.
