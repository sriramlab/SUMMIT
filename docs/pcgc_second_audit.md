# PCGC uncertainty and full-sample reference audit

This follow-up checks whether the remaining qualification failures identify an
implementation error, and removes redundant work in the binary reference path.
It follows [the first jackknife audit](pcgc_jackknife_audit.md). New outputs are
under `/data1/bronsonj/summit_pcgc_second_audit_20260926/`; earlier results remain
unchanged. The existing SNP deletion equations and experimental SE gate are
unchanged in this pass.

The subsequent [release validation](pcgc_release_validation.md) extends the
corrected-jackknife comparison to the other binary modes and annotation
components.

## What the official implementations actually resample

The pinned [direct PCGC routine](https://github.com/omerwe/PCGCs/blob/fdc5089f485fe25c04a8972665fee4216570764f/deprecated/pcgcs_direct.py)
deletes one **person** at a time. It removes every ordered pair involving that
person from the numerator and denominator. The pinned
[S-PCGC routine](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_main.py)
deletes **SNP blocks**, renormalizes the retained annotation score sums, and
holds its full reference normal matrix and supplied diagonal intercept fixed.
SUMMIT's binary path deletes target SNP rows from both score and reference
equations, removes their exact diagonal contributions, and keeps source
kernels and source masses fixed. These are different deletion statistics.

`compare_official_jackknife.py` fetches the pinned S-PCGC source, verifies its
SHA256, and executes the actual `SPCGC.compute_taus` function in isolation. An
adapter makes its full equations equal to SUMMIT's full equations. It also
compares every upstream deletion against an independent implementation of
the fixed-matrix rule. This isolates the jackknife arithmetic; it does **not**
claim to run the entire S-PCGC input/reference pipeline.

That distinction also matters for phenotype preparation: the inspected
[S-PCGC summary creator](https://github.com/omerwe/S-PCGC/blob/5211d173de45c6a88151928892581c058ba593cd/pcgc_sumstats_creator.py)
uses the globally standardized phenotype in `z_coeff`, whereas the direct
covariate-aware routine and our current moment contract use the conditionally
standardized response. The comparison supplies identical raw scores and risk
weights deliberately; it does not silently substitute the upstream phenotype
transformation or its reference approximation.

The source identities used by the executable comparison are:

| Source | SHA256 |
|---|---|
| S-PCGC `pcgc_main.py` | `f081d2a3d9f3b65b6a34e0a98be0f6321b83c4a7d0814e3939ce0f453bd08e57` |
| Direct PCGC `pcgcs_direct.py` | `e84fcb690c4376b177672d9c28a68b48a746067c0bb8ee5b7c79a65353d8f26a` |

## An exact check of the SE estimator, without Monte Carlo error

Let `F = diag(d) X`, and let `t_j` be the exactly diagonal-corrected score
square. For independent Bernoulli outcomes with their true individual risks,
conditional on X and d,

```
U = (F.T F)^2 - (F^2).T (F^2)       # squares are elementwise
Cov(t | X, risks) = 2 U
theta_-b = C_b t
E[V_JK | X, risks] = (B-1)/B sum_b 2 (C_b-C_bar) U (C_b-C_bar).T
```

The matrices C include the actual retained-target/full-source normalizations
and solves. An exhaustive unit test enumerates all 64 binary outcomes for six
people with unequal risks and overlapping annotations. It verifies the
expectation of the **jackknife covariance**, extending the earlier test of
the point estimator's covariance.

On the original 4,000-person/4,000-SNP null designs, the exact expected SNP-JK
variance divided by the exact conditional point-estimator variance is:

| Original seed | 50 blocks | 100 blocks |
|---|---:|---:|
| 827020 | 1.00079 | 1.00291 |
| 827065, the previous largest null outlier | 1.00074 | 1.00266 |

This rules out a missing universal multiplicative correction in those
conditional designs. It does not establish normal-interval coverage under
fixed-case-count recruitment, non-null genetic effects, fitted risks, or
reference/probe uncertainty.

The direct person-jackknife function is also executed and checked against an
independent pair-deletion calculation. They agree to floating-point precision.
In a 40-person independent-Bernoulli null example, its exact expected variance
is 2.09 times the true variance. This illustrates the known difficulty of an
ordinary person jackknife for a degenerate pair statistic near the null; it
is not an assessment of all published PCGC simulation settings. Adopting that
routine would not itself establish calibration for SUMMIT's SNP jackknife.

## Paired comparison on the original difficult simulations

The complete comparison replays the original 80 datasets per scenario, with
4,000 people, 4,000 SNPs, exact references, supplied/fitted risks, and 50/100
blocks. Full estimates reproduce the previous audit exactly; the largest SE
reproduction difference is `2.78e-17`. Every upstream full fit and deletion
passes the independent arithmetic comparison.

RMS SE divided by empirical SD, and nominal 95% normal coverage out of 80:

| Scenario | Risks | Blocks | SUMMIT SE/SD | Upstream rule SE/SD | Coverage, SUMMIT / upstream |
|---|---|---:|---:|---:|---:|
| S4: rare trait, strong covariate | supplied | 50 | 0.861 | 0.862 | 73 / 74 |
| S4 | fitted | 50 | 0.935 | 0.936 | 75 / 75 |
| S4 | supplied | 100 | 0.871 | 0.912 | 74 / 75 |
| S4 | fitted | 100 | 0.944 | 0.988 | 75 / 76 |
| S7: null, strong covariate | supplied | 50 | 0.861 | 0.860 | 74 / 74 |
| S7 | fitted | 50 | 0.861 | 0.860 | 74 / 74 |
| S7 | supplied | 100 | 0.879 | 0.880 | 75 / 75 |
| S7 | fitted | 100 | 0.880 | 0.880 | 75 / 75 |

The remaining deficit is therefore not specific to SUMMIT's deletion
implementation. In particular, replacing it with the actual upstream rule
does not fix the difficult null result. All eight SUMMIT SE/SD bootstrap
intervals include one. The supplied-risk S4 coverage test has one-sided
binomial p=0.105 with 50 blocks; the 50-block S7 result has p=0.211. These are
not evidence of equivalence or proof of calibration. They show why failing
our conservative qualification screen should not have been described as
establishing a SUMMIT-specific implementation defect.

The slightly larger upstream non-null SEs with 100 blocks are not evidence
for an omitted universal scaling. This simulation alternates 40-SNP LD blocks
with correlations 0.2 and 0.7. Fifty jackknife groups combine both LD types;
100 groups separate them. Holding H fixed can then create variation even in
noise-free score expectations. For a deterministic version with marginal
coefficient 0.125, `L_j = sum_k rho^(2 |j-k|)` within each LD block and score
expectations `t_j=N^2 L_j theta/M`, the fixed-H deletion SE is zero for 50
groups and 0.00561 for 100. Correctly deleting target H rows gives zero, up
to floating-point precision, at both counts. This calculation demonstrates
one mechanism for the difference; it does not allocate the entire observed
non-null discrepancy to that mechanism.

No additional jackknife algebra error was identified. The source/target
normalization, per-SNP diagonal subtraction, unequal-risk weighting, arbitrary
block labels, covariance aggregation, and marginal conversion are covered by
independent checks. Existing constant-risk tests also verify equality of
liability, standard PCGC, and inverse-PCGC point estimates and SEs. No SE
inflation factor, risk refit, change of resampling unit, or replacement
jackknife is introduced. **SEs remain experimental**, because these checks do
not qualify every non-null/covariate/annotation configuration.

## Reference specialization

The single-feature binary reference now uses one F-order source arena:

```
Pass 1: S_a = X sqrt(A_a) Z,       D = X^2 A
Barrier: S_a <- diag(d^2) S_a
Pass 2: L_ja = mean_b (X_j.T S_ab)^2 / N^2
                  - X_j^2.T [d^4 D_a] / N^2
```

The same counter-based variant probes are used for every annotation. The
source arena is weighted in place once, so no second contextual-source arena
or repeatedly weighted right-hand-side copies are needed. Pass 2 batches raw
trait scores and exact diagonal products. Each diagonal reference row is
subtracted immediately, removing the separate persistent M-by-K correction
array. The small same-person diagnostic comes from the pass-1 D accumulator,
so the generalized executor's second computation of kernel diagonals is also
unnecessary.

For disjoint annotations, pass 1 compacts only SNPs belonging to each bin.
Its leading source multiplication count falls from `2 N M K B` to at most
`2 N M B`. Dense overlapping annotations use batched source products; empty
annotation/block combinations are skipped. Pass 2 still costs `2 N M K B`.
This removes redundant work without a new reference approximation.

The implementation reuses SUMMIT's descriptor-owned BED/PGEN reads, affine
population scaling, native probe generator, protected NN/TN kernels, thread
placement guards, two-pass ledger, and existing inference reducers. It does
not use the native generalized BED executor directly: that executor's scale
and diagonal-statistic interface is not the binary contract. Cross-trait
multi-feature work retains the existing generalized path.

The planner admits source, score, decode, diagonal, and output-copy buffers
before their large allocations. File-reader preparation is deferred until the
first admitted traversal, and receives the admitted block capacity. This also
fixes the previous possibility of allocating a decoder for the requested block
size before a tighter reference plan selected a smaller block.

Under constrained memory, the planner balances genotype and probe tile sizes
instead of reducing genotype blocks to one SNP first. The number and identity
of probes are unchanged. For example, a 1-GiB single-annotation plan at the
full dimensions below selects 16-SNP/16-probe tiles rather than one-SNP tiles.
A larger memory budget remains appropriate for throughput.

With N=300,000, M=454,000, 256 probes, eight threads, and a preferred block width
of 256, the plans are:

| Annotations | New complete workspace, GiB | Previous generalized reference only, GiB |
|---:|---:|---:|
| 1 | 3.79 | 2.44 |
| 8 | 8.54 | 14.66 |
| 24 | 19.40 | 35.82 |
| 100 | 70.99 | 136.34 |

The previous column excludes the PCGC score/diagonal buffers. The new planner
reserves more decode/affine scratch explicitly, so its one-annotation bound is
larger despite using fewer source arrays. These are conservative workspace
plans, not measured process RSS. Caller metadata and unrelated runtime memory
remain additional. There is no N-by-N or M-by-M production allocation.

At 100 annotations, the plan demonstrates bounded memory, not statistical
qualification of that annotation design or adequate precision from 256 probes.
Probe counts must still be checked against component SEs, and strongly
overlapping or sparse annotations can be unidentifiable.

## Measured performance and verification

Private-BLIS benchmarks used four physical cores, process-local THP
disablement, explicit singleton OpenMP places, and SUMMIT's authenticated BLIS
worker-placement contract. The small benchmarks alternate engine order across
three repeats and retain identical genotype, annotation, response and probe
inputs.

| Design | Previous median, seconds | New median, seconds | Speedup |
|---|---:|---:|---:|
| N=4,000, M=4,000, eight disjoint annotations, 128 probes | 9.70 | 3.43 | 2.83x |
| Same dimensions, overlapping continuous annotations | 9.72 | 4.29 | 2.26x |
| Real BED: N=291,273, M=2,048, eight partitions, 64 probes; one paired run | 165.62 | 58.13 | 2.85x |

The real-BED run uses the existing population-frequency genotype scale and
synthetic positive risk weights/responses. It is an execution comparison, not
a new hypertension analysis. Its LD-row relative discrepancy is `3.14e-16`,
same-person discrepancy `1.27e-15`, and score-response discrepancy zero. The
new engine records exactly two passes, 4,096 retained-variant visits, no
duplicates, no retries, no numerical repairs, and no integrity failures.
Its planned workspace is 4.14 GiB versus the previous 7.11 GiB. Peak process
RSS across the paired process is 3.93 GiB; this is not a separate peak-RSS
measurement for each engine.

The final tile-planning refinement leaves the measured full-N plan and all
arithmetic unchanged; its dimensions, tiling and planned byte count were
checked against the benchmark record. Full 300k-by-454k **joint throughput has
not been measured**. The reference still has O(N M K B) target work; full-scale
wall time and probe accuracy for a chosen annotation design require a
representative production run. The smaller benchmark timings are not an ETA.

Final verification passed **163 portable OpenBLAS tests** and **55 focused
private-BLIS tests**, without skips. These cover native BED/PGEN preparation,
population scaling, both old/new reference equations, tile/probe boundaries,
weighted partitions, overlapping annotations, decoder admission, artifacts,
CLI behavior, cross-trait calculations, and the existing HE/LDSC/GxE paths.
The allocation-guard test was then updated to instrument the new arena
allocator and passed separately. Syntax checks and `git diff --check` passed.

The combined numerical report is `report.json`; its input hashes identify
the paired comparison, original audit, null calculation, and three benchmarks.

## Reproduction

All drivers refuse to replace existing result files/directories. The paired
comparison captures implementation hashes before execution and uses at most
100 original seeds per scenario.

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts/pcgc \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 taskset -c 56-57 \
  python scripts/pcgc/compare_official_jackknife.py \
  --out NEW_COMPARISON_DIRECTORY --scenarios S4 S7 \
  --replicates 80 --blocks 50 100

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts/pcgc \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 62 \
  python scripts/pcgc/diagnose_jackknife_null.py --out NEW_NULL_JSON
```

For the real-BED benchmark, this session used the summit Conda Python,
`LD_PRELOAD=/home/bronsonj/anaconda3/envs/summit/lib/libstdc++.so.6`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=4`, `BLIS_NUM_THREADS=4`,
`OMP_PROC_BIND=true`, `OMP_PLACES='{44},{45},{46},{47}'`,
`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, and `taskset -c 44-47`:

```
python -B scripts/pcgc/benchmark_reference.py \
  --extension-dir /data1/bronsonj/SUMMIT-cross-trait-response-covariance/build/pgs_optimization_private \
  --auxiliary-extension-dir /data1/bronsonj/SUMMIT-cross-trait-response-covariance/build/private_blis \
  --out NEW_BENCHMARK_JSON --samples 291273 --variants 2048 \
  --annotations 8 --probes 64 --threads 4 --repeats 1 --real-genotypes
```

CPU availability must be checked before reusing those core IDs. Without
`--real-genotypes`, the benchmark uses bounded synthetic arrays. The public
binary-method flags and artifact schema are unchanged; preparation selects
the specialized single-feature path automatically.
