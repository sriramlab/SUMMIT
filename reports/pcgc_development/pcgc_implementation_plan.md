**PCGC support in SUMMIT: implementation and qualification plan**

Date: 2026-09-26. Status: proposed implementation and experiments; no PCGC implementation or qualification is claimed by this document.

The objective is to implement and compare all five approaches discussed: constant-risk liability correction, standard covariate-aware PCGC, a PCGC-s-LD-style reference approximation, inverse-response moments, and a reusable covariate-basis reference. Public method choices will be restricted to approaches and input contracts supported by the results. Correctness and small-data statistical performance come first; large-data optimization is a subsequent phase.

The inspected checkout is `/home/bronsonj/SUMMIT`, HEAD `c53db60579b79e98b8b71ebc4bba9537b2be850c`. Its locally recorded `origin/main` is `e73c721f337dd4705dac165a06f061637adb1bdd`. These are observations of local Git state, not a fresh remote verification. The checkout contains existing changes to generalized GxE code, tests, and runners. Before implementation, choose and record the intended baseline and use an isolated worktree; preserve all existing work. This document does not authorize replacing those changes.

**1. Fix the scientific contract before adding a public flag**

Start with unrelated individuals, a liability-threshold model, disease-dependent sampling, a common retained SNP set, and population-referenced genotype scaling. Establish univariate heritability first, then genetic covariance and correlation. Cohort meta-analysis with heterogeneous ascertainment, familial sampling, rare-variant architectures, and arbitrary mixed-model GWAS inputs are outside the initial validated contract. They must not be silently accepted as equivalent inputs.

Use distinct notation for population prevalence `K`, sample case fraction `P`, individual population risk `k_i`, and individual sampled risk `p_i`. With the control-to-case sampling ratio `a = K(1-P)/(P(1-K))`, the basic sampling model gives `p_i = k_i / [k_i + a(1-k_i)]`. This identity assumes sampling depends on disease status as specified; additional recruitment dependence requires its own sampling model.

For standardized residual liability variance one, define

```math
t_i=\Phi^{-1}(1-k_i),\qquad
z_i=\frac{y_i-p_i}{\sqrt{p_i(1-p_i)}},\qquad
d_i=\frac{\phi(t_i)\sqrt{p_i(1-p_i)}}{k_i(1-k_i)}.
```

The first-order PCGC moment is

```math
E[z_i z_j]\simeq d_i d_j\sum_a\theta_a K_{a,ij},\qquad i\ne j.
```

Here `K_a = X A_a X^T / M_a`, with one explicit genotype scale and `M_a = sum_j A_ja`. The approximation is in liability correlation. An exact implementation of these equations is not an exact finite-correlation liability likelihood. Close relatives and concentrated large effects therefore require separate checks.

Record conditional genetic variance, conditional heritability, and marginal liability heritability separately. Under the independent covariate/genetic model with residual liability variance one and covariate variance `V_C`, marginal heritability is `sum(theta)/(1+V_C)` when the population-normalized component kernels have unit mean diagonal. Use the corresponding population trace factors otherwise. For genotype-covariate dependence, derive the conditional target and population variance decomposition explicitly instead of assuming this formula. Do not call an observed binary variance component liability heritability.

Standard PCGC's separable weights and diagonal exclusion are documented in the authors' [direct implementation](https://github.com/omerwe/PCGCs/blob/master/deprecated/pcgcs_direct.py) and [risk transformation](https://github.com/omerwe/PCGCs/blob/master/deprecated/pcgcs_utils.py). The constant-risk correction follows [Golan et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267399/). The distinction between exact study summaries and external-LD approximations is developed by [Weissbrod et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6035374/).

**2. Use a common moment engine, with explicit method differences**

The following unification is a proposed SUMMIT design derived from the PCGC moment, not a claim that the current code implements it. For internal `alpha` equal to one or zero, let

```math
W_\alpha=\operatorname{diag}(d^\alpha),\quad
B_{a,\alpha}=W_\alpha K_aW_\alpha,\quad
r_\alpha=d^{\alpha-1}\odot z.
```

Both choices satisfy the same first-order kernel moment form. Their full-data normal equations are

```math
H_{ab}=\operatorname{tr}(B_{a,\alpha}B_{b,\alpha})
       -\sum_i B_{a,\alpha,ii}B_{b,\alpha,ii},
\qquad
b_a=r_\alpha^T B_{a,\alpha}r_\alpha
       -\sum_i B_{a,\alpha,ii}r_{\alpha,i}^2.
```

The study-side SNP score is `s_j = x_j^T (d^(2 alpha - 1) * z)`. Thus both methods can use the same batched linear cross-product executor and exact squared-genotype diagonal reductions. The small solver consumes explicit `H` and `b`; it must not append the existing quantitative phenotype-variance equation.

| Proposed method name | Statistical definition | Reference computation | Required comparison |
|---|---|---|---|
| `liability` | Constant-risk off-diagonal HE divided by the known PCGC factor | Ordinary kernel moments | Equal constant-weight PCGC and inverse-response fits, including scale |
| `pcgc` | `alpha=1`, standard PCGC least-squares moments | Weighted kernel Gram and exact same-person terms | Independent dense pair regression |
| `pcgc-ld` | Standard PCGC numerator with the explicitly derived external-LD approximation to its denominator | Genotype-only LD plus study corrections | Exact PCGC and published PCGC-s-LD formulas on compatible inputs |
| `pcgc-inverse` | `alpha=0`, regress inverse-sensitivity response products on ordinary kernels | Ordinary kernel Gram and exact same-person terms | Its own dense regression; the target matches PCGC to first order, individual estimates need not |
| `pcgc-basis` | Standard PCGC evaluated through a shared covariate basis | Generalized contextual reference, contracted by trait coefficients | Equal direct weighted PCGC when the weights are represented exactly |

Do not expose `alpha` as another tuning parameter. The basis approach is a representation of standard PCGC, with approximation only when the basis does not span the risk weights. It does not justify freely estimating extra GxE variance components.

For basis columns `phi_q` and `d = sum_q c_q phi_q`, the weighted kernel is the sum of the existing symmetric contextual kernels with coefficients `c_q c_r` in SUMMIT's diagonal-first pair convention. Contract the Gram on both component axes and the RHS on one; contract the same-person terms as well. The existing off-diagonal kernel definition already contains both orientations: do not add another factor two. Test a low-rank coefficient contraction rather than fitting unconstrained contextual coefficients. Begin with exact stratum indicators. For continuous risks, compare a prespecified small basis and one richer basis, checking weight reconstruction, moment error, and estimate error. If approximation replaces `d` by `d_hat`, use that replacement consistently in the declared estimating equation; never mix an approximate denominator and an exact numerator without identifying that as a separate approximation.

For `pcgc-ld`, write the complete equations before implementing the adapter. A useful diagnostic is whether weighted off-diagonal Gram entries behave like a scalar risk-weight average times ordinary entries, but substituting `mean(d^2)^2` is not a complete implementation. Derive sample-size, finite-reference, annotation, same-person, population-scaling, and ancestry corrections. In particular, the finite-sample average over distinct pairs is `[(sum d_i^2)^2 - sum d_i^4]/[N(N-1)]`, not automatically `mean(d^2)^2`. Inspect and pin the relevant [S-PCGC estimator](https://github.com/omerwe/S-PCGC/blob/master/pcgc_main.py) and [reference calculation](https://github.com/omerwe/S-PCGC/blob/master/pcgc_r2.py). Use an independent formula comparison; do not make the old Python environment a production dependency. Start with binary disjoint annotations, then verify the mapping for weighted/overlapping annotations, whose conventions can differ between programs.

**3. Resolve three correctness hazards explicitly**

Risk estimation is separate from genotype nuisance projection. First support supplied known `k_i,p_i` for the mathematical oracle. Then implement an ascertainment-aware risk fit on phenotype/covariate data. A correctly specified sampled Bernoulli likelihood with `k_i` from a probit risk model is the primary simulation fit. The published logistic-fit/back-transformation shortcut is a compatibility comparison, since it need not be correctly specified for a probit generator. Estimate nuisance parameters without SNP effect predictors. Validate gradients, optimization convergence, rank deficiency, separation, intercept-only behavior, and sampled/population risk calibration. Use stable log-CDF/log-survival and log-odds calculations in the tails. Require externally supplied population prevalence; do not substitute sample prevalence. Report tail risk/weight diagnostics and fail on invalid probabilities; silently clipping risks or winsorizing inverse weights changes the method.

Projection requires a derivation, not a flag passed to GxE. Generally `P_C D X != D P_C X`. Moreover, projecting a response with diagonal heteroskedastic variance makes that nuisance covariance non-diagonal. Simply projecting the phenotype and then deleting the covariance diagonal is insufficient. Initially prove the unprojected, correctly risk-centered equations. Before admitting ancestry-adjusted use, derive the projection-compatible moment and diagonal corrections and compare to the original PCGC convention, including the interpretation after removing genotype PCs. Preserve the generalized engine's `P_C diag(phi) G` convention; if PCGC needs another operation order, add an explicit adapter with its own scientific identity rather than changing GxE semantics. A small counterexample with nonconstant weights and a noncommuting projector is mandatory. If the statistical mapping is not established, that configuration remains unsupported.

Reference transfer is a separate approximation. Test an exact same-study Gram first. Next test an independent reference generated under the same sampling/covariate distribution, and then the practically relevant population reference. An ancestry match alone does not guarantee a match in the joint distribution of genotype, risk covariates, and ascertainment. Weighted methods need reference covariates or an explicitly validated approximation. The shared-basis reference is reusable only across compatible genotype scales, covariate definitions, and sampling contracts. Do not automatically apply the current sample-count transfer formula to every PCGC artifact, and do not assume response transformation removes reference mismatch.

Keep genotype centering/scaling separate from reference-moment estimation. Even a same-study Gram should initially use the known population allele/scale convention, so ascertainment-biased sample frequencies do not confound the method comparison. Then test scales estimated from an independent population reference. The existing native reference emits its observed affine scale; verify whether its input contract can accept the required sealed scale and, if necessary, add that explicit capability without altering existing defaults. Re-estimating a separate scale for each weighted feature would define a different kernel.

**4. Integration with the existing implementation**

Keep the changes additive and small. Proposed new files below are a design, not existing modules.

| Responsibility | Proposed home | Existing machinery to reuse |
|---|---|---|
| Risks, ascertainment, scale metadata | `src/summit/sumstats/binary.py` | NumPy/SciPy; existing covariate alignment and rank utilities |
| Typed PCGC study scores and reductions | `src/summit/sumstats/pcgc.py` | Existing genotype input/alignment, block reads, batched score and squared-genotype products |
| Reference adapters and basis contraction | `src/summit/ldscore/pcgc_reference.py` | Generalized variant-probe kernels, exact component diagonals, annotation reducers; ordinary Trace moments where justified |
| Full/block equation assembly and inference | `src/summit/inference/pcgc.py` | Existing rank diagnostics, linear algebra, jackknife design and result conventions |
| Dispatch and admission checks | Existing `src/summit/cli.py` | Existing `--h2`, `--rg`, batch interfaces and logging |
| Independent tiny oracle | `tests/pcgc_oracle.py` | Test-only NumPy/SciPy, without calling production transformations or reducers |
| Simulation and comparison runner | `scripts/pcgc/` | Existing seed/config/result patterns, plus a binary-trait generator |

The current `sumstats/moments.py` reconstructs OLS score quantities from beta and SE. The current `inference/h2core.py` includes a fixed quantitative variance equation. The generalized study routines project and normalize their input phenotype. Add an explicit raw-score path below those transformations, preferably by extracting the smallest reusable block cross-product routine. Preserve current defaults and test them against the pre-change baseline. Arbitrary logistic, SAIGE, or mixed-model beta/SE files cannot be relabeled PCGC summaries.

Use an explicit binary artifact identity containing method/moment family, liability scale, prevalence and ascertainment model, risk-fit specification, sample/SNP/allele axes, genotype affine scale, annotation masses, projection convention, diagonal convention, reference role, and uncertainty convention. Follow existing concrete compatibility checks. Store aggregate corrections and per-SNP score rows needed for future fits; do not publish individual risks or sample-aligned kernel diagonals in a public summary artifact. Internal basis artifacts can retain the existing explicitly internal composable representation.

Compute all required score vectors for compatible methods/traits together while each genotype block is resident. Compute diagonal corrections with squared-genotype products in that traversal. Reuse genotype decoding, imputation, allele alignment, BLAS, native thread contracts, and memory admission. Do not create another BED/PGEN decoder. The observed generalized study path is BED-specific; use the existing genotype-source abstraction where possible, but do not claim that a generic resolver alone establishes full PGEN/dosage support. Qualify supported formats independently.

The generalized reference remains a variant-axis, two-pass estimator. No jackknife argument enters either genotype pass. Keep the separate sample-probe contextual estimator distinct; it can be a small test comparator but cannot be substituted under the production artifact identity. Exact pairwise `N x N` or SNPwise `M x M` matrices are confined to small oracles.

**5. Proposed practical interface**

Add one method-selection flag, provisionally `--binary-method`, with the qualified subset of the names in section 2. Omitting it preserves existing behavior. Keep it separate from `--weight-mode`, which currently selects HE versus LDSC equations; reject unsupported combinations instead of ignoring one option. Do not add automatic method selection before there is evidence supporting it.

Examples below describe the intended interface and are not runnable commands today:

```text
summit --h2 <binary-summary> --binary-method pcgc --ldscores <compatible-reference> ...
summit --h2 <binary-summary> --binary-method pcgc-inverse --ldscores <ordinary-reference> ...
summit --rg <binary-pair-manifest> --binary-method pcgc ...
```

The preparation stage must also receive the method or requested compatible score families. A proposed `--make-binary-sumstats` action would combine genotypes, phenotype, covariates, population prevalence, and the selected reference/scaling specification; its output feeds the existing inference workflow. Analysis of prepared summaries does not require individual-level data. A method flag cannot recover missing ascertainment-aware scores from ordinary published GWAS statistics. Allow compatible score families to coexist in one preparation artifact without manufacturing fake OLS beta/SE columns.

Use artifact metadata for prevalence, sample fraction, and scale; supplied overrides must agree or trigger an explicit recomputation requirement. A `--liability-scale` choice can select conditional versus marginal output when the required population variance information exists. For genetic correlation, attach each trait's risk/scale metadata to its existing manifest entry. Print method, reference approximation, estimand, and uncertainty assumptions in the result metadata. Explain missing-input errors in terms of the needed preparation step.

Initially expose experimental methods through the development runner/private API. Promote only qualified method/input combinations. `pcgc-basis` may ultimately be a reference backend choice for `pcgc` if it is exact and user-equivalent; retain separate labels in the comparison report so its approximation is visible.

**6. Validation ladder: fail cheaply before expanding simulations**

| Stage | Proposed size and work | Required evidence before advancing |
|---|---|---|
| A: analytic and algebraic | Pair probabilities; `N=8..40`, `M=6..80`, 1-3 annotations | Independent equation identities and sampling derivatives |
| B: integrated exact fixtures | Approximately `N=128..256`, `M=256..512`; a few seeds | Dense pairs, SNP summaries, artifact round trip and streamed/native paths agree |
| C: randomized reference | Same small fixtures; 16/64/256 probes with 8 fixed seeds | Native/Python fixed-probe parity and measured error against exact Gram |
| D: statistical pilot | First 20 replicate seeds per selected scenario, beginning with 5 | No deterministic failures; usable signal/noise; measured cost and numerical error |
| E: confirmation | Up to 80 additional seeds per scenario | Bias, precision and uncertainty evaluated on a frozen protocol; at most 100 total per scenario |

Stage A independently enumerates binary pairs under a bivariate-normal liability model, applies the sampling probabilities, and differentiates around zero correlation. Check the `d_i d_j` derivative, the inverse-response derivative, the constant-risk limit, and finite-correlation approximation error. Use deterministic quadrature rather than a noisy random CDF routine. Vary risk pairs and sampling ratios in these essentially free calculations.

Independently compare pairwise regression, trace-minus-diagonal moments, and SNP-score reductions. Check all components of `H,b`, not only the solved heritability. Include non-unit kernel diagonals, nonconstant risk weights, overlapping/continuous annotations, reordered alleles, a sign flip, duplicated and rank-deficient annotations, and a zero-mass annotation. Match full-data and declared block statistics. Test conditional/marginal scale conversion and case/control recoding with the prevalence and sampling model transformed consistently.

Use tight scale-aware FP64 tolerances (initially `rtol=1e-10`, `atol=1e-12` for well-conditioned small fixtures). Ill-conditioned fixtures must produce the documented rank/conditioning response; they do not justify loosening every tolerance. Test exact basis contraction for two/three strata and continuous-basis approximation separately. Compare the published no-covariate and covariate PCGC formulas on a few fixtures after aligning every convention. Agreement with external code is an additional check, not the only oracle.

Stage B exercises actual genotype files and existing read paths, with monomorphic SNPs, missing calls, sample/SNP subsets, and covariate alignment. Include an empty genotype nuisance basis and an intercept-only basis explicitly; a general matrix API does not by itself establish that either edge case works. Invalid probabilities, failed risk fits, incompatible reference metadata, and insufficient summaries must fail before a genotype scan or fit where feasible. Verify all advertised CLI paths and output scale labels. Add explicit regression tests that ordinary HE/LDSC, genetic correlation, and generalized GxE results remain unchanged when the new flag is absent.

Stage C separates two questions: a fixed randomized estimator must agree between implementations; its Monte Carlo error against the exact estimator must be assessed across probes/seeds. Do not assert that each seed improves monotonically as probes increase. Include error in the off-diagonal Gram after diagonal subtraction, which can be much more sensitive than error in the total trace. Shared probes alone do not guarantee identical finite-probe behavior after a change of basis representation; verify exact contraction and any claimed fixed-probe equivalence separately. Never repair an indefinite/noisy Gram by undocumented PSD clipping or ridge.

After the pilot supplies an empirical statistical SE, require probe-induced estimate RMS variation to be at most 10% of that SE for the primary comparison. If necessary increase probes only for affected configurations. Retain an exact small-data comparison so reference Monte Carlo noise cannot conceal an estimator bug.

**7. A compact simulation design**

Use the same simulated dataset, risk-fit inputs, reference realization, and compatible probes for all methods in each replicate. Method comparisons are paired. Reuse an expensive reference only when its genotype, sample, scaling, covariate-basis and weighting contract actually remains identical. In particular, estimated risks and ascertained sample membership may vary between replicates, making a weighted reference non-reusable.

The starting size is `N_study=4,000`, `M=4,000`, and an independent reference of `N_ref=2,000` where relevant. These are proposed sizes, not runtime promises. Use 100 independent LD blocks of 40 variants, alternating modest and stronger within-block correlation, initially AR(1) correlations 0.2 and 0.7. Use one default block jackknife with 50 blocks formed by pairing adjacent independent LD blocks. Calibrate whether this size gives useful SEs in the first five replicates. If precision is inadequate, change one information dimension for the affected scenario before freezing confirmation; do not compensate with an enormous grid.

For the basic conditional liability model set `Var(g+e)=1`. The main alternative has conditional genetic variance 0.25. A strong categorical covariate is balanced in the population with contribution `+/-0.5`, giving covariate variance 0.25; the continuous analogue is `0.5*C`, `C~N(0,1)`. Solve the threshold to obtain the requested population prevalence. Thus the independent-covariate main alternative has marginal heritability 0.20. Under ascertainment, do not force the population covariate distribution to remain balanced in the study.

| ID | Population K | Sample P | Genetic variance and risk design | Purpose |
|---|---:|---:|---|---|
| S0 | 0.10 | 0.10 | 0.25; no risk covariates | Population sampling and constant-risk identity |
| S1 | 0.10 | 0.50 | 0.25; no risk covariates | Ascertainment alone |
| S2 | 0.10 | 0.50 | 0.25; strong binary risk covariate | Standard PCGC, exact stratum basis, and a negative control for scalar-only correction |
| S3 | 0.10 | 0.50 | 0.25; continuous risk covariate | Inverse-response precision and approximate basis |
| S4 | 0.01 | 0.50 | 0.25; strong risk covariate | Rare disease, extreme sampling, risk-weight sensitivity |
| S5 | 0.10 | 0.50 | Two disjoint components, 0.05 and 0.20; unequal LD | Partitioned variance and enrichment |
| S6 | 0.10 | 0.50 | 0.25; population strata influence genotype and risk | Projection/conditional target and reference-transfer assumptions |
| S7 | 0.10 | 0.50 | Zero genetic variance; strong risk covariate | Bias and type-I-error calibration |

All eight receive the cheap pilot after algebraic gates pass. Begin confirmation with S2, S3, S4 and S7: these distinguish the methods and test uncertainty. Confirm additional scenarios when they are needed for the advertised scope. At 20 pilot plus 80 confirmation replicates, no scenario exceeds 100; the eight-scenario ceiling is 800 independently generated datasets, shared across methods, not 800 per method. Do not run this ceiling automatically. For release metrics use the confirmation results separately from development results; keep all failures in the accounting.

Use true supplied risks and fitted risks on the same datasets to separate the moment estimator from nuisance estimation. This adds fits and score columns, not another phenotype grid. Include the unmodified current SUMMIT HE fit followed by the global liability conversion as a baseline, alongside the mathematically defined off-diagonal `liability` method. Their difference isolates finite-sample equation/diagonal handling; neither is assumed equivalent to covariate-aware PCGC in S2 onward. For S2/S3/S6, compare same-study, independent design-matched, and population references. Start with these few sentinel cases rather than crossing reference type with every parameter. Repeat a small number of fixed datasets with alternative reference/probe seeds to quantify numerical/reference sensitivity; these are repeated computations, not independent statistical replicates.

Use two independently validated generators. The first uses Gaussian marker blocks with known covariance and liability effects, allowing exact population genetic variance and efficient conditional sampling of cases/controls. For fixed effects beta, draw liability in the appropriate truncated stratum distribution and draw markers from their Gaussian conditional distribution; validate against ordinary rejection sampling at common prevalence. This avoids generating millions of genotype rows solely to obtain rare cases. Sample risk strata conditional on case/control status correctly. For continuous covariates, use validated low-dimensional sampling/quadrature; do not invent a shortcut that breaks ascertainment-induced dependence.

The second generates discrete 0/1/2 genotypes from an independently specified haplotype/block model with known allele probabilities. Use streaming population generation and case/control selection on modest common-prevalence S2/S5 analogues, initially 20-30 replicates, rather than applying rare-disease rejection sampling everywhere. Validate allele frequencies, LD and prevalence before comparing estimators. These cases cover genotype realism, missingness and the actual file path. A thresholded Gaussian liability with Gaussian genotypes alone is insufficient qualification of the production input path.

Define truth from the population generator, never from an ascertained sample variance. For Gaussian fixed-effect simulations, compute `beta^T R beta` analytically; normalize effects using population quantities when imposing a fixed realized variance. For S5, use independent LD-block groups so component variances can be assigned separately. For random-effect simulations, report the generating variance-component target separately from realized effect variance; do not alternate between these definitions when assessing bias. S6 needs an explicit conditional genotype model and variance target, with a deliberately mismatched analysis labeled as such.

Use a few targeted sensitivity tests only after the core works: population prevalence misspecified by a factor of two, a nonlinear omitted risk covariate, and an independent reference with shifted covariate/ancestry composition. Choose at most two relevant core datasets/scenarios and approximately 20-30 replicates each. These define failure boundaries, not extra conditions that every method must magically pass. A sparse-effect or relatedness stress check is useful for the first-order approximation but does not broaden the initial public scope.

**8. Uncertainty is a separate release gate**

Keep raw variance-component estimates, including negative estimates. Do not clip to [0,1] before computing bias, MSE, coverage or null rejection. Evaluate genetic covariance directly when heritability is near zero; do not turn undefined genetic correlations into zero.

Reuse the established post-hoc SNP-block machinery: complete per-SNP statistics first, then reduce fixed target rows. Preserve `frozen_full_genome_variant_ldscore_delete_block_v1` and the current generalized reference rule that reuses the full same-person matrix. Specify the PCGC RHS diagonal subtraction and its retained-mass convention just as explicitly. Compare every replicate equation to an independently coded implementation of that declared surrogate. It is not exact deletion of both sides of a kernel. Exact deleted-kernel calculations on tiny fixtures are a separate diagnostic, not an equality the existing surrogate can be expected to satisfy. If this uncertainty convention is uncalibrated for PCGC, do not change the existing GxE behavior or disguise a different estimator under its identity; develop a separately identified PCGC correction and validate it before exposure.

Assess nuisance-fitting uncertainty by comparing supplied-risk and fitted-risk calibration. The SNP-block jackknife does not refit individual-level covariate risks and does not automatically capture uncertainty in externally supplied prevalence. On a small diagnostic subset, use independent whole-study simulation replicates and, only if needed, a bounded participant bootstrap with risk refitting to locate missing uncertainty. Do not treat participant bootstrap as an unquestioned gold standard for all variance sources. If a correction is needed, derive a joint influence/sandwich correction including risk-estimation cross-covariances, or another explicitly validated resampling scheme. Adding a nuisance variance in quadrature without cross terms is not sufficient. Cross-fitting is an option to investigate if estimated risks cause bias, not a mandatory expensive default. Prevalence sensitivity remains separate unless an uncertainty model for prevalence is supplied.

Likewise distinguish inference conditional on a supplied reference/probe realization from inference over repeated references. Check study-to-study coverage for a fixed reference and total variability across independently regenerated references on a small sentinel set. Do not reuse one LD realization across all confirmation replicates and describe the resulting coverage as unconditional. If reference uncertainty is material, either incorporate it or state the restricted inference contract and withhold unsupported small-reference use.

Collect bias, empirical SD, RMSE, median and RMS reported SE, 95% coverage, two-sided 5% null rejection, failure rate, condition diagnostics, risk-weight tails, and stage-specific time/peak memory. Report paired estimate differences and paired squared-error differences between methods. Include numerical/probe error separately from statistical error.

Set proposed scientific tolerances before confirmation: for total h2, an absolute bias margin `max(0.02, 0.10*truth)`; for individual variance components and genetic covariance on their declared scale, `max(0.01, 0.10*abs(truth))`; for well-defined rg, 0.05. Use a confidence interval for mean bias with Monte Carlo SE `SD(error)/sqrt(R)`; a non-significant bias test alone is not evidence of equivalence. An interval contained within the margin supports qualification, one outside identifies failure, and an overlapping interval is inconclusive. Diagnose enrichment using its own uncertainty rather than only total h2. These are proposed tolerances to freeze with the protocol, not claims of already demonstrated accuracy.

With at most 100 replicates, 95% coverage and 5% type-I-error estimates have considerable binomial uncertainty. Report exact or Wilson intervals and avoid claiming fine calibration from an observed 95/100. As a coarse release screen, require compatibility with the nominal rate and evidence against severe miscalibration (coverage at or below 85%, or false-positive rate at or above 15%). Also inspect RMS(SE)/empirical SD, with a proposed central acceptable range 0.8-1.25 and an uncertainty interval. These are modest-simulation screens, not universal accuracy guarantees. Borderline results remain inconclusive; they do not trigger more than 100 replicates or relaxed post-hoc thresholds.

Do not promote a less efficient method merely because it is unbiased. Compare inverse-response and approximate-reference methods with standard PCGC using paired MSE and runtime. A method can be offered for a documented computational tradeoff without being the default. No default winner is chosen before the results.

**9. Genetic covariance and correlation follow the univariate gates**

Derive the rectangular cross-study moment with trait-specific `d_1,d_2`, genotype scales, sampling probabilities, and kernels. Remove same-person contributions only for actual sample overlaps using aligned IDs and the required transformed score/diagonal products. A count of overlapping controls or the current ordinary phenotype covariance alone is generally insufficient. Carry each trait's marginal/conditional liability scaling through covariance and both variance estimates before taking a ratio.

Start with independent studies and then shared controls. Add a binary-quantitative pair with the quantitative trait's standard linear score convention. Validate cross-products, sign/allele orientation, zero overlap, full overlap, and pair exchange symmetry in tiny fixtures. Restrict initial bivariate sampling to a specified design; ascertainment on both traits simultaneously requires a joint selection derivation. Do not import univariate risk weights into a joint-selection simulation and assume validity.

Use three compact bivariate confirmation designs after h2 passes: binary-binary `rg=0`, binary-binary `rg=0.5` with a specified shared-control design, and binary-quantitative `rg=0.5`. Begin with 20 replicates each and expand only the needed designs to at most 100. Generate shared controls from the correct joint eligibility distribution; validate the sampler separately. Report covariance before rg, and record invalid-ratio frequencies. All methods are compared on the same trait pair within a design. Methods lacking a qualified overlap contract are not exposed for that setting.

**10. Work order, resource discipline, and deliverables**

Implement in reviewable increments: scientific contract and independent oracle; risk preparation and scale handling; constant-risk/standard/inverse exact moments; typed summaries and existing genotype-path integration; external-LD adapter; exact and approximate basis contraction; uncertainty qualification; public dispatch and bivariate extension. Keep the two main point estimators on one underlying reduction/solve path so bug fixes do not diverge across implementations.

Before increasing any run, benchmark one actual end-to-end replicate, including risk fitting, both reference passes, study scoring, all fits/jackknife, and publication. Measure peak memory, useful CPU time, physical genotype passes, and output volume. Multiply observed stage costs by the remaining tasks to decide whether a local batch is appropriate. Avoid speculative runtime claims. If Hoffman becomes necessary, first read `/home/bronsonj/UKBB/data_audit/06_compute_and_servers.md` and apply its allocation, binding and scratch-output rules; no cluster work is requested by this planning step.

Save one scenario/seed manifest, compact per-replicate result rows and diagnostics, a small comparison report, and only reusable fixtures/artifacts. Batch result publication rather than repeatedly rewriting large files. Use new output directories and do not overwrite existing results. Preserve the seeds needed to reproduce failures without retaining every dense matrix or population pool. Record implementation revision, method contract, generator truth, and numerical settings in every experiment manifest. If code changes after confirmation, identify which results are obsolete; do not tune repeatedly against the same confirmation results and call them untouched validation.

Take existing batching, exact diagonals, decode reuse, sufficient-statistic reduction and small batched solves immediately. Defer new kernels, caching schemes, precision changes and broad native refactors until a measured bottleneck appears. A proposed optimization must preserve full/per-SNP/block moments, pass counts, memory bounds and existing-feature tests, and improve end-to-end cost meaningfully on the representative small benchmark. Microbenchmark speed alone is insufficient. Qualify the portable OpenBLAS test path and relevant private-BLIS production path when changes touch their shared routines.

The deliverables are a documented statistical contract, independent oracles and regression tests, integrated score/reference/inference adapters, a reproducible compact experiment runner, and a comparison report with a method-by-input qualification matrix. That matrix governs the public flag choices. If all approaches fail a setting, the outcome is an identified modeling or uncertainty limitation, not selection of the least visibly biased result. Further scalability work begins after at least the standard PCGC path and its advertised uncertainty are established on the bounded experiments.
