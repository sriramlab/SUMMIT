# Independent-pilot calibration of the regional scale test

This procedure tests the same null as the full-coefficient Box--Cox test:
some power in the declared interval makes the tested interaction block zero,
conditional on all supplied additive, dominance, covariate and other
interaction terms. It preserves exact transformed-outcome fitting.

## Construction and validity

Let A denote pilot data independent of confirmation data B. Features must be
fixed independently of outcomes in both sets, or learned in a third sample.
Both sets must target the same population regression and use matching feature
and nuisance coordinates. Disjoint IDs do not establish independence in the
presence of relatives, shared random effects or outcome-dependent selection.

On A, invert the full-block pointwise HC3 tests at gamma to obtain

    C_A = {lambda : p_A,full(lambda) >= gamma}.

The implementation retains a continuous outer approximation: an interval
is discarded only when its entire p-value upper bound is below gamma.
Unresolved pieces and disconnected components are retained. If lambda0 is
a true null witness, P(lambda0 not in C_A) <= gamma + o(1), under pointwise
HC3 validity. Uniform validity across all sampling models is not implied.

Choose an anchor from pilot evaluations and estimate the derivative d of the
coefficient vector with respect to power. A training-only rule either retains
the full block or selects a full-rank contrast matrix L satisfying L d = 0.
The default projects out one direction when q>1 and its pilot derivative
Wald p<.01; otherwise it retains q coefficients. This threshold affects power,
not the size proof. Pilot selection and uncertainty in d are allowed because
L is fixed conditional on A and the null full vector is zero.

For each lambda in C_A, B fits the exact transformed outcome and computes

    eta_B(lambda) = L beta_B(lambda),
    W_B(lambda) = eta_B' [L V_B(lambda) L']^-1 eta_B,
    p_B(lambda) = chi2_rank(L).sf(W_B(lambda)).

At a true null power, L beta(lambda0)=0 for EVERY pilot-selected L. This does
not require d to be identified or lambda0 to be interior. A rotation of the
full feature basis implements these contrasts while retaining every original
effect in the model. The deleted direction is not removed from adjustment.

The reported upper bound dominates

    p_honest = min(1, gamma + sup_{lambda in C_A} p_B(lambda)).

Use sup(empty)=0. For alpha>gamma, the rejection probability is bounded by

    P(lambda0 not in C_A)
      + P(p_B(lambda0) <= alpha-gamma, lambda0 in C_A)
    <= gamma + (alpha-gamma) + o(1) = alpha + o(1).

The second term follows by conditioning on A. Dependence between the pilot
confidence set, anchor, projection and choice of full-block fallback does not
break the proof. Confirmation outcomes must not choose these quantities.
No Taylor remainder confidence bound, boundary mixture approximation, or
post-profile subtraction of a degree of freedom is used in this argument.

This uses the confidence-set nuisance-parameter principle of
[Berger and Boos (1994)](https://doi.org/10.1080/01621459.1994.10476836), with
independent-pilot contrasts and a certified numerical outer set.

## Power and limits

At a regular interior root, a consistent pilot direction annihilates the
first-order scale movement. If the pilot size is m, root/direction errors
are O_p(m^-1/2), and curvature is bounded, projected mean variation over
the shrinking pilot set is O_p(1/m). When sqrt(n)/m tends to zero, and the
numerical outer-set width shrinks sufficiently, the confirmation supremum
behaves like one chi2_(q-1) p rather than another fitted-scale statistic.
The small gamma reserve remains. At fixed gamma the asymptotic reference
thus targets approximately alpha-gamma, not exactly alpha. Gamma may shrink
with sample size only subject to tail/coverage and rate requirements.

At a regular boundary this projection can still yield the q-1 pointwise
reference, because the direction was learned independently. It is different
from minimizing a q-dimensional confirmation statistic and comparing that
minimum with q-1 degrees of freedom. The latter has a boundary-mixture limit
and can be anticonservative.

Weak identification, broad or disconnected confidence sets, unresolved
numerical bounds, or an uninformative pilot can reduce power. The procedure
does not promise an exact 5% rejection rate for every null. A frozen contrast
can discard some alternatives. Empty pilot sets give p=gamma and represent
pilot evidence against the full null, with its error budget included.

HC3 remains asymptotic and sensitive to dependence, leverage and effective
support. The proof is for a common population projection (or a correctly
specified conditional mean shared across fixed designs). Matching ancestry
labels alone does not establish that two different populations have the same
projection. Related-sample/polygenic covariance calibration and genome-wide
extreme-tail qualification are separate tasks.

## Public workflow

The NPZ contains `features` (N by p), `fixed_effects` (N by k, with an
intercept), positive `phenotype` (N), unique one-dimensional `sample_ids` (N),
and a prespecified Boolean `pilot_mask` (N). Supply the same encoding for all
rows. Feature learning on pilot phenotypes requires an independent third
sample; the two-way mask alone does not make that learning honest.

    python -m summit.epistasis.cli scale-test-honest arrays.npz \
      --out result.json --groups groups.json \
      --lambda-min -2 --lambda-max 2 --alpha 0.000625 \
      --gamma 0.0000625 --max-pilot-evaluations 257 \
      --max-evaluations 257 --num-threads 1 --memory-gib 8

Use the qualified platform launcher and CPU placement safeguards. `groups`
maps names to original feature indices; omit it for one omnibus test. Budget
gamma within the predeclared per-test alpha AFTER multiple-testing adjustment.
Increasing the numerical budget can tighten bounds without changing the null.

Outputs include the outer pilot sets, frozen contrasts, derivative selection,
confirmation bounds, support diagnostics, sample-order hashes and native
execution evidence. IDs are not written to the report. Results never
overwrite an existing output. The original `scale-test` remains available.

The API is `summit.epistasis.scale_honest.boxcox_honest_scale_test`; the array
workflow reuses shared native products, existing fixed-effect geometry and
continuous-search envelopes. No independent genotype decoder is introduced.
