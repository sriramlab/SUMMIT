# Continuous scale inference with genomic covariance

This backend tests the original, genotype-defined finite mean coefficients.
It does **not** replace or relabel the conditional estimator for outcome-trained
regional scores. The latter's pilot-independence argument cannot be carried
over to genetically correlated cohorts without additional conditioning.

## Model and pointwise inference

Freeze genotypes, a row split A/B, nuisance matrices C, interaction features F,
kernel scaling, contexts, variance surfaces, and the tested coefficient groups.
For a possible null power lambda, assume

    g_lambda(Y) = C alpha_lambda + F beta_lambda + e_lambda,
    Cov(e_lambda | genotypes, covariates) = sum_k theta_k(lambda) K_k,

where theta is nonnegative. The model is required at the true null power;
transformations at other powers need not obey a zero-interaction null.
Gaussian errors imply the known-covariance result below; otherwise a suitable
central limit theorem for each fitted coefficient is required. A large nominal
sample does not establish that theorem for concentrated contrasts.

On B set R = P_C F, H = R'R, W = R H^{-1}. All supplied F columns stay in this
joint mean fit even if only one group is tested. The estimator and covariance are

    beta_hat_B(lambda) = W' g_lambda(Y_B),
    V_B(lambda) = sum_k theta_k(lambda) G_k,  G_k = W' K_BB,k W.

For an identifiable group J, beta_J' V_JJ^{-1} beta_J has a chi-square reference
with |J| degrees of freedom when the covariance is known and beta_J=0. Other
groups may have nonzero means. No training outcome enters W. Consequently this
is a **marginal** coefficient test and genetic covariance across A and B does
not invalidate its known-covariance law.

Estimate theta on A after projecting out **both C_A and all F_A**. The shared
nonnegative HE estimator uses moments m_k = y_A' P_A K_AA,k P_A y_A and a
genotype-only estimate of T_kl = tr(P_A K_k P_A K_l). Write D_kk=sqrt(T_kk),
h=D^{-1}TD^{-1}, b=D^{-1}m and eta=D theta. Solve

    eta_hat = argmin_{eta >= 0} (eta' h eta / 2 - b' eta).

The implementation refits these moments and this constrained solution at each
power. It does not reuse raw-phenotype variance components after transformation.
A common outcome unit/reference is used on A and B.

Plug-in inference is asymptotic, not an exact finite-sample result. One sufficient
condition for a fixed-dimensional group is

    || V_JJ^{-1/2} (Vhat_JJ - V_JJ) V_JJ^{-1/2} || -> 0 in probability.

Together with asymptotic normality of the standardized coefficient this permits
Slutsky's theorem **without independence of theta_hat_A and beta_hat_B**. This
condition is stronger and more relevant than merely requiring each variance
component's absolute error to approach zero. It also requires adequate accuracy
of the randomized trace geometry. Sparse support, weakly identified covariance
components, model misspecification and finite pilot sizes remain empirical
qualification questions. A zero NNLS estimate is not evidence that a biological
component is absent.

## Continuous union null

The null is the existence of one common power in the declared compact interval
that sets all coefficients in J to zero. If p(lambda) is valid at that null
power, sup_lambda p(lambda) is conservative for the union null. Validity does
not require every transformation to satisfy the covariance model. A sampled
maximum is a lower bound on that supremum and cannot certify rejection.

The adaptive search reports a lower bound from evaluated powers and an upper
bound covering all unsampled intervals. A nonrejection witness can stop the
search; an unresolved upper bound is never converted into a rejection.

For an interval centered at c with radius r, use analytic Box-Cox derivatives
through order d and the integral remainder

    |R_i| <= r^(d+1) |t_i|^(d+2)
              exp(max(0, c t_i + r |t_i|)) / ((d+1)! (d+2)),

where t_i = log(Y_i) - the common reference. For training component k, the
triangle inequality in its PSD seminorm and ||K_k|| <= tr(K_k) give

    delta_k <= sum_{j=1}^d r^j/j! ||K_k^(1/2) P_A y_A^(j)(c)||
                 + sqrt(tr(K_k)) ||R_A||,
    |m_k(lambda)-m_k(c)| <= 2 sqrt(m_k(c)) delta_k + delta_k^2.

Strong convexity of the nonnegative HE objective gives

    ||eta_hat(lambda)-eta_hat(c)|| <= ||b(lambda)-b(c)|| / lambda_min(h) = E.

This follows by adding the two convex variational inequalities; it remains
valid across changes in the NNLS active set. Since each G_k is PSD,

    -E S <= Vhat(lambda)-Vhat(c) <= E S,
    S = sum_k G_k / D_kk.

An interval only receives a nontrivial p upper bound after proving
Vhat_JJ(c)-E S_JJ positive definite, with numerical slack. For any center-fixed
direction d_J, the Taylor coefficients of W' y_B bound its numerator change A,
and its variance is at most d_J' (Vhat(c)+E S) d_J. Therefore

    Wald(lambda) >= max(0, |d_J' beta_hat(c)|-A)^2
                   / [d_J' (Vhat(c)+E S) d_J].

The center-optimal direction Vhat(c)^{-1} beta_hat(c) supplies the omnibus
bound. This calculation uses float64 slack, not interval-arithmetic proof of
all floating-point operations.

An optional, genotype-coordinate sparse hybrid combines the omnibus p and a
Bonferroni-corrected maximum-coordinate p as min(1,2 min(p_full,p_sparse)) at
each **common** power. This can favor sparse effects but costs omnibus power;
it is not the outcome-trained pilot-score hybrid. Its interval bounds are
combined pointwise before taking the supremum. The default remains omnibus.

## Computation and interpretation

`prepare_polygenic_scale` constructs the frozen coefficient contrasts, contracts
all G_k in one confirmation-genotype pass, and prepares one pilot HE geometry.
`boxcox_polygenic_scale_test` reuses these objects for each trait. Pilot moments
for a batch of powers and their derivatives share a genotype pass. Heavy products
use existing native prediction/GEMM machinery and the shared genotype decoder.
No N-by-N covariance or repeated confirmation inverse-covariance solve is built.

The corresponding project driver retains the complete local additive/dominance
nuisance design and genome-wide covariance components. It reports a specified
Box-Cox-family result, neither arbitrary-monotone invariance nor biological
causality. Application to an already examined cohort remains exploratory.

Dense arithmetic checks and focused known-truth experiments are distinct from
scientific validation of the real-data covariance model. Do not label the backend
calibrated at genome-wide tails on the basis of a small nominal-5% experiment.
