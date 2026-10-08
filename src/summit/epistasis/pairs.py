"""Signed supplied-pair scores under a declared Gaussian nuisance covariance.

This is a complementary score/GLS test, not FAME's variance-component Wald
test. The caller provides a covariance solve (dense for tiny validation or an
existing matrix-free SUMMIT solver). Covariance estimation uncertainty is not
silently ignored: known and fitted covariance inputs receive different labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy.stats import norm, chi2

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.spec import array_sha256, canonical_sha256, owned_readonly_array, freeze_context_mapping


@dataclass(frozen=True)
class PairScores:
    scores: np.ndarray  # pair, trait; signed
    information: np.ndarray  # pair, pair
    pair_ids: tuple[tuple[str, str], ...]
    trait_names: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        object.__setattr__(self, "pair_ids", tuple(tuple(p) for p in self.pair_ids))
        object.__setattr__(self, "trait_names", tuple(self.trait_names))
        for key in ("scores", "information"):
            value = owned_readonly_array(getattr(self, key), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError("pair summaries must be finite")
            object.__setattr__(self, key, value)
        n = len(self.pair_ids)
        if (not n or self.scores.shape != (n, len(self.trait_names))
                or self.information.shape != (n, n) or len(set(self.pair_ids)) != n):
            raise ValueError("pair-summary axes disagree")
        if (not self.trait_names or len(set(self.trait_names)) != len(self.trait_names)
                or any(not isinstance(v, str) or not v for v in self.trait_names)
                or any(len(p) != 2 or p[0] >= p[1] for p in self.pair_ids)):
            raise ValueError("invalid trait names or unordered pair identifiers")
        if not np.allclose(self.information, self.information.T, atol=1e-10):
            raise ValueError("asymmetric score information")
        eig = np.linalg.eigvalsh(self.information)
        if eig[0] < -1e-10*max(eig[-1], np.finfo(float).tiny):
            raise ValueError("invalid score covariance")
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))


def prepare_pair_scores(genotype, variant_axis, pairs, phenotypes, *, fixed_effects,
                        covariance_solve, covariance_identity, covariance_known,
                        trait_names, sample_identity):
    """GLS residual scores with all supplied pair main effects included.

    Pair features are unnormalized products on the common genotype scale.
    Intended for supplied small pair panels, never an exhaustive scan.
    ``covariance_solve`` must apply a fixed symmetric positive definite V^-1.
    """
    x = np.asarray(genotype, dtype=float)
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    fixed = np.asarray(fixed_effects, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(variant_axis.ids) or y.shape[0] != len(x)
            or fixed.ndim != 2 or fixed.shape[0] != len(x)
            or any(not np.all(np.isfinite(v)) for v in (x, y, fixed))):
        raise ValueError("pair inputs must be finite and sample/variant aligned")
    lookup = {v: i for i, v in enumerate(variant_axis.ids)}
    canonical = []
    for pair in pairs:
        if len(pair) != 2 or any(v not in lookup for v in pair) or pair[0] == pair[1]:
            raise ValueError("pairs require two distinct known variants")
        canonical.append(tuple(sorted(pair)))
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("pairs must be nonempty and unique as unordered pairs")
    if len(x)*len(canonical) > 8_000_000:
        raise MemoryError("supplied-pair feature panel exceeds bounded API capacity")
    involved = sorted({lookup[v] for p in canonical for v in p})
    fixed = np.column_stack([np.ones(len(x)), fixed, x[:, involved]])
    u = thin_rank_revealing_fixed_effect_basis(fixed)
    f = np.column_stack([x[:, lookup[a]]*x[:, lookup[b]] for a, b in canonical])
    packed = np.column_stack([u, f, y])
    solved = np.asarray(covariance_solve(packed), dtype=float)
    if solved.shape != packed.shape or not np.all(np.isfinite(solved)):
        raise ValueError("covariance solve returned invalid values")
    # Test the supplied solve on the complete subspace used by this estimator.
    gram = packed.T @ solved
    if not np.allclose(gram, gram.T, rtol=1e-8, atol=1e-8):
        raise ValueError("covariance solve is not symmetric on the score subspace")
    if np.linalg.eigvalsh((gram+gram.T)/2)[0] < -1e-8*max(np.linalg.norm(gram, 2), 1):
        raise ValueError("covariance solve is not positive on the score subspace")
    vi_u = solved[:, :u.shape[1]]
    residualized = solved[:, u.shape[1]:] - vi_u @ np.linalg.solve(u.T @ vi_u, u.T @ solved[:, u.shape[1]:])
    h = f.T @ residualized[:, :len(canonical)]
    s = f.T @ residualized[:, len(canonical):]
    if not covariance_identity or not sample_identity or type(covariance_known) is not bool:
        raise ValueError("covariance and sample provenance are required")
    return PairScores(s, (h+h.T)/2, tuple(canonical), tuple(trait_names), dict(
        covariance_identity=covariance_identity, covariance_known=covariance_known,
        sample_identity=sample_identity, variants=variant_axis.to_dict(),
        genotype_hash=array_sha256(x), fixed_hash=array_sha256(fixed),
        inference="exact_gaussian_known_covariance" if covariance_known else "plugin_gaussian_score",
        fixed_main_effect_variants=[variant_axis.ids[i] for i in involved],
    ))


def pair_tests(summary, *, trait=0, burden_weights=None,kernel_weights=None):
    """Marginal signed GLS tests, joint mixed-sign score and sparse Bonferroni.

    A three-way Bonferroni adaptive test (joint, burden, sparse) is valid under
    the declared Gaussian known-covariance model without choosing an
    uncorrected minimum. Fitted covariance retains its plug-in qualification.
    """
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    if isinstance(trait, bool) or not isinstance(trait, (int, np.integer)) or not 0 <= trait < len(summary.trait_names):
        raise ValueError("invalid trait selection")
    s, h = summary.scores[:, trait], summary.information
    diagonal = np.diag(h)
    if np.any(diagonal <= np.finfo(float).eps*max(diagonal.max(),np.finfo(float).tiny)):
        raise ValueError("a pair is annihilated by main effects or fixed effects")
    z = s/np.sqrt(diagonal)
    p = 2*norm.sf(abs(z))
    eigenvalues, vectors = np.linalg.eigh(h)
    retained = eigenvalues > max(eigenvalues[-1]*1e-10, np.finfo(float).tiny)
    joint = float(np.sum((vectors[:, retained].T @ s)**2/eigenvalues[retained]))
    joint_p = float(chi2.sf(joint, retained.sum()))
    result = dict(marginal_beta=s/diagonal, marginal_se=1/np.sqrt(diagonal),
                  signed_z=z, marginal_p_two_sided=p, joint_score=joint,
                  joint_df=int(retained.sum()), joint_p=joint_p,
                  sparse_bonferroni_p=min(1., float(len(s)*p.min())),
                  inference=summary.metadata["inference"])
    if np.all(retained):
        inverse=(vectors/eigenvalues)@vectors.T
        beta=inverse@s;se=np.sqrt(np.diag(inverse))
        result.update(joint_beta=beta,joint_beta_se=se,joint_beta_interval_95=np.column_stack([beta-1.95996398454*se,beta+1.95996398454*se]))
    if kernel_weights is not None:
        from .quadratic import quadratic_sf
        w=np.asarray(kernel_weights,dtype=float)
        if w.shape!=s.shape or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:
            raise ValueError("independent-effect kernel weights must be nonnegative")
        lam=np.linalg.eigvalsh(np.sqrt(w[:,None])*h*np.sqrt(w[None,:]))
        lam=lam[lam>max(lam[-1]*1e-10,np.finfo(float).tiny)]
        tail=quadratic_sf(float(np.sum(w*s*s)),lam)
        result.update(weighted_kernel_p=tail["p"],weighted_kernel_error=tail["absolute_error"],
                      kernel_alternative="independent signed effects with the prespecified relative variances")
    if burden_weights is not None:
        w = np.asarray(burden_weights, dtype=float)
        if w.shape != s.shape or not np.all(np.isfinite(w)) or w @ h @ w <= 0:
            raise ValueError("invalid prespecified burden weights")
        burden_z = float(w @ s/np.sqrt(w @ h @ w))
        burden_p = float(2*norm.sf(abs(burden_z)))
        result.update(burden_z=burden_z, burden_p=burden_p,
                      adaptive_bonferroni_p=min(1., 3*min(joint_p, burden_p, result["sparse_bonferroni_p"])))
    return result


def combine_independent_pair_scores(summaries):
    """Common signed pair effects across explicitly nonoverlapping cohorts.

    Independent Gaussian scores add: s=sum s_c, H=sum H_c. This differs from
    summing quadratic forms. The current interface requires identical alleles,
    pair order, units and trait labels; recoding/overlap must be resolved before
    calling. Distinct cohort hashes alone do not establish nonoverlap.
    """
    summaries = tuple(summaries)
    if len(summaries) < 2:
        raise ValueError("at least two cohorts required")
    first = summaries[0]
    for s in summaries:
        if s.metadata.get("overlap_contract") != "mutually_disjoint_cohorts":
            raise ValueError("explicit disjoint-cohort contract required")
        for key in ("effect_units", "genotype_scale_contract"):
            if not s.metadata.get(key) or s.metadata[key] != first.metadata.get(key):
                raise ValueError("cohort effect units or genotype scaling disagree")
        if (s.pair_ids != first.pair_ids or s.trait_names != first.trait_names
                or s.metadata["variants"] != first.metadata["variants"]):
            raise ValueError("cohort pair/allele/trait axes require harmonization")
    ids = [identity for s in summaries for identity in
           s.metadata.get("cohort_sample_identities", (s.metadata["sample_identity"],))]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cohort identities")
    return PairScores(sum(s.scores for s in summaries), sum(s.information for s in summaries),
                      first.pair_ids, first.trait_names, dict(first.metadata,
                      sample_identity=canonical_sha256(dict(cohort_sample_identities=ids)),
                      cohort_sample_identities=ids, estimand="common_signed_pair_effect",
                      covariance_known=all(s.metadata["covariance_known"] for s in summaries),
                      inference=("exact_gaussian_known_covariance" if all(s.metadata["covariance_known"] for s in summaries)
                                 else "plugin_gaussian_score")))


def write_pair_scores(summary, path):
    from .summary import _publish_bundle
    arrays = {"scores": summary.scores, "information": summary.information}
    manifest = dict(kind="summit.epistasis.signed_pair_scores", schema_version=1,
        pair_ids=summary.pair_ids, trait_names=summary.trait_names, metadata=summary.metadata,
        digests={key: array_sha256(value) for key, value in arrays.items()})
    return _publish_bundle(path, manifest, arrays)


def load_pair_scores(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"manifest", "scores", "information"}:
            raise ValueError("unexpected pair artifact fields")
        m = json.loads(str(archive["manifest"]))
        if m["kind"] != "summit.epistasis.signed_pair_scores" or m["schema_version"] != 1:
            raise ValueError("unsupported pair artifact")
        arrays = {key: archive[key] for key in ("scores", "information")}
        if m["digests"] != {key: array_sha256(value) for key, value in arrays.items()}:
            raise ValueError("pair artifact digest mismatch")
        return PairScores(**arrays, pair_ids=m["pair_ids"], trait_names=m["trait_names"], metadata=m["metadata"])


def prepare_bounded_pair_summary(study, pairs, phenotypes, nuisance, *, trait_names,
                                 burden_weights=None):
    """Exact bounded pair manifest preparation with explicit nuisance covariance.

    V=sum_a coefficient_a X diag(w_a) X'/W_a + residual_variance I.
    GLS jointly removes supplied fixed effects and all pair main effects.
    Coefficients are provided, never inferred to be known from a fitted file.
    The same declared V applies to each trait in this job. Dense V is restricted
    to the tiny-data path; the general Python score API accepts a solver.
    """
    from dataclasses import replace
    from scipy.linalg import cho_factor, cho_solve
    from summit.prediction._validation import closed
    closed(nuisance, ("additive_coefficients", "residual_variance", "known", "identity"), name="pair nuisance")
    if max(study.n, study.m) > 2048 or study.h != 1:
        raise ValueError("bounded pair manifests require iid residual and N,M <= 2048")
    if not np.array_equal(study.modifiers, np.ones_like(study.modifiers)):
        raise ValueError("pair nuisance study must contain only additive components")
    y = np.asarray(phenotypes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or len(y) != study.n or not np.all(np.isfinite(y)):
        raise ValueError("phenotypes must be finite and sample aligned")
    annotations = tuple(study.metadata["definitions"]["additive_annotations"])
    if set(nuisance["additive_coefficients"]) != set(annotations):
        raise ValueError("provide a coefficient for every declared additive nuisance")
    coefficients = np.asarray([nuisance["additive_coefficients"][name] for name in annotations], dtype=float)
    residual = float(nuisance["residual_variance"])
    if (not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0)
            or not np.isfinite(residual) or residual <= 0):
        raise ValueError("pair nuisance coefficients must be nonnegative with positive residual variance")
    planned = 8*(5*study.n*study.m + 5*study.n**2
                 + 6*study.n*(len(pairs)+y.shape[1]+study.fixed.shape[1])) + 64*2**20
    if planned > study.memory_bytes:
        raise MemoryError("bounded pair workspace exceeds the declared memory budget")
    study.nn.begin_execution()
    study.tn.begin_execution()
    x = np.empty((study.n, study.m), order="F")
    for start, stop, block in study._blocks("bounded_pair_genotype"):
        x[:, start:stop] = block
    covariance = residual*np.eye(study.n)
    for a, coefficient in enumerate(coefficients):
        covariance += coefficient*study._nn(x, (x*study.weights[:, a]).T)/study.masses[a]
        study.nn.finish_execution()
    factor = cho_factor(covariance, check_finite=True)
    summary = prepare_pair_scores(x, study.source.variants, pairs, y,
        fixed_effects=study.fixed, covariance_solve=lambda z: cho_solve(factor, z),
        covariance_identity=nuisance["identity"], covariance_known=nuisance["known"],
        trait_names=trait_names, sample_identity=study.metadata["sample_hash"])
    if burden_weights is not None:
        # Validate now; fitting later uses these same prespecified weights.
        pair_tests(summary, burden_weights=burden_weights)
    return replace(summary, metadata=dict(summary.metadata, study=study.metadata,
        genome_build=study.source.variants.genome_build,
        inverse_scales={v:float(study.scale.inverse_scale[study.source.variants.ids.index(v)]) for pair in summary.pair_ids for v in pair},
        nuisance=nuisance, burden_weights=burden_weights, phenotype_scale="raw",
        method="bounded_dense_nuisance_pair_scores_v1", workspace_plan_bytes=planned))
