"""Optional pilot mean calibration; never changes discovery posterior weights."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg

from ._validation import array, digest, identifier, metadata, positive_int
from .spec import sample_ids


@dataclass(frozen=True)
class CalibrationCandidate:
    id: str
    features: np.ndarray
    names: tuple[str, ...]
    penalized: tuple[bool, ...]
    ridge: float
    provenance: dict

    def __post_init__(self):
        identifier(self.id)
        x = array(self.features, name="pilot features", ndim=2)
        names, penalized = tuple(self.names), tuple(self.penalized)
        if len(names) != x.shape[1] or len(set(names)) != len(names) or any(not isinstance(v, str) or not v for v in names):
            raise ValueError("calibration feature names are invalid")
        if len(penalized) != len(names) or any(type(v) is not bool for v in penalized):
            raise ValueError("calibration needs an explicit penalty mask")
        if not np.isfinite(self.ridge) or self.ridge < 0 or not self.provenance:
            raise ValueError("invalid ridge or missing frozen-model provenance")
        object.__setattr__(self, "features", x)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "penalized", penalized)
        object.__setattr__(self, "provenance", metadata(self.provenance))


@dataclass(frozen=True)
class MeanCalibration:
    candidate_id: str
    names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    kept: np.ndarray
    coefficients: np.ndarray
    ridge: float
    provenance: dict

    def __post_init__(self):
        identifier(self.candidate_id)
        names = tuple(self.names)
        if len(names) != len(set(names)) or any(not isinstance(v, str) or not v for v in names):
            raise ValueError("invalid calibration feature names")
        center, scale = array(self.center, name="center", ndim=1), array(self.scale, name="scale", ndim=1)
        raw_kept = np.asarray(self.kept)
        if raw_kept.ndim != 1 or raw_kept.dtype.kind not in "iu" or np.any(raw_kept < 0) or np.any(raw_kept >= len(names)):
            raise ValueError("invalid retained calibration features")
        kept = array(raw_kept, name="retained features", ndim=1, dtype=np.int64)
        coef = array(self.coefficients, name="calibration coefficients", ndim=1)
        if center.shape != (len(names),) or scale.shape != center.shape or np.any(scale <= 0) or coef.shape != (len(kept)+1,) or len(set(kept)) != len(kept):
            raise ValueError("calibration parameter dimensions disagree")
        if not np.isfinite(self.ridge) or self.ridge < 0 or not self.provenance:
            raise ValueError("invalid calibration ridge/provenance")
        for key, value in dict(names=names, center=center, scale=scale, kept=kept, coefficients=coef,
                               provenance=metadata(self.provenance)).items():
            object.__setattr__(self, key, value)

    def predict(self, features, *, names):
        x = np.asarray(features, dtype=float)
        if tuple(names) != tuple(self.names) or x.ndim != 2 or x.shape[1] != len(self.names) or not np.all(np.isfinite(x)):
            raise ValueError("calibration features/order do not match the frozen model")
        standardized = (x[:, self.kept]-self.center[self.kept])/self.scale[self.kept]
        result = self.coefficients[0] + standardized @ self.coefficients[1:]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite calibrated predictions")
        return result

    def to_dict(self):
        return dict(kind="summit.prediction.mean_calibration", schema_version=1,
            candidate_id=self.candidate_id, names=list(self.names), center=self.center.tolist(),
            scale=self.scale.tolist(), kept=self.kept.tolist(), coefficients=self.coefficients.tolist(),
            ridge=self.ridge, provenance=self.provenance)


def _fit(candidate, y, rows):
    x, yy = candidate.features[rows], y[rows]
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    tolerance = np.finfo(float).eps * np.maximum(np.max(np.abs(x), axis=0), 1)
    kept = np.flatnonzero(scale > tolerance)
    safe_scale = np.where(scale > tolerance, scale, 1.)
    design = np.column_stack([np.ones(len(rows)), (x[:, kept]-center[kept])/safe_scale[kept]])
    penalty = np.r_[False, np.asarray(candidate.penalized)[kept]]
    # Solve augmented least squares, avoiding squared condition numbers from
    # normal equations. Penalty convention is MSE + ridge * ||beta_pen||^2.
    if candidate.ridge:
        regularizer = np.diag(np.sqrt(len(rows)*candidate.ridge)*penalty)
        design = np.vstack([design, regularizer])
        yy = np.r_[yy, np.zeros(len(penalty))]
    coefficients = linalg.lstsq(design, yy, cond=1e-12, lapack_driver="gelsy")[0]
    return MeanCalibration(candidate.id, candidate.names, center, safe_scale, kept, coefficients,
        candidate.ridge, candidate.provenance)


def _folds(n, splits, seed):
    positive_int(splits, "folds")
    if splits < 2 or n < 2*splits:
        raise ValueError("each fold needs at least two samples")
    labels = np.empty(n, dtype=np.int64)
    labels[np.random.default_rng(seed).permutation(n)] = np.arange(n) % splits
    return labels


def _select(y, candidates, rows, folds):
    losses = []
    for candidate in candidates:
        predictions = np.empty(len(rows))
        for fold in np.unique(folds):
            train, test = rows[folds != fold], rows[folds == fold]
            fit = _fit(candidate, y, train)
            predictions[folds == fold] = fit.predict(candidate.features[test], names=candidate.names)
        losses.append(float(np.mean((y[rows]-predictions)**2)))
    best = min(range(len(candidates)), key=lambda j: (losses[j], candidates[j].id))
    return best, losses


def select_and_calibrate(y, candidates, samples, *, role, folds=5, outer_folds=5, seed=0,
                         discovery_samples=()):
    """Nested pilot selection with fold-local preprocessing and exact tie rules.

    Each candidate represents a complete fixed mean/ridge/score family. External
    additive scores and response augmentations are supplied as ordinary named
    features with their frozen model identities in provenance. This estimates a
    point mean, not a phenotype interval or uncertainty in true genetic effects.
    """
    if role != "pilot":
        raise ValueError("selection requires an explicitly declared pilot sample role")
    samples = sample_ids(samples)
    if set(samples) & set(tuple(x) for x in discovery_samples):
        raise ValueError("pilot and discovery samples overlap")
    y = array(y, name="pilot phenotype", ndim=1)
    candidates = tuple(candidates)
    if len(y) != len(samples) or not candidates or len({c.id for c in candidates}) != len(candidates):
        raise ValueError("invalid pilot axes or candidate IDs")
    if any(len(c.features) != len(y) for c in candidates):
        raise ValueError("pilot feature rows disagree")
    outer = _folds(len(y), outer_folds, seed)
    nested = np.empty(len(y))
    outer_choices = []
    for fold in range(outer_folds):
        train, test = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        inner = _folds(len(train), folds, seed+fold+1)
        best, _ = _select(y, candidates, train, inner)
        calibration = _fit(candidates[best], y, train)
        nested[test] = calibration.predict(candidates[best].features[test], names=candidates[best].names)
        outer_choices.append(candidates[best].id)
    final_folds = _folds(len(y), folds, seed+outer_folds+1)
    best, losses = _select(y, candidates, np.arange(len(y)), final_folds)
    calibration = _fit(candidates[best], y, np.arange(len(y)))
    report = dict(kind="summit.prediction.pilot_selection", schema_version=1,
        role=role, sample_identity=digest(samples), n=len(y), seed=seed,
        folds=folds, outer_folds=outer_folds, outer_choices=outer_choices,
        candidate_mse={c.id: value for c, value in zip(candidates, losses)},
        candidate_provenance={c.id: c.provenance for c in candidates},
        selected=candidates[best].id, nested_mse=float(np.mean((y-nested)**2)),
        tie_rule="minimum_MSE_then_lexicographic_ID", penalty="MSE_plus_ridge_beta_squared")
    return calibration, report, nested


def load_mean_calibration(path):
    from .artifacts import read_json
    from ._validation import closed, indices
    spec = read_json(path)
    closed(spec, ("kind", "schema_version", "candidate_id", "names", "center", "scale", "kept", "coefficients", "ridge", "provenance"), name="mean calibration")
    if spec["kind"] != "summit.prediction.mean_calibration" or spec["schema_version"] != 1:
        raise ValueError("unsupported calibration artifact")
    names = tuple(spec["names"])
    center, scale = array(spec["center"], name="center", ndim=1), array(spec["scale"], name="scale", ndim=1)
    kept = np.asarray(spec["kept"])
    if kept.size:
        kept = indices(kept, name="retained features", size=len(names))
    else:
        kept = np.empty(0, dtype=np.int64)
    coef = array(spec["coefficients"], name="calibration coefficients", ndim=1)
    if center.shape != (len(names),) or scale.shape != center.shape or np.any(scale <= 0) or coef.shape != (len(kept)+1,):
        raise ValueError("calibration parameter dimensions disagree")
    return MeanCalibration(spec["candidate_id"], names, center, scale, kept, coef, spec["ridge"], spec["provenance"])
