"""Genotype-level contextual posterior fitting, distinct from LD estimation.

Imports stay light so CLI thread placement can precede numerical imports.
"""

from importlib import import_module

_EXPORTS = {
    "fit_prediction": "api", "plan_prediction": "batch",
    "load_prediction_models": "artifacts", "load_genotype_scale": "artifacts",
    "write_genotype_scale": "artifacts", "PredictionModel": "artifacts",
    "score_prediction": "score", "ScoreInput": "score",
    "FileGenotypeSource": "genotype", "ArrayGenotypeSource": "genotype", "estimate_scale": "genotype",
    "ShardedGenotypeSource": "genotype",
    "TraitTraining": "spec", "CandidatePrior": "spec", "GenotypeScale": "spec", "SolverSpec": "spec", "VariantAxis": "spec",
    "ResponseGeometry": "priors", "common_scale": "priors", "separate_scales": "priors",
    "select_and_calibrate": "selection", "CalibrationCandidate": "selection", "fit_calpred": "calibration",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
