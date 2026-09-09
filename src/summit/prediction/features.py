"""Portable feature recipes; reference calibration stays in context code."""
from __future__ import annotations

import numpy as np
from ._validation import closed


def fit_contexts(sources, source_specs):
    from summit.context.multienvironment import MultiEnvironmentSourceSpec, calibrate_multienvironment_basis
    specs = [MultiEnvironmentSourceSpec(**s) for s in source_specs]
    n = len(sources[specs[0].name])
    calibration = calibrate_multienvironment_basis(sources, specs, mask=np.ones(n, dtype=bool))
    spec = dict(kind="summit.prediction.context", schema_version=1, names=list(calibration.names),
                source_calibrations=[s.to_dict() for s in calibration.source_calibrations],
                transform=np.eye(len(calibration.names)).tolist(), calibration_identity=calibration.digest)
    return calibration.basis, spec, calibration.basis_metric


def evaluate_contexts(spec, sources):
    from summit.context.multienvironment import (
        MultiEnvironmentSourceCalibration, MultiEnvironmentSourceSpec, evaluate_multienvironment_sources,
    )
    closed(spec, ("kind", "schema_version", "names", "source_calibrations", "transform", "calibration_identity"), name="context recipe")
    if spec["kind"] != "summit.prediction.context" or spec["schema_version"] != 1:
        raise ValueError("unsupported context recipe")
    fitted, raw_names = [], ["intercept"]
    for item in spec["source_calibrations"]:
        closed(item, ("spec", "center", "scale", "categories", "reference_category", "category_probabilities", "emitted_names"), name="source calibration")
        closed(item["spec"], ("name", "kind", "categories", "reference_category", "include_fixed_effect"), name="context source")
        source = MultiEnvironmentSourceSpec(**item["spec"])
        if source.kind == "continuous":
            if not np.isfinite(item["center"]) or not np.isfinite(item["scale"]) or item["scale"] <= 0:
                raise ValueError("invalid frozen context affine transform")
        else:
            probabilities = np.asarray(item["category_probabilities"], dtype=float)
            if probabilities.shape != (len(item["categories"]),) or np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1):
                raise ValueError("invalid categorical reference probabilities")
            if item["reference_category"] not in item["categories"]:
                raise ValueError("invalid categorical reference level")
        fitted.append(MultiEnvironmentSourceCalibration(spec=source,
            **{k: tuple(v) if isinstance(v, list) else v for k, v in item.items() if k != "spec"}))
        raw_names.extend(item["emitted_names"])
    if not fitted:
        raise ValueError("context recipe needs at least one source")
    basis = evaluate_multienvironment_sources(fitted, raw_names, sources)
    transform = np.asarray(spec["transform"], dtype=float)
    if transform.shape != (basis.shape[1], len(spec["names"])) or not np.all(np.isfinite(transform)):
        raise ValueError("invalid context support/recoding transform")
    result = basis @ transform
    if not np.array_equal(result[:, 0], np.ones(len(result))):
        raise ValueError("context transform must preserve the declared baseline")
    return result


def evaluate_fixed(spec, sources, phi, context_names):
    """Explicit products/powers of raw covariates and frozen context columns."""
    closed(spec, ("kind", "schema_version", "terms"), name="fixed recipe")
    if spec["kind"] != "summit.prediction.fixed" or spec["schema_version"] != 1:
        raise ValueError("unsupported fixed recipe")
    columns, names = [], set()
    contexts = dict(zip(context_names, np.asarray(phi).T))
    n = len(phi)
    for term in spec["terms"]:
        closed(term, ("name", "factors"), name="fixed term")
        if not term["name"] or term["name"] in names:
            raise ValueError("fixed term names must be unique")
        names.add(term["name"])
        value = np.ones(n)
        for factor in term["factors"]:
            closed(factor, ("source", "name", "power"), name="fixed factor")
            mapping = contexts if factor["source"] == "context" else sources if factor["source"] == "covariate" else None
            if mapping is None or factor["name"] not in mapping:
                raise ValueError("missing/unknown fixed feature source")
            power = factor["power"]
            if type(power) is not int or not 1 <= power <= 8:
                raise ValueError("fixed feature powers must be integers in [1,8]")
            raw = np.asarray(mapping[factor["name"]], dtype=float)
            if raw.shape != (n,) or not np.all(np.isfinite(raw)):
                raise ValueError("invalid fixed feature column")
            value *= raw**power
        columns.append(value)
    result = np.column_stack(columns) if columns else np.empty((n, 0))
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite fixed feature transform")
    return result
