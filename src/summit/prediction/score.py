"""Shared-source scoring with training affine scales and exact allele mapping."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from ._validation import array, digest, indices, positive_int
from .genotype import RawBlockStream, standardize, native_module


@dataclass(frozen=True)
class ScoreInput:
    rows: np.ndarray
    phi: np.ndarray
    fixed: np.ndarray
    context_spec: dict
    fixed_spec: dict

    def __post_init__(self):
        object.__setattr__(self, "rows", indices(self.rows, name="scoring rows"))
        object.__setattr__(self, "phi", array(self.phi, name="scoring contexts", ndim=2))
        object.__setattr__(self, "fixed", array(self.fixed, name="scoring fixed features", ndim=2))
        if self.phi.shape[0] != len(self.rows) or self.fixed.shape[0] != len(self.rows) or not self.phi.shape[1]:
            raise ValueError("scoring feature/row dimensions disagree")
        if not np.array_equal(self.phi[:, 0], np.ones(len(self.rows))):
            raise ValueError("scoring phi must start with baseline-one")


@dataclass
class ScoreResult:
    samples: dict
    components: dict
    genetic: dict
    prediction: dict
    model_identities: dict
    report: dict


def align_variants(model_axis, source_axis, *, missing_variants="error"):
    if model_axis.genome_build != source_axis.genome_build:
        raise ValueError("scoring genome build differs from model")
    if missing_variants not in ("error", "mean_impute"):
        raise ValueError("unknown missing model variant policy")
    lookup = {v: i for i, v in enumerate(source_axis.ids)}
    model_rows, source_rows, flips = [], [], []
    for j, variant in enumerate(model_axis.ids):
        i = lookup.get(variant)
        if i is None:
            if missing_variants == "error":
                raise ValueError("a required model variant is absent from the scoring source")
            continue
        if (model_axis.chromosome[j], model_axis.position[j]) != (source_axis.chromosome[i], source_axis.position[i]):
            raise ValueError("variant ID maps to a different genomic position")
        alleles = (model_axis.counted[j], model_axis.other[j])
        source = (source_axis.counted[i], source_axis.other[i])
        if source == alleles:
            flip = False
        elif source == alleles[::-1]:
            flip = True
        else:
            raise ValueError("scoring allele mismatch; strand inference is not supported")
        model_rows.append(j)
        source_rows.append(i)
        flips.append(flip)
    order = np.argsort(source_rows)
    return np.asarray(model_rows, dtype=np.int64)[order], np.asarray(source_rows, dtype=np.int64)[order], np.asarray(flips, dtype=bool)[order]


def score_prediction(models, source, inputs, *, block_size=512, rhs_columns=64, threads=1,
                     memory_bytes=16*2**30, missing_variants="error", backend="native"):
    models = tuple(models)
    source.check()
    if not models or len({m.key for m in models}) != len(models):
        raise ValueError("scoring requires distinct models")
    if set(inputs) != {m.trait_id for m in models}:
        raise ValueError("scoring inputs must match the requested trait IDs exactly")
    if backend not in ("native", "numpy"):
        raise ValueError("unknown scoring backend")
    for key, val in dict(block_size=block_size, rhs_columns=rhs_columns, threads=threads, memory_bytes=memory_bytes).items():
        positive_int(val, key)
    groups, maps, reports = {}, {}, {}
    for model in models:
        model.check()
        if model.trait_id not in inputs:
            raise ValueError("missing trait scoring features")
        features = inputs[model.trait_id]
        if np.max(features.rows) >= len(source.samples):
            raise ValueError("scoring sample index outside source")
        if digest(features.context_spec) != digest(model.context_spec) or digest(features.fixed_spec) != digest(model.fixed_spec):
            raise ValueError("scoring context/fixed recipe does not match model")
        if features.phi.shape[1] != model.weights.shape[1] or features.fixed.shape[1] != len(model.fixed_coefficients):
            raise ValueError("scoring feature dimensions disagree with model")
        if rhs_columns < model.weights.shape[1]:
            raise ValueError("scoring RHS tile must fit a complete model")
        mapping = align_variants(model.variants, source.variants, missing_variants=missing_variants)
        group = (model.trait_id, model.scale.identity, model.variants.identity)
        groups.setdefault(group, []).append(model)
        maps[group] = mapping
        reports["/".join(model.key)] = dict(used_variants=len(mapping[0]), total_variants=len(model.variants.ids),
            allele_swaps=int(mapping[2].sum()), missing_variants=len(model.variants.ids)-len(mapping[0]))
    rows = np.unique(np.concatenate([inputs[m.trait_id].rows for m in models]))
    variants = np.unique(np.concatenate([x[1] for x in maps.values()]))
    nmax = max(len(inputs[m.trait_id].rows) for m in models)
    outputs_bytes = sum(8*len(inputs[m.trait_id].rows)*(m.weights.shape[1]+2) for m in models)
    scratch_bytes = 8*(4*len(rows)*block_size+8*nmax*rhs_columns)+256*2**20
    model_bytes = sum(m.weights.nbytes for m in models) + 256*(len(source.samples)+len(source.variants.ids))
    if outputs_bytes+scratch_bytes+model_bytes > memory_bytes:
        raise MemoryError("scoring outputs and bounded blocks exceed budget; split the requested samples/models and account for additional passes")
    components = {m.key: np.zeros((len(inputs[m.trait_id].rows), m.weights.shape[1]), order="F") for m in models}
    native = native_module() if backend == "native" else None
    if native is not None:
        native.configure_blas_threads(threads)
    ledger = None
    if len(variants):
        stream = RawBlockStream(source, rows, variants, block_size=block_size, threads=threads)
        row_maps = {t: np.searchsorted(rows, features.rows) for t, features in inputs.items()}
        for _, v, raw in stream.blocks("scoring"):
            for group, batch in groups.items():
                model = batch[0]
                mr, sr, flip = maps[group]
                lo = int(np.searchsorted(sr, v[0], side="left"))
                hi = int(np.searchsorted(sr, v[-1], side="right"))
                if lo == hi:
                    continue
                selected = raw[np.ix_(row_maps[model.trait_id], np.searchsorted(v, sr[lo:hi]))].astype(np.float64)
                mask = selected != -127
                selected[:, flip[lo:hi]] = np.where(mask[:, flip[lo:hi]], 2-selected[:, flip[lo:hi]], -127)
                model_index = mr[lo:hi]
                g = standardize(selected, model.scale.mean[model_index], model.scale.inverse_scale[model_index])
                q = model.weights.shape[1]
                width = max(1, rhs_columns // q)
                for begin in range(0, len(batch), width):
                    tile = batch[begin:begin+width]
                    weights = np.asfortranarray(np.concatenate([m.weights[model_index] for m in tile], axis=1))
                    if native is None:
                        product = g @ weights
                    else:
                        product = np.empty((len(g), weights.shape[1]), order="F")
                        native.prediction_product(g, weights, product, False, threads)
                    for j, m in enumerate(tile):
                        components[m.key] += product[:, j*q:(j+1)*q]
        ledger = asdict(stream.ledger)
    genetic, predictions = {}, {}
    for m in models:
        m.check()
        f = inputs[m.trait_id]
        genetic[m.key] = np.einsum("nq,nq->n", components[m.key], f.phi)
        predictions[m.key] = genetic[m.key] + f.fixed @ m.fixed_coefficients
        if not np.all(np.isfinite(predictions[m.key])):
            raise FloatingPointError("nonfinite scoring output")
    source.check()
    return ScoreResult({t: tuple(source.samples[int(i)] for i in f.rows) for t, f in inputs.items()},
        components, genetic, predictions, {m.key: m.identity for m in models},
        dict(alignment=reports, ledger=ledger, output_units="model phenotype units",
             allocated_output_bytes=outputs_bytes, estimated_scratch_bytes=scratch_bytes,
             model_and_axis_bytes=model_bytes))
