from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from test_prediction_core import fixture
from summit.prediction.genotype import FileGenotypeSource, ArrayGenotypeSource, RawBlockStream, estimate_scale
from summit.prediction.spec import CandidatePrior, SolverSpec, VariantAxis
from summit.prediction.batch import plan_prediction
from summit.prediction.operator import GenotypeOperator
from summit.prediction.solver import solve
from summit.prediction.priors import ResponseGeometry
from summit.prediction._validation import indices


def test_zero_rhs_singular_prior_shared_residual_and_no_hidden_scans():
    source, traits = fixture()
    t = replace(traits[0], y=np.zeros(len(traits[0].rows)))
    assert len({id(c.residual) for c in t.candidates}) == 1
    operator = GenotypeOperator(source, [t], plan_prediction([t], source, block_size=7), backend="native")
    operator.setup()
    result = solve(operator, SolverSpec(rtol=1e-12))
    assert operator.ledger.traversals == {"setup": 1, "verification": 1}
    for key, u in result.solutions.items():
        np.testing.assert_array_equal(u, 0)
        assert result.reports[key]["iterations"] == 0
        assert result.reports[key]["converged"]


def test_rank_zero_geometry_uint_overflow_and_stale_stream():
    geometry = ResponseGeometry(np.diag([.2, 0, 0]), np.eye(2), "test")
    np.testing.assert_array_equal(geometry.prior(rank=1), geometry.omega)
    empty = ResponseGeometry([[.2]], [], "test")
    assert empty.metric.shape == (0, 0)
    with pytest.raises(ValueError, match="range"):
        indices(np.array([2**64-1], dtype=np.uint64), name="test")
    source, traits = fixture()
    stream = RawBlockStream(source, traits[0].rows, traits[0].variants)
    source.prepare(traits[1].rows, 512, 1)
    with pytest.raises(RuntimeError, match="another stream"):
        list(stream.blocks("test"))


def test_fractional_pgen_source_preserves_dosages_and_compact_storage(tmp_path):
    import pgenlib
    rng = np.random.default_rng(7)
    n, m = 31, 13
    alt = rng.uniform(0, 2, (m, n)).astype(np.float32)
    alt[2, 4] = -9
    prefix = tmp_path/"dosage"
    with pgenlib.PgenWriter(bytes(prefix.with_suffix(".pgen")), sample_ct=n, variant_ct=m,
                           nonref_flags=False, dosage_present=True) as writer:
        writer.append_dosages_batch(np.ascontiguousarray(alt))
    with prefix.with_suffix(".psam").open("x") as handle:
        handle.write("#FID\tIID\n")
        for i in range(n):
            handle.write(f"f\ti{i}\n")
    with prefix.with_suffix(".pvar").open("x") as handle:
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for j in range(m):
            handle.write(f"1\t{j+1}\trs{j}\tA\tG\n")
    with FileGenotypeSource(prefix, genome_build="GRCh37") as source:
        rows = np.arange(n, dtype=np.int64)[::2]
        source.prepare(rows, m, 1)
        decoded = source.read(np.arange(m))
        expected = 2-alt[:, rows].T.astype(float)
        expected[alt[:, rows].T == -9] = -127
        np.testing.assert_allclose(decoded, expected, atol=1/16384, rtol=0)
        assert np.any((decoded > 0) & (decoded < 2) & (decoded != np.rint(decoded)))
        stream = RawBlockStream(source, rows, np.arange(m), storage="compact", block_size=5)
        first = np.column_stack([b.copy() for _, _, b in stream.blocks("build", build_cache=True)])
        second = np.column_stack([b.copy() for _, _, b in stream.blocks("read")])
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, decoded)
        assert stream.cache.dtype == np.float64


def test_native_output_alias_and_shape_rejection():
    from summit.prediction.genotype import native_module
    native = native_module()
    x = np.eye(3, order="F")
    with pytest.raises(RuntimeError, match="alias"):
        native.prediction_product(x, x, x, False, 1)
    with pytest.raises(RuntimeError, match="dimensions"):
        native.prediction_product(x, np.ones((2, 3), order="F"), np.empty((3, 3), order="F"), False, 1)


def test_strongly_scaled_aliased_fixed_design_matches_projected_dense():
    source, traits = fixture()
    t = traits[0]
    z = np.column_stack([t.fixed[:, 0], t.fixed[:, 1]*1e5, t.fixed[:, 2]*1e-2,
                         t.fixed[:, 3], t.fixed[:, 1]*2e5])
    t = replace(t, fixed=z)
    operator = GenotypeOperator(source, [t], plan_prediction([t], source, block_size=11), backend="native")
    operator.setup()
    result = solve(operator, SolverSpec(rtol=1e-10, qr_rtol=1e-12))
    u = result.solutions[(t.id, t.candidates[0].id)]
    from test_prediction_core import dense
    expected, _, fixed, _ = dense(source, t, t.candidates[0].covariance, t.candidates[0].residual)
    np.testing.assert_allclose(u, expected, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(z @ result.fixed_coefficients[(t.id, t.candidates[0].id)], fixed, atol=2e-8)


def test_sharded_source_boundary_and_sample_axis(tmp_path):
    from bed_reader import to_bed
    from summit.prediction.genotype import ShardedGenotypeSource
    source, traits = fixture()
    paths = []
    for j, variants in enumerate(np.array_split(np.arange(len(source.variants.ids)), 2)):
        path = tmp_path/f"chr{j+1}.bed"
        values = source.values[:, variants].astype(float)
        values[values == -127] = np.nan
        to_bed(path, values, num_threads=1, properties=dict(fid=[x[0] for x in source.samples],
            iid=[x[1] for x in source.samples], sid=[source.variants.ids[v] for v in variants],
            chromosome=[str(j+1)]*len(variants), bp_position=[int(v)+1 for v in variants],
            allele_1=["A"]*len(variants), allele_2=["G"]*len(variants)))
        paths.append(path)
    with ShardedGenotypeSource(paths, genome_build="GRCh37") as shards:
        rows = traits[0].rows
        variants = np.arange(15, 29, dtype=np.int64)
        shards.prepare(rows, 14, 1)
        actual = shards.read(variants)
        np.testing.assert_array_equal(actual, source.values[np.ix_(rows, variants)])
        assert shards.sources[0].samples is shards.sources[1].samples
