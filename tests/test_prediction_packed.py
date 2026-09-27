"""Lossless hard-call cache, including missingness and non-multiple-of-four axes."""
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from test_prediction_core import fixture
from summit.prediction.genotype import RawBlockStream
from summit.prediction.batch import plan_prediction
from summit.prediction.mixture import MixtureSpec, MixtureSolverSpec, fit_mixture_prediction


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 17, 64])
@pytest.mark.parametrize('use_native', [False, True])
def test_cache_matches_decoded_calls(n, use_native):
    from summit import gxeldcore as native
    source, _ = fixture(n=max(n,7),m=11)
    rows=np.arange(n)
    variants=np.array([0,1,3,5,6,8,10])
    stream=RawBlockStream(source,rows,variants,storage='packed',block_size=3,
                          threads=prediction_threads(),native=native if use_native else None)
    original=[raw.copy() for _,_,raw in stream.blocks('build',build_cache=True)]
    assert stream.cache.nbytes == len(variants)*((n+3)//4)
    assert not stream.cache.flags.writeable
    for expected,(_,_,actual) in zip(original,stream.blocks('read')):
        np.testing.assert_array_equal(actual,expected)
    assert stream.ledger.source_variants==len(variants)
    assert stream.ledger.cache_variants==len(variants)


def test_cache_rejects_fractional_calls_and_aliasing():
    from summit import gxeldcore as native
    source,traits=fixture(n=31,m=9)
    source.hard_calls=False
    with pytest.raises(ValueError,match='exact hard calls'):
        RawBlockStream(source,traits[0].rows,traits[0].variants,storage='packed')
    with pytest.raises(ValueError,match='exact hard calls'):
        plan_prediction(traits,source,storage='packed')
    raw=np.zeros((5,2),dtype=np.int8,order='F')
    packed=np.empty((2,2),dtype=np.uint8,order='F')
    raw[1,1]=4
    with pytest.raises(RuntimeError,match='Invalid hard call'):
        native.prediction_pack_calls(raw,packed,prediction_threads())
    shared=np.zeros(16,dtype=np.uint8)
    with pytest.raises(RuntimeError,match='alias'):
        native.prediction_pack_calls(shared.view(np.int8).reshape(8,2,order='F'),
                                     shared[:4].reshape(2,2,order='F'),prediction_threads())
    with pytest.raises(RuntimeError,match='dimensions'):
        native.prediction_unpack_calls(packed,np.empty((9,2),dtype=np.int8,order='F'),prediction_threads())


@pytest.mark.parametrize('storage',['compact','packed'])
def test_interrupted_cache_build_cannot_expose_uninitialized_calls(storage):
    source,_=fixture(n=31,m=9)
    stream=RawBlockStream(source,np.arange(31),np.arange(9),storage=storage,block_size=3)
    build=stream.blocks('build',build_cache=True)
    next(build)
    build.close()
    assert not stream.cache_ready
    with pytest.raises(RuntimeError,match='cache build is incomplete'):
        next(stream.blocks('use'))


@pytest.mark.parametrize('backend', ['numpy','native'])
def test_packed_fit_is_bitwise_identical(tmp_path, backend):
    source,traits=fixture(n=53,m=23)
    mixture={(t.id,c.id):MixtureSpec(.1,.3) for t in traits for c in t.candidates}
    outputs=[]
    plans=[]
    for storage in ('compact','packed'):
        plans.append(plan_prediction(traits,source,storage=storage))
        outputs.append(fit_mixture_prediction(traits,source,output=tmp_path/storage,
            mixtures=mixture,storage=storage,block_size=7,backend=backend,threads=prediction_threads(),
            solver=MixtureSolverSpec(rtol=1e-9,max_sweeps=200)))
    assert plans[0].allocations['genotype_cache'] > 3*plans[1].allocations['genotype_cache']
    for a,b in zip(*outputs):
        assert a.key==b.key
        np.testing.assert_array_equal(a.weights,b.weights)
        np.testing.assert_array_equal(a.fixed_coefficients,b.fixed_coefficients)
        assert a.convergence==b.convergence
