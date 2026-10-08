"""Independent scoring check for a development-only nuisance diagnosis."""
from dataclasses import replace
import os
from types import SimpleNamespace
import numpy as np
import pytest


def test_partition_score_preserves_training_units_orientation_and_missingness():
    from summit.prediction.genotype import ArrayGenotypeSource,estimate_scale
    from summit.prediction.spec import VariantAxis
    from scripts.epistasis.additive_partition_diagnostic import score_partition
    rng=np.random.default_rng(981613);n,m=64,20
    raw=rng.binomial(2,.3,(n,m)).astype(np.int8);raw[3::7,2::3]=-127
    axis=VariantAxis(tuple(map(str,range(m))),('2',)*m,tuple(range(1,m+1)),('A',)*m,('C',)*m)
    samples=[(str(j),str(j)) for j in range(n)]
    from epistasis_helpers import epistasis_threads
    threads=epistasis_threads()
    source=ArrayGenotypeSource(raw,samples,axis,hard_calls=True)
    scale=estimate_scale(source,np.arange(32),np.arange(m),threads=threads)
    weights=rng.normal(size=(m,2))
    models=[SimpleNamespace(variants=axis,scale=scale,weights=weights),
        SimpleNamespace(variants=axis,scale=scale,weights=-2*weights)]
    columns=np.array([2,4,8,11,17]);rows=np.arange(32,64)
    source_raw=raw.copy();flipped=np.arange(m)%2==0
    source_raw[:,flipped]=np.where(raw[:,flipped]==-127,-127,2-raw[:,flipped])
    recoded=replace(axis,counted=tuple('C' if v else 'A' for v in flipped),
        other=tuple('A' if v else 'C' for v in flipped))
    target=ArrayGenotypeSource(source_raw,samples,recoded,hard_calls=True)
    observed=raw[np.ix_(rows,columns)]
    g=np.where(observed==-127,0.,observed-scale.mean[columns])*scale.inverse_scale[columns]
    expected=g@weights[columns,0]
    actual,ledger=score_partition(target,rows,models,{axis.ids[j] for j in columns},threads=threads)
    np.testing.assert_allclose(actual,np.column_stack([expected,-2*expected]),atol=1e-12)
    with pytest.raises(ValueError,match='ordered'):
        score_partition(target,rows[::-1],models,{'2'},threads=threads)
