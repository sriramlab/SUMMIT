import numpy as np
import pytest
from summit.pcgc.score_sets import polynomial_nonpositive_set


@pytest.mark.parametrize('coefficients,expected',[
    ([-1,0,1],[[-1,1]]),([1,0,-1],[[None,-1],[1,None]]),
    ([1,0,1],[]),([-1,0,-1],[[None,None]]),([0],[[None,None]]),
    ([1],[]),([0,0,1],[[0,0]]),([-2,1],[[None,2]]),
    ([4,0,-5,0,1],[[-2,-1],[1,2]]),([0,0,0,0,1],[[0,0]])])
def test_complete_polynomial_sets(coefficients,expected):
    actual = polynomial_nonpositive_set(coefficients)
    assert len(actual)==len(expected)
    for a,e in zip(actual,expected):
        for x,y in zip(a,e):
            if y is None: assert x is None
            else: assert x==pytest.approx(y,abs=1e-7)


def test_quartic_ratio_variance_polynomial():
    rng = np.random.default_rng(8891)
    A,B = rng.normal(size=(7,4)),rng.normal(size=(7,4))
    g0,g1 = rng.normal(size=(2,4))
    C0,C1,C2 = A.T@A,A.T@B+B.T@A,B.T@B
    v = np.array([g0@C0@g0,2*g1@C0@g0+g0@C1@g0,
        g1@C0@g1+2*g1@C1@g0+g0@C2@g0,g1@C1@g1+2*g1@C2@g0,g1@C2@g1])
    for t in (-8,-2,-.5,0,1.3,5):
        direct = np.linalg.norm((A+t*B)@(g0+t*g1))**2
        assert np.polynomial.polynomial.polyval(t,v)==pytest.approx(direct,rel=1e-12)
