"""Production launcher qualification; portable tests have no private pool."""
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits


def test_private_numpy_pool_survives_openmp_singleton_binding():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/generalized_gxe'))
    import workflow
    evidence=getattr(workflow,'_NUMPY_POOL_WORKER_AFFINITY',None)
    if evidence is None:pytest.skip('requires the qualified private_python launcher')
    cpus=set(workflow._PRE_NUMERICAL_CPU_AFFINITY)
    assert len(evidence)>=len(cpus)-1
    with threadpool_limits(limits=len(cpus),user_api='blas'):
        x=np.ones((512,512));np.testing.assert_array_equal(x@x,np.full_like(x,512))
        for tid in evidence:assert os.sched_getaffinity(tid)==cpus
    from summit import gxeldcore
    assert gxeldcore.build_info()['openmp_placement_contract_configured']
