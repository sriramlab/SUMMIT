import pytest


def test_scheduler_placement_is_exact_and_sorted():
    from scripts.epistasis.hoffman_launch import environment
    result=environment(2,[7,2],{2:(0,1),7:(1,1)},dict(GOMP_CPU_AFFINITY='0-99'))
    assert result['OMP_PLACES']=='{2},{7}'
    assert result['OMP_WAIT_POLICY']=='PASSIVE' and 'GOMP_CPU_AFFINITY' not in result
    with pytest.raises(RuntimeError,match='exactly'):
        environment(2,[2,3,7],{}, {})
    with pytest.raises(RuntimeError,match='physical'):
        environment(2,[2,7],{2:(0,1),7:(0,1)}, {})


def test_scheduler_smt_subset_cannot_expand_or_take_extra_physical_cores():
    from scripts.epistasis.hoffman_launch import physical_cpu_subset,environment
    topology={2:(0,1),6:(0,1),3:(1,1),7:(1,1),8:(1,2)}
    cpus=physical_cpu_subset(2,[7,6,3,2],topology)
    assert cpus==[2,3]
    assert environment(2,cpus,topology,{})['OMP_PLACES']=='{2},{3}'
    assert physical_cpu_subset(2,[7,2],topology)==[2,7]
    for inherited in ([2,6],[2,3,8],[],[2,2],[2,99]):
        with pytest.raises(RuntimeError,match='exactly NSLOTS physical cores'):
            physical_cpu_subset(2,inherited,topology)
