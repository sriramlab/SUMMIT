"""Stable learner IDs for bounded batches of one prespecified experiment."""


def replicate_ids(count, start=0):
    if (type(count) is not int or type(start) is not int or count < 1
            or start < 0 or start+count > 100):
        raise ValueError('schedule 1..100 replicates with IDs in 0..99')
    return list(range(start, start+count))


def simulation_replicates(simulation):
    """Older simulations have implicit consecutive IDs starting at zero."""
    default = replicate_ids(simulation['replicates'])
    values = simulation.get('replicate_ids', default)
    if (not isinstance(values, list) or len(values) != len(default)
            or any(type(i) is not int or not 0 <= i < 100 for i in values)
            or sorted(set(values)) != values):
        raise ValueError('simulation replicate IDs must be distinct, ordered and in 0..99')
    return values


def replicate_columns(ids):
    return [f'rep{i:03d}' for i in ids]
