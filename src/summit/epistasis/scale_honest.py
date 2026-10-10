"""Independent-pilot projection with Berger--Boos continuous scale inference.

The pilot fixes both the tested contrasts and an outer confidence set for a
null scale. Confirmation refits exact transformed outcomes, including the
entire original nuisance/feature span. No population Taylor approximation or
identified-scale chi-square subtraction enters the validity argument.
"""
from copy import copy
import hashlib

import numpy as np
from scipy.stats import chi2

from .robust import _inverse
from .scale import boxcox_derivatives, boxcox_scale_test


def _sample_keys(values, n):
    ids = np.asarray(values)
    if ids.ndim != 1 or len(ids) != n or ids.dtype.kind not in 'USiu':
        raise ValueError('aligned, nonmissing string or integer sample IDs required')
    try:
        keys = [v.decode('utf-8') if isinstance(v,bytes) else str(v) for v in ids.tolist()]
    except UnicodeDecodeError as error:
        raise ValueError('sample ID bytes must be valid UTF-8') from error
    if any(not v or '\n' in v for v in keys) or len(set(keys)) != len(keys):
        raise ValueError('unique nonempty single-line sample IDs required')
    return keys


def _rotate_geometry(geometry, index, direction):
    """Orthogonal feature reparameterization; all original effects remain fit."""
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('nonzero finite pilot direction required')
    # Right singular vectors span the Euclidean annihilator of d. The Wald
    # covariance supplies the appropriate metric; no identity-covariance
    # approximation is made.
    _, _, vh = np.linalg.svd((direction/norm)[None,:], full_matrices=True)
    basis = np.column_stack([vh[1:].T, vh[0]])
    p = geometry.r.shape[1]
    rotation = np.eye(p)
    rotation[np.ix_(index,index)] = basis
    result = copy(geometry)
    result.r = geometry.nn(geometry.r, rotation)
    result.h = rotation.T @ geometry.h @ rotation
    result.inverse, result.condition = _inverse((result.h+result.h.T)/2)
    for value in (result.r,result.h,result.inverse):
        value.setflags(write=False)
    return result, rotation[:,index[:-1]].T


def boxcox_honest_scale_test(pilot_geometry, pilot_phenotype,
                            confirmation_geometry, confirmation_phenotype, *,
                            pilot_ids, confirmation_ids, groups=None,
                            bounds=(-2.,2.), alpha=.05, gamma=None,
                            max_pilot_evaluations=257, max_evaluations=129,
                            confidence_width=None, direction_alpha=.01,
                            batch_size=8, order=3):
    """Test the same full-coefficient scale union null using an honest pilot.

    Rows must be independent across splits, and feature/nuisance coordinates
    must represent the same population projection. Features must be specified
    independently of pilot and confirmation outcomes (or learned in a third
    independent sample). IDs enforce disjointness, not unrelatedness or honest
    feature selection. These statistical requirements remain caller duties.

    ``gamma`` is fixed before confirmation analysis, defaults to alpha/10,
    and is added to the confirmation supremum p. A training-only derivative
    test selects q-1 contrasts or the full-q fallback; either choice is valid
    at every true null power conditional on the pilot. q=1 always falls back.
    """
    gamma = alpha/10 if gamma is None else gamma
    if (not np.isfinite(alpha+gamma+direction_alpha) or not 0 < gamma < alpha < 1
            or not 0 < direction_alpha < 1):
        raise ValueError('require 0 < gamma < alpha < 1 and a valid direction threshold')
    p = pilot_geometry.r.shape[1]
    if confirmation_geometry.r.shape[1] != p:
        raise ValueError('pilot and confirmation feature axes must agree')
    pid = _sample_keys(pilot_ids,len(pilot_geometry.r))
    cid = _sample_keys(confirmation_ids,len(confirmation_geometry.r))
    if set(pid).intersection(cid):
        raise ValueError('pilot and confirmation sample IDs overlap')
    py = np.asarray(pilot_phenotype,float)
    cy = np.asarray(confirmation_phenotype,float)
    if cy.shape != (len(cid),) or not np.all(np.isfinite(cy)) or np.any(cy <= 0):
        raise ValueError('positive aligned confirmation phenotype required')
    groups = {'joint':list(range(p))} if groups is None else dict(groups)
    width = ((bounds[1]-bounds[0])/max(32.,np.sqrt(len(pid)))
             if confidence_width is None else confidence_width)
    pilot = boxcox_scale_test(pilot_geometry,py,groups=groups,bounds=bounds,
        alpha=gamma,max_evaluations=max_pilot_evaluations,batch_size=batch_size,
        order=order,confidence_width=width)
    log_y = np.log(py)
    log_y -= log_y.mean()
    tests = {}
    for name,selected in groups.items():
        index = np.asarray(selected,dtype=int)
        domain = pilot['confidence_sets'][name]
        anchor = pilot['tests'][name]['sampled_maximizer']
        item = dict(original_df=len(index),pilot_anchor=anchor,pilot_confidence_set=domain,
                    gamma=gamma,threshold=alpha)
        if not domain:
            item.update(p_upper=gamma,p_sup_lower=gamma,df=len(index),
                status='rejected_by_pilot_confidence_set',projection='not_needed',
                confirmation_evaluations=0)
            tests[name] = item
            continue
        _, beta, _, meat, _ = pilot_geometry.fit(boxcox_derivatives(log_y,anchor,order=1))
        covariance = pilot_geometry.inverse @ meat[1] @ pilot_geometry.inverse.T
        d = beta[index,1]
        try:
            vi,_ = _inverse(covariance[np.ix_(index,index)])
            direction_p = float(chi2.sf(float(d @ vi @ d),len(index)))
        except (ValueError,np.linalg.LinAlgError):
            direction_p = 1.
        projected = len(index)>1 and direction_p < direction_alpha
        if projected:
            geometry, contrast = _rotate_geometry(confirmation_geometry,index,d)
            tested = index[:-1].tolist()
        else:
            geometry, tested = confirmation_geometry, index.tolist()
            contrast = np.eye(p)[index]
        result = boxcox_scale_test(geometry,cy,groups={name:tested},bounds=bounds,
            alpha=alpha-gamma,max_evaluations=max_evaluations,batch_size=batch_size,
            order=order,search_intervals=domain)
        test = result['tests'][name]
        upper, lower = min(1.,gamma+test['p_upper']),min(1.,gamma+test['p_sup_lower'])
        item.update(p_upper=upper,p_sup_lower=lower,df=len(tested),
            status=('rejected_specified_scale_family' if upper<alpha else
                    'compatible_scale_found' if lower>=alpha else test['status']),
            projection='pilot_direction_removed' if projected else 'full_block_fallback',
            pilot_direction_p=direction_p,contrast=contrast.tolist(),
            confirmation_evaluations=result['evaluations'],confirmation=result)
        tests[name] = item
        del geometry
    def digest(ids):
        return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
    return dict(method='honest_pilot_projection_Berger_Boos_boxcox_HC3_v1',
        tests=tests,pilot=pilot,bounds=list(bounds),alpha=alpha,gamma=gamma,
        direction_alpha=direction_alpha,pilot_n=len(pid),confirmation_n=len(cid),
        pilot_sample_order_sha256=digest(pid),confirmation_sample_order_sha256=digest(cid),
        inference='Independent-split HC3 pointwise inference plus continuous outer confidence-set inversion; '
                  'asymptotic IID population projection with matching feature/nuisance coordinates.',
        null='Some power in the declared family makes the original tested coefficient block zero.',
        scope='Frozen contrasts can lose alternatives. Nonrejection is not equivalence or absence of epistasis. '
              'Honest feature selection, unrelated splits, common target population and support remain required.')
