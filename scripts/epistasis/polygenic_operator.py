"""Compatibility import for the packaged conditional-polygenic operator.

The original research implementation is preserved with interrupted full-run
sources; new preparation, solving and checkpoint identities use the package.
"""
from summit.epistasis.polygenic import (
    fit_kernel_scales, PolygenicKernels, project, he_geometry,
    estimate_components, projected_solve, conditional_score,
    conditional_tangents_batch, conditional_scores_batch,
)
