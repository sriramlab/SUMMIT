"""Ascertainment-aware generalized contextual covariance for binary traits.

The separate PCGC moment contract has no quantitative variance row or fixed
effect projection. Re-export the binary API here beside the quantitative one.
"""
from summit.pcgc.gxe import (
    GxEMoments, prepare_gxe_moments, prepare_gxe_external, fit_gxe,
    evaluate_contexts, plan_gxe_reference, EXTERNAL_CONTRACT,
)
from summit.pcgc.sampling import SamplingMoments

__all__ = ["GxEMoments", "prepare_gxe_moments", "prepare_gxe_external",
           "fit_gxe", "evaluate_contexts", "plan_gxe_reference", "EXTERNAL_CONTRACT", "SamplingMoments"]
