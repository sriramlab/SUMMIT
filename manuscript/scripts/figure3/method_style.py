"""Plot method style for the SUMMIT manuscript."""
from __future__ import annotations

import common

POPS = ["EUR", "SAS", "AFR"]


SIGNED_METHODS = ["S-GW", "U-W", "LDAK"]


SIGNED_TO_COMMON = {
    "S-GW": "summit",
    "U-W": "covldsc",
    "LDAK": "sumher_ldak",
}


SIGNED_LABEL = {
    "S-GW": "SUMMIT",
    "U-W": "cov-LDSC",
    "LDAK": "SumHer-LDAK",
}


def signed_method_color(method: str):
    return common.sim.METHOD_COLOR[SIGNED_TO_COMMON[method]]


def signed_method_marker(method: str) -> str:
    return common.sim.METHOD_MARKER[SIGNED_TO_COMMON[method]]
