"""Plot baseline common for the SUMMIT manuscript."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HUJOEL_PAIRS: List[Tuple[str, str, str]] = [
    (
        "Ancient enhancer / generic enhancer",
        "Ancient_Sequence_Age_Human_Enhancer_w_flanking",
        "Human_Enhancer_Villar_w_flanking",
    ),
    (
        "Ancient promoter / generic promoter",
        "Ancient_Sequence_Age_Human_Promoter_w_flanking",
        "Human_Promoter_Villar_w_flanking",
    ),
    (
        "LoF-intolerant promoter / generic promoter",
        "Human_Promoter_Villar_ExAC_w_flanking",
        "Human_Promoter_Villar_w_flanking",
    ),
]


CORE_REGULATORY_PAIRS: List[Tuple[str, str, str]] = [
    (
        "DHS peaks / DHS + flank",
        "DHS_peaks_Trynka",
        "DHS_Trynka_w_flanking",
    ),
    (
        "H3K4me1 peaks / H3K4me1 + flank",
        "H3K4me1_peaks_Trynka",
        "H3K4me1_Trynka_w_flanking",
    ),
    (
        "H3K4me3 peaks / H3K4me3 + flank",
        "H3K4me3_peaks_Trynka",
        "H3K4me3_Trynka_w_flanking",
    ),
    (
        "H3K9ac peaks / H3K9ac + flank",
        "H3K9ac_peaks_Trynka",
        "H3K9ac_Trynka_w_flanking",
    ),
]
