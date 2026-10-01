"""Plot cov style for the SUMMIT manuscript."""
from __future__ import annotations

from matplotlib import colormaps
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt


ANNOTATIONS_SELECTED_PLUS_QTL = [
    "Backgrd_Selection_Stat",
    "Conserved_LindbladToh_w_flanking",
    "GERP.NS",
    "Ancient_Sequence_Age_Human_Promoter_w_flanking",
    "CpG_Content_50kb",
    "GTEx_eQTL_MaxCPP",
    "BLUEPRINT_H3K27acQTL_MaxCPP",
    "BLUEPRINT_H3K4me1QTL_MaxCPP",
    "BLUEPRINT_DNA_methylation_MaxCPP",
    "Coding_UCSC_w_flanking",
    "non_synonymous",
    "Promoter_UCSC_w_flanking",
    "DHS_Trynka_w_flanking",
]


ANNOTATION_LABEL = {
    "Backgrd_Selection_Stat": "Background selection",
    "Conserved_LindbladToh_w_flanking": "Conserved (LindbladToh)",
    "GERP.NS": "GERP.NS",
    "Ancient_Sequence_Age_Human_Promoter_w_flanking": "Ancient promoter age",
    "CpG_Content_50kb": "CpG content (50 kb)",
    "GTEx_eQTL_MaxCPP": "GTEx eQTL MaxCPP",
    "BLUEPRINT_H3K27acQTL_MaxCPP": "BLUEPRINT H3K27ac QTL",
    "BLUEPRINT_H3K4me1QTL_MaxCPP": "BLUEPRINT H3K4me1 QTL",
    "BLUEPRINT_DNA_methylation_MaxCPP": "BLUEPRINT methylation QTL",
    "Coding_UCSC_w_flanking": "Coding",
    "non_synonymous": "Nonsynonymous",
    "Promoter_UCSC_w_flanking": "Promoter",
    "DHS_Trynka_w_flanking": "DHS + flank",
}


POPS = ["EUR_300k", "SAS", "AFR"]


POP_LABEL = {"EUR_300k": "EUR", "SAS": "SAS", "AFR": "AFR"}


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]


POP_COLOR = {"EUR_300k": TAB20[1], "SAS": TAB20[4], "AFR": TAB20[6]}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 11.8,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
