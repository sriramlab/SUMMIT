"""Plot gazal labels for the SUMMIT manuscript."""
from __future__ import annotations

CALLOUT_LABELS = {
    "Backgrd_Selection_Stat": "Background selection",
    "CpG_Content_50kb": "CpG content",
    "GERP.NS": "GERP.NS",
    "GERP.RSsup4": "GERP RS >= 4",
    "Conserved_LindbladToh_w_flanking": "Conserved L-T",
    "Coding_UCSC_w_flanking": "Coding",
    "Promoter_UCSC_w_flanking": "Promoter",
    "DHS_Trynka_w_flanking": "DHS",
}


CALLOUT_POSITIONS = {
    r"$\tau^*$": {
        "Backgrd_Selection_Stat": (-0.08, 0.56),
        "CpG_Content_50kb": (0.16, 0.37),
        "GERP.NS": (0.47, 0.50),
        "GERP.RSsup4": (0.37, -0.05),
        "Conserved_LindbladToh_w_flanking": (-0.54, 0.11),
        "Coding_UCSC_w_flanking": (-0.52, -0.35),
        "Promoter_UCSC_w_flanking": (-0.56, -0.03),
        "DHS_Trynka_w_flanking": (-0.29, -0.18),
    },
    r"$\log_2$ enrichment": {
        "GERP.RSsup4": (2.18, 1.55),
        "Conserved_LindbladToh_w_flanking": (1.05, 0.24),
        "Coding_UCSC_w_flanking": (-0.40, 0.82),
        "Promoter_UCSC_w_flanking": (-0.40, 1.02),
        "DHS_Trynka_w_flanking": (-0.39, -0.46),
    },
}
