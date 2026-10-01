"""Restore the typography and canvas recorded in the manuscript PDFs."""
import mafld

mafld.TITLE_FONTSIZE = 16
mafld.YTICK_LABEL_FONTSIZE = 10
mafld.stacked_figsize = lambda n_cols, n_methods: (6.5 * n_cols, 9.6)
mafld.xtick_labelsize = lambda n_methods: 12
# The retained S13 uses pooled Tukey limits with 15% padding and hides fliers.
# This preserves its displayed range; the current caption needs correction.
mafld.robust_error_ylim_by_group = (
    lambda frame, value_col, **kwargs: mafld.robust_error_ylim(
        frame[value_col].to_numpy()
    )
)
mafld.main()
