"""Calculate the ROC and precision–recall curves for Figure S14."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import coding_curves_balanced as source


def main():
    assert source.WINDOWS_KEEP_BEST == {20000}
    for pop in ["EUR", "SAS", "AFR"]:
        for folder, expected in [("coding", 2400), ("coding_null", 900)]:
            path = Path("data/sim_h2") / folder / pop / "estimates.csv"
            frame = pd.read_csv(path, usecols=["method", "window"])
            for method in ["ldsc", "sumher", "sumher_ldak"]:
                assert (
                    frame.method.eq(method) & frame.window.eq(20000)
                ).sum() == expected, (pop, folder, method)
    calculate = source.build_balanced_curves

    def save_curves(*args, **kwargs):
        results = calculate(*args, **kwargs)
        expected = {
            (pop, method)
            for pop in ["EUR", "SAS", "AFR"]
            for method in source.MAIN_TEXT_KEEP
        }
        assert all(set(group) == expected for group in results)
        records = [
            [
                dict(pop=key[0], method=key[1], values=value)
                for key, value in group.items()
            ]
            for group in results
        ]

        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value))

        Path("data/sim_h2/coding_curves.json").write_text(
            json.dumps(records, default=serializable) + "\n"
        )
        # The figure renderer reads these curve summaries.
        raise SystemExit(0)

    source.build_balanced_curves = save_curves
    sys.argv = [__file__, "--main-text"]
    source.main()


if __name__ == "__main__":
    main()
