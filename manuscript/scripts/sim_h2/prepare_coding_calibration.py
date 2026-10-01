"""Calculate the one-sided calibration curves for Figure S15."""
from pathlib import Path
import pandas as pd
import calibration_statistics as cal

POPS = ["EUR", "SAS", "AFR"]
METHODS = [
    "rhe",
    "sumrhe",
    "covsumrhe",
    "ldsc_20000",
    "covldsc",
    "sumher_20000",
    "sumher_ldak_20000",
]


def main():
    curves = []
    for pop in POPS:
        path = Path("data/sim_h2/coding_null") / pop / "estimates.csv"
        df = pd.read_csv(path)
        df["pop"] = pop
        for column in ["window", "num_bins", "p_causal", "true_h2", "run"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["num_bins"] = df["num_bins"].fillna(df["num_bins"].mode().iat[0]).astype(int)
        df["window"] = df["window"].fillna(-1).astype(int)
        df = cal.restrict_methods_and_windows(df, windows_keep={20000})
        df = cal.attach_method_labels(df, windows_keep={20000})
        df = df[df.method_label.isin(METHODS) & df.p_causal.isin([1.0, 0.01])].copy()
        counts = df.groupby("method_label").size()
        assert set(counts.index) == set(METHODS) and counts.eq(600).all(), (pop, counts)
        long_df = cal.to_long(
            df, "enr", "enrichment", "enrichment_se", allow_missing_se=True
        )
        long_df["is_causal"] = (long_df["bin"] != 0).astype(int)
        boot = cal.build_calibration_overall_bootstrap(
            long_df=long_df,
            alpha_vals=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
            metric="enr",
            both_sides=False,
            B=2000,
            seed=11,
            ci_level=0.95,
            cluster_cols=["arch", "scenario", "true_h2", "p_causal", "run"],
            complete_case=False,
            methods_ord_for_cc=None,
        )
        assert len(boot) == 49, (pop, len(boot))
        curves.append(boot.assign(pop=pop))
    pd.concat(curves, ignore_index=True).to_csv(
        "data/sim_h2/coding_calibration.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
