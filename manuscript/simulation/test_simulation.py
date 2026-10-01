"""Small synthetic checks against outputs from the original manuscript scripts."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from bed_reader import to_bed

from _common import new_directory, read_annotations, read_maf_ld, read_samples
import simulate_h2
import simulate_rg


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "inputs"
        self.geno = self.inputs / "genotypes" / "TEST"
        self.annot = self.inputs / "annotations" / "TEST"
        self.geno.mkdir(parents=True)
        self.annot.mkdir(parents=True)
        rng = np.random.default_rng(742)
        x = rng.binomial(2, np.linspace(0.08, 0.47, 48), size=(128, 48)).astype(float)
        x[rng.random(x.shape) < 0.025] = np.nan
        x[:, 3] = 0
        self.bed = self.geno / "genotypes.bed"
        to_bed(
            str(self.bed),
            x,
            properties={
                "fid": [f"{i:03d}" for i in range(128)],
                "iid": [f"{i:03d}" for i in range(128)],
            },
        )
        pd.DataFrame(
            np.eye(2, dtype=int)[np.arange(48) % 2], columns=["target", "rest"]
        ).to_csv(self.annot / "bins.tsv", sep="\t", index=False)
        pd.DataFrame(
            {"maf": np.linspace(0.08, 0.47, 48), "ldscore": np.linspace(1, 3, 48)}
        ).to_csv(self.annot / "features.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "second": (np.arange(48) % 3 == 0).astype(int),
                "CHR": 1,
                "first": (np.arange(48) % 2 == 0).astype(int),
                "base": 1,
            }
        ).to_csv(self.annot / "overlap.tsv.gz", sep="\t", index=False)

    def check_phenotype(self, path, expected):
        table = pd.read_csv(path, sep=r"\s+", dtype={"FID": str, "IID": str})
        self.assertEqual(table.shape, (128, 3))
        self.assertEqual(table.IID.iloc[0], "000")
        self.assertTrue(np.isfinite(table.pheno).all())
        np.testing.assert_allclose(table.pheno.iloc[:8], expected, rtol=0, atol=1.1e-6)

    def test_h2_original_outputs(self):
        # First eight phenotypes from the original scripts, seed 42, replicate 0.
        cases = [
            (
                "continuous",
                ["--p-causal", ".1"],
                [
                    -0.862362,
                    -0.374074,
                    -0.578252,
                    1.068171,
                    1.559950,
                    -1.682409,
                    -0.158497,
                    -1.057236,
                ],
            ),
            ("binary", ["--prevalence", ".3"], [2, 1, 1, 2, 2, 1, 1, 1]),
            (
                "enrichment",
                [
                    "--scenario",
                    "strong",
                    "--total-h2",
                    ".3",
                    "--architecture",
                    "ldak",
                    "--p-causal",
                    ".1",
                ],
                [
                    -0.727500,
                    -0.746308,
                    -0.590488,
                    1.055935,
                    1.694813,
                    -1.623828,
                    0.194282,
                    -0.922374,
                ],
            ),
            (
                "maf-ld",
                ["--architecture", "gcta", "--p-causal", ".1"],
                [
                    -1.715979,
                    -0.135636,
                    -1.175401,
                    -0.564516,
                    -1.745949,
                    -0.448747,
                    2.532928,
                    -1.534079,
                ],
            ),
            (
                "maf-ld",
                ["--architecture", "ldak", "--p-causal", ".1"],
                [
                    -1.603737,
                    -1.331480,
                    -0.418035,
                    -0.099721,
                    -1.253713,
                    0.885827,
                    1.052238,
                    -0.979317,
                ],
            ),
        ]
        for index, (model, options, expected) in enumerate(cases):
            with self.subTest(model=model, options=options):
                out = self.root / f"h2-{index}"
                args = [
                    "--model",
                    model,
                    "--bed",
                    str(self.bed),
                    "--annot",
                    str(self.annot / "bins.tsv"),
                    "--mafld",
                    str(self.annot / "features.tsv"),
                    "--num-reps",
                    "3",
                    "--seed",
                    "42",
                    "--chunk-size",
                    "17",
                    "--rep-batch",
                    "2",
                    "--out-dir",
                    str(out),
                    *options,
                ]
                if model != "enrichment":
                    args += ["--sigma", ".1,.2"]
                with contextlib.redirect_stdout(io.StringIO()):
                    simulate_h2.main(args)
                self.check_phenotype(out / "sim_0.phen", expected)
                oracle = pd.read_csv(out / "sim.oracle_h2.tsv", sep="\t")
                np.testing.assert_allclose(oracle.total_h2, 0.3, atol=1e-7)
                with self.assertRaises(ValueError):
                    simulate_h2.main(args)

    def test_rg_original_outputs(self):
        cases = [
            (
                "gcta",
                [
                    0.951367,
                    0.103192,
                    -0.594011,
                    -0.650533,
                    1.030272,
                    -2.087150,
                    0.672041,
                    1.598911,
                ],
            ),
            (
                "ldak",
                [
                    0.900987,
                    0.074352,
                    -0.597547,
                    -0.690650,
                    1.027964,
                    -2.103422,
                    0.664177,
                    1.630146,
                ],
            ),
            (
                "overlap",
                [
                    0.662461,
                    -0.117076,
                    -1.231134,
                    -0.264142,
                    0.505726,
                    -1.708506,
                    0.854008,
                    0.493648,
                ],
            ),
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                row = dict(
                    pop="TEST",
                    annot="bins.tsv",
                    out_prefix="traits",
                    sig1=".1,.2",
                    sig2=".2,.1",
                    rho_g=".3,-.3",
                    gamma_e=0,
                    num_reps=3,
                    seed=42,
                    max_mem=0,
                    architecture=model,
                    mafld="features.tsv",
                )
                if model == "overlap":
                    row.update(
                        annot="overlap.tsv.gz",
                        annot_cols="first,second",
                        add_base=0,
                        annotation_model="overlap_additive",
                        architecture="gcta",
                    )
                settings = self.root / f"{model}.csv"
                pd.DataFrame([row]).to_csv(settings, index=False)
                out = self.root / model
                args = [
                    "--settings",
                    str(settings),
                    "--input-dir",
                    str(self.inputs),
                    "--chunk-size",
                    "17",
                    "--rep-batch",
                    "2",
                    "--out-dir",
                    str(out),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    simulate_rg.main(args)
                self.check_phenotype(out / "TEST/traits/sim_0_1.phen", expected)
                with self.assertRaises(ValueError):
                    simulate_rg.main(args)

    def test_annotation_order_and_base(self):
        a, names = read_annotations(
            self.annot / "overlap.tsv.gz",
            48,
            columns="first,second",
            add_base=True,
            partition=False,
        )
        self.assertEqual(names, ["base", "first", "second"])
        np.testing.assert_array_equal(a[:, 0], 1)
        np.testing.assert_array_equal(a[:, 1], np.arange(48) % 2 == 0)
        np.testing.assert_array_equal(a[:, 2], np.arange(48) % 3 == 0)
        with self.assertRaises(ValueError):
            read_annotations(self.annot / "overlap.tsv.gz", 48)

    def test_feature_columns_and_sample_ids(self):
        features = np.column_stack(
            [np.full(48, 0.16), np.full(48, 2), np.full(48, 0.2)]
        )
        path = self.root / "features.tsv"
        np.savetxt(path, features)
        observed, _ = read_maf_ld(path, 48)
        np.testing.assert_array_equal(observed, features)
        self.assertEqual(read_samples(self.bed).IID.iloc[0], "000")
        with self.assertRaises(ValueError):
            new_directory(self.inputs)


if __name__ == "__main__":
    unittest.main()
