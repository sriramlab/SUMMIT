from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed


def _inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(7741)
    n, m = 41, 17
    genotype = rng.binomial(
        2, rng.uniform(0.15, 0.43, size=m), size=(n, m)
    ).astype(float)
    prefix = root / "geno"
    to_bed(str(prefix) + ".bed", genotype)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env = root / "env.tsv"
    covar = root / "covar.tsv"
    phenotype = root / "phenotypes.tsv"
    ids.assign(E=rng.normal(size=n)).to_csv(env, sep="\t", index=False)
    ids.assign(C=rng.normal(size=n)).to_csv(covar, sep="\t", index=False)
    ids.assign(Y1=rng.normal(size=n), Y2=rng.normal(size=n)).to_csv(
        phenotype, sep="\t", index=False
    )
    return prefix, env, covar, phenotype


def _run(monkeypatch, cli, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["summit", *map(str, arguments), "--suppress"])
    cli.main()


def test_cli_reference_wide_score_and_fit_roundtrip(tmp_path, monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    monkeypatch.setattr(cli, "_make_low_level_env", lambda _: None)
    genotype, environment, covariates, phenotypes = _inputs(tmp_path)

    common_generation = (
        "--geno", genotype,
        "--env", environment,
        "--covar", covariates,
        "--write-gxe-jackknife",
        "--njack", "3",
        "--gxe-kernel-mode", "standardized",
        "--gxe-genotype-scale", "sample",
        "--rand-dist", "rademacher",
        "--dtype", "float64",
        "--step_size", "17",
        "--num-threads", "1",
        "--target-xz-mem", "0.01",
    )
    reference_prefix = tmp_path / "reference"
    _run(
        monkeypatch,
        cli,
        *common_generation,
        "--nvecs", "100",
        "--seed", "81",
        "--out", reference_prefix,
    )
    reference = Path(str(reference_prefix) + ".gxe.ref.json")
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    assert reference_payload["kind"] == "summit.gxe.reference"
    assert reference_payload["randomization"]["num_vectors"] == 100

    score_prefix = tmp_path / "scores"
    _run(
        monkeypatch,
        cli,
        "--gxe-score-reference", reference,
        "--geno", genotype,
        "--env", environment,
        "--covar", covariates,
        "--gxe-pheno", phenotypes,
        "--gxe-pheno-cols", "Y1,Y2",
        "--step_size", "17",
        "--num-threads", "1",
        "--out", score_prefix,
    )
    for trait in ("Y1", "Y2"):
        assert Path(f"{score_prefix}.{trait}.gxe.gwas.tsv.gz").is_file()
        assert Path(f"{score_prefix}.{trait}.gxe.gwis.tsv.gz").is_file()
        assert Path(f"{score_prefix}.{trait}.gxe.moments.json").is_file()

    fit_prefix = tmp_path / "fit-Y1"
    _run(
        monkeypatch,
        cli,
        "--gxe-fit", reference,
        "--gxe-gwas", f"{score_prefix}.Y1.gxe.gwas.tsv.gz",
        "--gwis", f"{score_prefix}.Y1.gxe.gwis.tsv.gz",
        "--gxe-moments", f"{score_prefix}.Y1.gxe.moments.json",
        "--gxe-max-condition", "1e16",
        "--allow-ill-conditioned-gxe",
        "--out", fit_prefix,
    )
    fit_payload = json.loads(Path(f"{fit_prefix}.gxe.fit.json").read_text(encoding="utf-8"))
    assert fit_payload["component_names"] == ["G:L2_0", "GxE:L2_0", "NxE", "residual"]
    assert len(fit_payload["proportions"]) == 4

    batch_manifest = tmp_path / "fit-batch.json"
    batch_manifest.write_text(
        json.dumps(
            {
                "kind": "summit.gxe.fit_batch",
                "schema_version": 1,
                "reference": reference.name,
                "traits": [
                    {
                        "name": trait,
                        "moments": f"{score_prefix.name}.{trait}.gxe.moments.json",
                        "gwas": f"{score_prefix.name}.{trait}.gxe.gwas.tsv.gz",
                        "gwis": f"{score_prefix.name}.{trait}.gxe.gwis.tsv.gz",
                        "out": f"batch-{trait}",
                    }
                    for trait in ("Y1", "Y2")
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _run(
        monkeypatch,
        cli,
        "--gxe-fit-batch", batch_manifest,
        "--gxe-max-condition", "1e16",
        "--allow-ill-conditioned-gxe",
        "--out", tmp_path / "batch-job",
    )
    for trait in ("Y1", "Y2"):
        batch_payload = json.loads(
            (tmp_path / f"batch-{trait}.gxe.fit.json").read_text(encoding="utf-8")
        )
        assert batch_payload["component_names"] == fit_payload["component_names"]
        assert len(batch_payload["proportions"]) == 4
    np.testing.assert_allclose(
        json.loads(
            (tmp_path / "batch-Y1.gxe.fit.json").read_text(encoding="utf-8")
        )["proportions"],
        fit_payload["proportions"],
        rtol=2e-12,
        atol=2e-12,
    )
    for path in tmp_path.glob("*.gxe.*"):
        assert (path.stat().st_mode & 0o777) == 0o600
