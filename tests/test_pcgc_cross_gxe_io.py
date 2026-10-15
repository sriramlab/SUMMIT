import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from summit.pcgc.cross_gxe import prepare_cross_gxe
from summit.pcgc.cross_gxe_io import write_cross_artifacts,load_cross_artifacts,output_paths
from summit.prediction.spec import VariantAxis
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from test_pcgc_cross_gxe import fixture


def test_separate_roundtrip_identity_and_content_checks(tmp_path):
    x,a,traits,ids,rows=fixture()
    m=len(a)
    axis=VariantAxis(tuple(f"s{i}" for i in range(m)),("1",)*m,tuple(range(1,m+1)),("A",)*m,("G",)*m,"GRCh37")
    original=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,probes=31,native=False)
    receipt=write_cross_artifacts(original,tmp_path/"a",variant_axis=axis,annotation_names=["a","b"])
    ref,summary=output_paths(tmp_path/"a")
    with pytest.raises(MemoryError,match="archive loading"):
        load_cross_artifacts(summary,ref,memory_bytes=1024)
    loaded,h=load_cross_artifacts(summary,ref)
    for p,q in zip(original.pairs,loaded.pairs):
        np.testing.assert_array_equal(p.ldscores,q.ldscores)
        np.testing.assert_array_equal(p.rhs_rows,q.rhs_rows)
    assert os.stat(ref).st_mode&0o777==0o600
    assert os.stat(summary).st_mode&0o777==0o600
    with pytest.raises(FileExistsError):
        write_cross_artifacts(original,tmp_path/"a",variant_axis=axis,annotation_names=["a","b"])
    other=prepare_cross_gxe(ArraySequentialGenotypeOperator(x),a,traits,ids,probes=31,seed=1,native=False)
    write_cross_artifacts(other,tmp_path/"b",variant_axis=axis,annotation_names=["a","b"])
    with pytest.raises(ValueError,match="exact"):
        load_cross_artifacts(summary,output_paths(tmp_path/"b")[0])
    with np.load(summary,allow_pickle=False) as arc:
        arrays={name:arc[name] for name in arc.files}
    arrays["scores_0"][0,0]*=-1
    np.savez(tmp_path/"tampered.npz",**arrays)
    with pytest.raises(ValueError,match="checksum"):
        load_cross_artifacts(tmp_path/"tampered.npz",ref)
    # Equal dimensions, a changed allele, and a fresh seal still cannot be paired.
    changed=VariantAxis(axis.ids,axis.chromosome,axis.position,axis.other,axis.counted,axis.genome_build)
    write_cross_artifacts(original,tmp_path/"c",variant_axis=changed,annotation_names=["a","b"])
    with pytest.raises(ValueError,match="exact"):
        load_cross_artifacts(summary,output_paths(tmp_path/"c")[0])


@pytest.mark.parametrize("kind",["quantitative","binary"])
def test_actual_bed_cli_and_genotype_free_fit(tmp_path,kind):
    from summit.entrypoint import main
    from test_pcgc_io import fixture as raw_fixture
    from test_pcgc_gxe_io import write_bed
    from prediction_helpers import prediction_threads
    raw,x,axis,risk,scale=raw_fixture(144,96)
    write_bed(tmp_path/"study.bed",raw,axis)
    rng=np.random.default_rng(344)
    people=pd.DataFrame(dict(FID=["study"]*len(raw),IID=[f"i{i}" for i in range(len(raw))],
        Y=(risk.z>0).astype(int) if kind=="binary" else rng.normal(size=len(raw)),
        E=rng.uniform(-1,1,len(raw)),RISK=risk.population_risk))
    people.to_csv(tmp_path/"a.tsv",sep="\t",index=False)
    people.Y=rng.normal(size=len(raw))
    people.iloc[::-1].to_csv(tmp_path/"b.tsv",sep="\t",index=False)
    pd.DataFrame(dict(SNP=axis.ids,A1=axis.counted,A2=axis.other,MEAN=scale.mean,
                     INV_SD=scale.inverse_scale)).to_csv(tmp_path/"scale.tsv",sep="\t",index=False)
    first=dict(name="A",kind=kind,table="a.tsv",contexts=["E"],selection="outcome_independent")
    first.update(dict(unit_liability=True,risk_column="RISK") if kind=="binary" else dict(population_sd=1.))
    config=dict(genotype="study.bed",population_scale="scale.tsv",traits=[first,
        dict(name="B",kind="quantitative",table="b.tsv",contexts=[],population_sd=1.,
             selection="outcome_independent")])
    (tmp_path/"input.json").write_text(json.dumps(config))
    assert main(["pcgc-cross","prepare",str(tmp_path/"input.json"),"--out",str(tmp_path/"prep"),
                 "--nvecs","257","--block-size","31","--probe-tile","17","--memory-gib","3",
                 "--num-threads",str(prediction_threads())])==0
    ref,summary=output_paths(tmp_path/"prep")
    moments,header=load_cross_artifacts(summary,ref)
    assert moments.diagnostics["genotype_passes"]==2
    assert [s.shape[1] for s in moments.signed_scores]==[2,1]
    # Fit is entirely independent of the source genotype descriptors.
    Path(tmp_path/"study.bed").rename(tmp_path/"unavailable.bed")
    assert main(["pcgc-cross","fit","--sumstats",str(summary),"--ldscores",str(ref),
                 "--out",str(tmp_path/"fit"),"--njack","8","--num-threads",str(prediction_threads())])==0
    result=json.loads((tmp_path/"fit.cross.json").read_text())
    assert np.array(result["pairs"]["0:1"]["omega"]).shape==(1,2,1)
    assert "joint_minimum_eigenvalue" in result["pairs"]["0:1"]
    assert result["uncertainty_status"].startswith("experimental")


    metric=dict(schema="summit.pcgc.cross_population_metrics.v1",reference_identity=header["manifest_hash"],
        target_population="synthetic population",centering="population_feature_means_removed",
        kernel_second_moments={"0:0":[[[1.,0.],[0.,1.]]],"0:1":[[[1.],[0.]]],"1:1":[[[1.]]]},
        total_variances=[1.,1.])
    metric_path=tmp_path/"population.json"
    metric_path.write_text(json.dumps(metric))
    args=["pcgc-cross","fit","--sumstats",str(summary),"--ldscores",str(ref),
          "--out",str(tmp_path/"population_fit"),"--njack","8","--num-threads",str(prediction_threads()),
          "--population-metrics",str(metric_path)]
    assert main(args)==0
    result=json.loads((tmp_path/"population_fit.cross.json").read_text())
    assert result["population"]["centering"]=="population_feature_means_removed"
    metric["centering"]="uncentered"
    metric_path.write_text(json.dumps(metric))
    args[args.index("--out")+1]=str(tmp_path/"invalid_population_fit")
    with pytest.raises(ValueError,match="centered genotype-by-context"):
        main(args)

    metric["centering"]="population_feature_means_removed"
    metric["reference_identity"]="wrong reference"
    metric_path.write_text(json.dumps(metric))
    with pytest.raises(ValueError,match="declared cross-trait reference"):
        main(args)
    metric["reference_identity"]=header["manifest_hash"]
    metric["kernel_second_moments"]["1:0"]=metric["kernel_second_moments"].pop("0:1")
    metric_path.write_text(json.dumps(metric))
    with pytest.raises(ValueError,match="ordered pair axis"):
        main(args)


def test_cli_runtime_sets_no_host_cpu_policy(monkeypatch):
    # Placement is the caller's contract (outer taskset or scheduler binding).
    # The library keeps only the process-local THP guard, as the epistasis CLI.
    from types import SimpleNamespace
    from summit.pcgc import cross_gxe_cli
    environment=dict(os.environ)
    environment.pop("OMP_THREAD_LIMIT",None)
    monkeypatch.setattr(os,"environ",environment)
    monkeypatch.setattr(os,"uname",lambda: SimpleNamespace(nodename="tabla"))
    monkeypatch.setattr(os,"sched_getaffinity",lambda pid: set(range(40,48)),raising=False)
    cross_gxe_cli._runtime(8)
    assert os.environ["OMP_THREAD_LIMIT"]=="8"
    with pytest.raises(ValueError,match="positive"):
        cross_gxe_cli._runtime(0)


def test_manifest_case_status_risks_and_fit_calibration_flags(tmp_path):
    from summit.entrypoint import main
    from test_pcgc_io import fixture as raw_fixture
    from test_pcgc_gxe_io import write_bed
    from prediction_helpers import prediction_threads
    raw,x,axis,risk,scale=raw_fixture(144,96)
    write_bed(tmp_path/"study.bed",raw,axis)
    rng=np.random.default_rng(345)
    people=pd.DataFrame(dict(FID=["study"]*len(raw),IID=[f"i{i}" for i in range(len(raw))],
        Y=(risk.z>0).astype(int),E=rng.uniform(-1,1,len(raw)),RISK=risk.population_risk))
    people.to_csv(tmp_path/"a.tsv",sep="\t",index=False)
    pd.DataFrame(dict(SNP=axis.ids,A1=axis.counted,A2=axis.other,MEAN=scale.mean,
                     INV_SD=scale.inverse_scale)).to_csv(tmp_path/"scale.tsv",sep="\t",index=False)
    trait=dict(name="A",kind="binary",table="a.tsv",contexts=["E"],selection="own_case_status",
               prevalence=.05,unit_liability=True,risk_column="RISK")
    config=dict(genotype="study.bed",population_scale="scale.tsv",traits=[trait])
    (tmp_path/"input.json").write_text(json.dumps(config))
    command=["pcgc-cross","prepare",str(tmp_path/"input.json"),"--nvecs","32","--block-size","31",
             "--memory-gib","3","--num-threads",str(prediction_threads())]
    with pytest.raises(ValueError,match="fitted probit risks"):
        main(command+["--out",str(tmp_path/"refused")])
    trait["allow_uncalibrated_supplied_risks"]=True
    (tmp_path/"input.json").write_text(json.dumps(config))
    assert main(command+["--out",str(tmp_path/"kept")])==0
    del trait["allow_uncalibrated_supplied_risks"],trait["risk_column"]
    (tmp_path/"input.json").write_text(json.dumps(config))
    assert main(command+["--out",str(tmp_path/"fitted")])==0
    flags={}
    for name in ("kept","fitted"):
        ref,summary=output_paths(tmp_path/name)
        assert main(["pcgc-cross","fit","--sumstats",str(summary),"--ldscores",str(ref),
                     "--out",str(tmp_path/(name+"_fit")),"--njack","8"])==0
        result=json.loads((tmp_path/(name+"_fit.cross.json")).read_text())
        flags[name]=result["calibration"]["pairs"]["0:0"]
        assert [i["calibration"]["status"] for i in result["component_intervals"]]==result["component_calibration"]
    assert "supplied_risks_under_case_status_sampling" in flags["kept"]["reasons"]
    assert "supplied_risks_under_case_status_sampling" not in flags["fitted"]["reasons"]
    # Prevalence 0.05 is rare: the heteroscedasticity flag remains for fitted risks.
    assert flags["fitted"]["status"]=="uncalibrated" and "rare_outcome" in flags["fitted"]["heteroscedasticity_triggers"]
