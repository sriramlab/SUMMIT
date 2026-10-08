import copy
import json
import numpy as np
import pandas as pd
import pytest
from summit.pcgc.gxe import prepare_gxe_moments, EXTERNAL_CONTRACT
from summit.pcgc.gxe_io import (
    GxEArtifact, make_gxe_artifact, write_gxe_artifact, load_gxe_artifact,
    prepare_gxe_from_source, is_gxe_artifact,
)
from summit.prediction.genotype import FileGenotypeSource
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
from test_pcgc_io import fixture
from prediction_helpers import prediction_threads


def write_bed(path,raw,axis,label="study"):
    from bed_reader import to_bed
    to_bed(path,raw,count_A1=True,num_threads=1,properties={
        "fid":[label]*len(raw),"iid":[f"i{i}" for i in range(len(raw))],
        "sid":list(axis.ids),"chromosome":list(axis.chromosome),"bp_position":list(axis.position),
        "allele_1":list(axis.counted),"allele_2":list(axis.other)})


def test_artifact_roundtrip_and_tampering(tmp_path):
    _,x,axis,risk,scale=fixture(83,65)
    phi=np.column_stack((np.ones(len(x)),np.linspace(-1,1,len(x))))
    m,d=prepare_gxe_moments(Operator(x),np.ones((len(axis.ids),1)),risk,phi,
        liability_sd=1.,native=False,probes=61)
    artifact=make_gxe_artifact(m,variant_axis=axis,annotation_names=["all"],context_names=["intercept","e"],
        sample_identity="a"*64,genotype_scale_identity=scale.identity,risk=risk,contexts=phi,liability_sd=1.,diagnostics=d)
    path=write_gxe_artifact(artifact,tmp_path/"test.binary.npz")
    assert is_gxe_artifact(path)
    loaded=load_gxe_artifact(path)
    np.testing.assert_array_equal(loaded.moments.ldscores,m.ldscores)
    assert loaded.manifest==artifact.manifest
    with pytest.raises(FileExistsError): write_gxe_artifact(artifact,path)
    h=copy.deepcopy(dict(artifact.manifest));h["context_identity"]="b"*64
    with pytest.raises(ValueError,match="checksum"): GxEArtifact(m,h)
    h=copy.deepcopy(dict(artifact.manifest));h["liability_scale_contract"]="unit"
    with pytest.raises(ValueError,match="scale"): GxEArtifact(m,h)
    from summit.pcgc.artifacts import load_artifact
    with pytest.raises(ValueError,match="contract"): load_artifact(path)


@pytest.mark.parametrize("method",["pcgc","pcgc-inverse","pcgc-basis","pcgc-ld"])
@pytest.mark.parametrize("sampling",[False,True,"architecture"])
def test_actual_cli_preparation_and_fit_all_modes(tmp_path,method,sampling):
    from summit.entrypoint import main
    raw,x,axis,risk,scale=fixture(144,128)
    phi=np.column_stack((np.ones(len(raw)),np.random.default_rng(2719).uniform(-1,1,len(raw))))
    write_bed(tmp_path/"study.bed",raw,axis)
    pd.DataFrame(dict(FID=["study"]*len(raw),IID=[f"i{i}" for i in range(len(raw))],
        Y=(risk.z>0).astype(int),E=phi[:,1],RISK=risk.population_risk,D=risk.sensitivity,SD=np.full(len(raw),1.3))).iloc[::-1].to_csv(tmp_path/"people.tsv",sep="\t",index=False)
    pd.DataFrame(dict(SNP=axis.ids,A1=axis.counted,A2=axis.other,MEAN=scale.mean,INV_SD=scale.inverse_scale)).to_csv(tmp_path/"scale.tsv",sep="\t",index=False)
    extra=[]
    if method=="pcgc-basis": extra=["--binary-basis-columns","D","--binary-basis-coefficients","1"]
    if method=="pcgc-ld":
        ref_raw=np.random.default_rng(210).binomial(2,.3,(192,len(axis.ids))).astype(float)
        write_bed(tmp_path/"ref.bed",ref_raw,axis,label="ref")
        extra=["--binary-reference-geno",str(tmp_path/"ref.bed"),"--binary-ld-factorization"]
    if sampling:
        extra += ["--binary-sampling-partners","32"]
    if sampling == "architecture":
        extra += ["--binary-architecture-probes","8"]
    argv=["--binary-method",method,"--make-binary-sumstats",str(tmp_path/"people.tsv"),
          "--geno",str(tmp_path/"study.bed"),"--binary-scale",str(tmp_path/"scale.tsv"),
          "--binary-prevalence",".1","--binary-risk-column","RISK","--binary-context-columns","E",
          "--binary-liability-sd-column","SD","--nvecs","251","--seed","836","--block-size","37",
          "--memory-gib","1","--num-threads",str(prediction_threads()),"--out",str(tmp_path/"prepared"),*extra]
    assert main(argv)==0
    artifact=load_gxe_artifact(tmp_path/"prepared.binary.npz")
    assert artifact.moments.num_contexts==2
    assert tuple(artifact.manifest["context_names"])==("intercept","E")
    if method!="pcgc-ld":
        options=dict(basis=risk.sensitivity[:,None],coefficients=[1.]) if method=="pcgc-basis" else {}
        expected,_=prepare_gxe_moments(Operator(x),np.ones((len(axis.ids),1)),risk,phi,method,
            liability_sd=1.3,probes=251,seed=836,block_size=37,native=False,**options)
        np.testing.assert_allclose(artifact.moments.ldscores,expected.ldscores,atol=2e-12)
        np.testing.assert_allclose(artifact.moments.rhs_rows,expected.rhs_rows,atol=2e-10)
    assert main(["--binary-method",method,"--h2",str(tmp_path/"prepared.binary.npz"),
                 "--njack","8","--out",str(tmp_path/"fit")])==0
    result=json.loads((tmp_path/"fit.binary.json").read_text())
    assert result["kind"]=="summit.pcgc.context_fit"
    assert result["jackknife_blocks"]==8
    assert np.asarray(result["omega"]).shape==(1,2,2)
    assert np.isfinite(result["population_heritability_se"])
    if sampling:
        assert artifact.manifest['schema_version'] == 2
        assert 'reference_probe_covariance' in artifact.manifest['features']
        assert result['reference_probe_uncertainty_included']
        assert result['external_reference_uncertainty_included'] == (method == 'pcgc-ld')
        assert result['uncertainty_method'] == ('individual_and_gaussian_architecture_v1' if sampling == "architecture" else 'individual_sampling_fixed_genome_v1')
        assert 'snp_block_covariance' in result


def test_pgen_fractional_missing_and_subset_equivalence(tmp_path):
    import pgenlib
    raw,_,axis,risk,scale=fixture(100,64)
    phi=np.column_stack((np.ones(len(raw)),np.linspace(-1,1,len(raw))))
    dosage=np.random.default_rng(515).uniform(0,2,raw.T.shape).astype(np.float32)
    dosage[2,1]=-9
    prefix=tmp_path/"dosage"
    with pgenlib.PgenWriter(bytes(prefix.with_suffix(".pgen")),sample_ct=len(raw),variant_ct=len(axis.ids),
                           nonref_flags=False,dosage_present=True) as writer:
        writer.append_dosages_batch(dosage)
    prefix.with_suffix(".psam").write_text("#FID IID\n"+"".join(f"f{i} i{i}\n" for i in range(len(raw))))
    prefix.with_suffix(".pvar").write_text("#CHROM POS ID REF ALT\n"+"".join(f"1 {i+1} rs{i} A G\n" for i in range(len(axis.ids))))
    selected=np.arange(0,len(axis.ids),2)
    with FileGenotypeSource(prefix,genome_build="test") as source:
        source.prepare(np.arange(len(raw)),len(selected),prediction_threads())
        decoded=source.read(selected)
        x=np.where(decoded==-127,0,(decoded-scale.mean[selected])*scale.inverse_scale[selected])
    with FileGenotypeSource(prefix,genome_build="test") as source:
        artifact=prepare_gxe_from_source(source,scale,np.arange(len(raw)),risk,np.ones((len(selected),1)),phi,
            annotation_names=["all"],context_names=["intercept","E"],liability_sd=1.,variant_indices=selected,
            probes=83,seed=112,block_size=13,threads=prediction_threads())
    expected,_=prepare_gxe_moments(Operator(x,genotype_format="pgen"),np.ones((len(selected),1)),risk,phi,
        liability_sd=1.,probes=83,seed=112,block_size=13,native=False)
    np.testing.assert_allclose(artifact.moments.ldscores,expected.ldscores,atol=1e-11)
    np.testing.assert_allclose(artifact.moments.rhs_rows,expected.rhs_rows,atol=1e-9)
    assert tuple(artifact.manifest["variant_axis"]["ids"])==axis.subset(selected).ids


def test_cli_scale_and_external_guards_before_reads():
    from summit.cli import build_parser
    from summit.pcgc.gxe_cli import prepare
    common=["--binary-method","pcgc","--binary-context-columns","E"]
    parser=build_parser()
    with pytest.raises(ValueError,match="requires"): prepare(parser.parse_args(common))
    with pytest.raises(ValueError,match="supplied"): prepare(parser.parse_args([*common,"--binary-liability-sd-column","SD"]))
    with pytest.raises(ValueError,match="factorization"):
        prepare(parser.parse_args([*common,"--binary-unit-liability","--binary-ld-factorization"]))
