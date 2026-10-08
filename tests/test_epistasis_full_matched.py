"""Public full-marker driver versus independently formed complete regressions."""
from argparse import Namespace
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed


def test_full_matched_public_path(tmp_path, monkeypatch):
    from scripts.epistasis.full_matched import reference, run, load_reference

    from summit.prediction.genotype import native_module
    native = native_module()
    threads = int(native.configured_blas_threads()) or int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))
    native.configure_blas_threads(threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        monkeypatch.setenv(key, str(threads))
    rng = np.random.default_rng(77213)
    n, m = 384, 96
    g = rng.binomial(2, 0.35, (n, m)).astype(float)
    missing = rng.random(g.shape) < 0.001
    missing[:, :2] = False
    g[missing] = np.nan
    ids = list(map(str, range(n)))
    variants = ["12:66358347"] + [f"v{i}" for i in range(1, m)]
    to_bed(
        tmp_path / "g.bed",
        g,
        properties=dict(
            fid=ids,
            iid=ids,
            sid=variants,
            chromosome=["12"] * 8 + ["5"] * (m - 8),
            bp_position=[66358347 + (250000 if i == 7 else i) for i in range(m)],
            allele_1=["C"] * m,
            allele_2=["T"] * m,
        ),
    )
    cov = pd.DataFrame(
        dict(FID=ids, IID=ids, PC1=rng.normal(size=n), age=rng.normal(size=n))
    )
    cov.to_csv(tmp_path / "cov.tsv", sep="\t", index=False)
    reference(
        Namespace(
            out=tmp_path / "ref",
            genotypes=str(tmp_path / "g.bed"),
            covariates=str(tmp_path / "cov.tsv"),
            target=variants[0],
            background_chromosome="5",
            num_threads=threads,
            memory_gib=2,
            reference_samples=None,
            minimum_reference_cell=10,
            architecture_seed=7213,
            interaction_variance=0.1,
        )
    )
    run(
        Namespace(
            out=tmp_path / "fit",
            reference=tmp_path / "ref",
            training_samples=128,
            confirmation_samples=256,
            num_threads=threads,
            memory_gib=2,
            settings="dense,mixed",
            panel_seed=6327,
            seed=81763,
            replicates=1,
            structure_covariates="PC1",
        )
    )
    table = pd.read_csv(tmp_path / "fit/replicates.csv")
    assert len(table) == 8 and not table.failed.any(), table.to_dict("records")
    selected = pd.read_csv(tmp_path / "fit/confirmation.tsv", sep="\t", dtype=str)
    for setting in ("dense", "mixed"):
        root = tmp_path / "fit" / f"{setting}_000"
        ytable = pd.read_csv(
            root / "phenotype.tsv", sep="\t", dtype={"FID": str, "IID": str}
        )
        y = selected.merge(
            ytable, on=["FID", "IID"], validate="one_to_one"
        ).y.to_numpy()
        for name in ("learned", "joint"):
            ref = load_reference(root / "prepared" / f"{name}.cohort-reference.npz")
            fit = json.loads((root / f"{name}.fit.json").read_text())["fits"][0]
            design = np.column_stack([ref.fixed_effects, ref.features])
            p = ref.features.shape[1]
            inverse = np.linalg.pinv(design, rcond=1e-11)
            coef = inverse @ y
            residual = y - design @ coef
            hat = np.sum(design * inverse.T, axis=1)
            influence = inverse[-p:] * (residual / (1 - hat))
            covariance = influence @ influence.T
            np.testing.assert_allclose(fit["beta"], coef[-p:], atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(
                fit["coefficient_covariance"], covariance, atol=1e-10, rtol=1e-9
            )
    design = json.loads((tmp_path / "ref/reference.json").read_text())
    assert design["m"] == m and design["alignment"]["selected_n"] == n
    assert design["definitions"]["dense"]["biological_null"]
    assert not design["definitions"]["mixed"]["biological_null"]
    from scripts.epistasis.full_matched_population import generating_components
    with np.load(tmp_path / "ref/reference.npz") as data:
        for name in design["settings"]:
            pieces = generating_components(design, data, name, np.arange(n), .4)
            if name == "structure":
                assert set(pieces) == {"dense_additive", "local", "dominance", "structure", "interaction"}
    from scripts.epistasis.make_trans_inputs import build
    from summit.epistasis.cli import main as public_cli
    from summit.prediction.artifacts import load_prediction_models
    built = tmp_path / "constructed"
    train_spec, prepare_spec = build(Namespace(out=built,genotypes=str(tmp_path / "g.bed"),
        genome_build="GRCh37",target=variants[0],background_chromosome="5",
        training_samples=tmp_path / "fit/training.tsv",confirmation_samples=tmp_path / "fit/confirmation.tsv",
        phenotypes=tmp_path / "fit/dense_000/phenotype.tsv",phenotype_column="y",unit="fixed reference units",
        covariates=tmp_path / "ref/covariates.tsv",covariate_columns="",structure_covariates="PC1",
        local_window_bp=100000,minimum_cell_fraction=.02,num_threads=threads))
    recipe = dict(kind="summit.epistasis.trans_inputs",schema_version=1,
        genotypes=dict(geno="g.bed",genome_build="GRCh37"),target=variants[0],background_chromosome="5",
        training_samples="fit/training.tsv",confirmation_samples="fit/confirmation.tsv",
        phenotype=dict(file="fit/dense_000/phenotype.tsv",column="y",unit="fixed reference units"),
        covariates=dict(file="ref/covariates.tsv",columns=["PC1","age"],varying_effects=["PC1"]))
    (tmp_path / "input-recipe.json").write_text(json.dumps(recipe))
    public_cli(["make-inputs",str(tmp_path / "input-recipe.json"),"--out",str(tmp_path / "recipe-inputs"),
        "--num-threads",str(threads)])
    for filename in ("train.json","prepare.json","inputs.json"):
        assert json.loads((tmp_path / "recipe-inputs" / filename).read_text()) == json.loads((built / filename).read_text())
    explicit = dict(recipe)
    explicit.pop("background_chromosome")
    explicit["interaction_variants"] = "constructed/background.txt"
    (tmp_path / "explicit-recipe.json").write_text(json.dumps(explicit))
    public_cli(["make-inputs",str(tmp_path / "explicit-recipe.json"),"--out",str(tmp_path / "explicit-inputs"),
        "--num-threads",str(threads)])
    for filename in ("train.json","prepare.json"):
        assert json.loads((tmp_path / "explicit-inputs" / filename).read_text()) == json.loads((built / filename).read_text())
    explicit["interaction_variants"] = "constructed/variants.txt"
    (tmp_path / "invalid-background.json").write_text(json.dumps(explicit))
    with pytest.raises(ValueError, match="target chromosome"):
        public_cli(["make-inputs",str(tmp_path / "invalid-background.json"),"--out",str(tmp_path / "invalid-inputs"),
            "--num-threads",str(threads)])
    unusual = pd.read_csv(tmp_path / "ref/covariates.tsv",sep="\t",dtype={"FID":str,"IID":str})
    unusual["PC1"] *= 1e-8
    unusual["age"] *= 1e8
    unusual.to_csv(tmp_path / "unusual-units.tsv",sep="\t",index=False)
    recipe["covariates"]["file"] = "unusual-units.tsv"
    (tmp_path / "unusual-recipe.json").write_text(json.dumps(recipe))
    public_cli(["make-inputs",str(tmp_path / "unusual-recipe.json"),"--out",str(tmp_path / "unusual-inputs"),
        "--num-threads",str(threads)])
    normal = pd.read_csv(built / "covariates.tsv",sep="\t")[["PC1","age"]].to_numpy()
    changed = pd.read_csv(tmp_path / "unusual-inputs/covariates.tsv",sep="\t")[["PC1","age"]].to_numpy()
    np.testing.assert_allclose(changed,normal,atol=1e-7,rtol=1e-7)
    for command,manifest,outfile in [("train-direction","train.json","trained"),("prepare","prepare.json","prepared")]:
        public_cli([command,str(built / manifest),"--out",str(built / outfile),"--num-threads",str(threads)])
    learned = load_prediction_models(built / "trained/models")[0]
    baseline = load_prediction_models(tmp_path / "fit/dense_000/trained/models")[0]
    np.testing.assert_allclose(learned.weights,baseline.weights,atol=1e-9,rtol=1e-8)
    _check_training_nuisance_invariance(built, threads)
    from summit.epistasis.robust import load_robust_scores
    constructed = load_robust_scores(built / f"prepared/{prepare_spec['jobs'][0]['id']}.robust-score.npz")
    expected = load_robust_scores(tmp_path / "fit/dense_000/prepared/learned.robust-score.npz")
    for field in ("scores","information","score_covariance"):
        np.testing.assert_allclose(getattr(constructed,field),getattr(expected,field),atol=1e-9,rtol=1e-8)
    _check_multiple_targets(tmp_path, built, threads)
    from scripts.epistasis.full_matched_targets import run as retarget
    retarget(Namespace(reference=tmp_path / "ref", out=tmp_path / "ref2",
        target="v8", background_chromosome="12", direction_seed=811,
        num_threads=threads, memory_gib=2))
    retarget(Namespace(reference=tmp_path / "ref", out=tmp_path / "ref2_explicit",
        target="v8", interaction_variants=tmp_path / "ref2/interaction_variants.txt", direction_seed=811,
        num_threads=threads, memory_gib=2))
    with np.load(tmp_path / "ref2/reference.npz") as first, np.load(tmp_path / "ref2_explicit/reference.npz") as explicit:
        for key in first.files:
            np.testing.assert_array_equal(first[key], explicit[key])
    with np.load(tmp_path / "ref2/reference.npz") as second, np.load(tmp_path / "ref/reference.npz") as first:
        np.testing.assert_array_equal(second["means"][:,0],first["means"][second["rows"],0])
        np.testing.assert_allclose(second["signals"][:,5:8].var(0),.1,atol=1e-12)
    _check_public_batch(tmp_path, monkeypatch, threads)
    from scripts.epistasis.full_matched_batch import run as batch_run
    batch_run(Namespace(reference=tmp_path / "ref", out=tmp_path / "batch-driver",
        training_samples=128, confirmation_samples=256, panel_seed=6327,
        seed=47913, replicates=2, batch_size=2, settings="local_withheld,local_withheld_mixed",
        structure_covariates="PC1", phase="development", primary="learned",
        sampling_model="fixed_design_correct_mean", signal_multiplier=1.,
        num_threads=threads, memory_gib=2, interrupt_training=False))
    batch_table = pd.read_csv(tmp_path / "batch-driver/replicates.csv")
    assert len(batch_table) == 16 and not batch_table.failed.any(), batch_table.to_dict("records")
    assert "v7" not in (tmp_path / "batch-driver/training_variants.txt").read_text().splitlines()
    exclusion = json.loads((tmp_path / "batch-driver/prediction_exclusion.json").read_text())
    assert exclusion["fitted_marker_count"] == 94
    from scripts.epistasis.full_matched_assessment import assess
    assessment, _ = assess([tmp_path / "batch-driver"])
    assert len(assessment) == 8
    assert all(r["scheduled"] == 2 and r["unsupported"] == 2 for r in assessment)
    batch_run(Namespace(reference=tmp_path / "ref", out=tmp_path / "primary-only",
        training_samples=128, confirmation_samples=256, panel_seed=6327,
        seed=47913, replicates=2, batch_size=2, settings="local_withheld,local_withheld_mixed",
        structure_covariates="PC1", phase="development", primary="learned", methods="learned",
        sampling_model="fixed_design_correct_mean", signal_multiplier=1.,
        num_threads=threads, block_size=32, memory_gib=2, interrupt_training=False))
    from summit.epistasis.robust import load_robust_scores
    for setting in ("local_withheld","local_withheld_mixed"):
        for rep in range(2):
            path = f"{setting}_000/prepared/rep{rep:03d}_learned.robust-score.npz"
            together = load_robust_scores(tmp_path / "batch-driver" / path)
            alone = load_robust_scores(tmp_path / "primary-only" / path)
            for field in ("scores","information","score_covariance"):
                np.testing.assert_allclose(getattr(together,field),getattr(alone,field),atol=1e-10,rtol=1e-8)
    primary_assessment, _ = assess([tmp_path / "primary-only"])
    assert len(primary_assessment) == 2 and all(r["scheduled"] == 2 for r in primary_assessment)
    from scripts.epistasis.full_polygenic_reference import run as covariance_reference
    from scripts.epistasis.full_polygenic_fit import run as conditional_fit
    covariance_reference(Namespace(training=tmp_path / "primary-only/local_withheld_000/train.json",
        confirmation=tmp_path / "primary-only/confirmation.tsv",out=tmp_path / "covariance-reference",
        num_threads=threads,memory_gib=2,probes=128,seed=98179))
    from scripts.epistasis.full_polygenic_simulation import run as random_outcomes
    random_outcomes(Namespace(reference=tmp_path / "covariance-reference",signal_reference=tmp_path / "ref",
        out=tmp_path / "random-outcomes",replicates=2,replicate_start=8,seed=412731,signal_multiplier=np.sqrt(.2),
        num_threads=threads,memory_gib=2,block_size=19))
    simulation=json.loads((tmp_path / "random-outcomes/simulation.json").read_text())
    np.testing.assert_allclose(simulation['signal_reference_variance'],.02)
    assert simulation['replicates']==2
    assert simulation['replicate_ids']==[8,9]
    random_outcomes(Namespace(reference=tmp_path/'covariance-reference',signal_reference=tmp_path/'ref',
        out=tmp_path/'random-split',replicates=1,replicate_start=9,seed=412731,
        signal_multiplier=np.sqrt(.2),num_threads=threads,memory_gib=2,block_size=23))
    for setting in simulation['settings']:
        together=pd.read_csv(tmp_path/'random-outcomes'/setting/'phenotypes.tsv',sep='\t')
        split=pd.read_csv(tmp_path/'random-split'/setting/'phenotypes.tsv',sep='\t')
        np.testing.assert_allclose(together['rep009'],split['rep009'],rtol=2e-13,atol=2e-13)
        with np.load(tmp_path/'random-outcomes'/setting/'diagnostic_truth.npz') as full, \
             np.load(tmp_path/'random-split'/setting/'diagnostic_truth.npz') as part:
            np.testing.assert_allclose(full['mean'][:,1],part['mean'][:,0],rtol=2e-13,atol=2e-13)
    with np.load(tmp_path / "random-outcomes/random_structure/diagnostic_truth.npz") as null, \
         np.load(tmp_path / "random-outcomes/random_structure_mixed/diagnostic_truth.npz") as alt:
        assert not np.any(null['signal'])
        np.testing.assert_allclose(alt['mean']-null['mean'],np.broadcast_to(alt['signal'][:,None],alt['mean'].shape),atol=1e-14)
        assert not np.array_equal(null['mean'][:,0],null['mean'][:,1])
    public_cli(['train-direction',str(tmp_path / 'random-outcomes/random_structure/train.json'),
        '--out',str(tmp_path / 'random-outcomes/random_structure/trained'),'--num-threads',str(threads)])
    public_cli(['prepare',str(tmp_path / 'random-outcomes/random_structure/prepare.json'),
        '--out',str(tmp_path / 'random-outcomes/random_structure/prepared'),'--num-threads',str(threads)])
    random_prepared=load_robust_scores(tmp_path / 'random-outcomes/random_structure/prepared/rep008.robust-score.npz')
    assert random_prepared.scores.shape==(1,1)
    assert random_prepared.metadata['method']=='conditional_polygenic_mean_tangent_v1'
    from scripts.epistasis.public_conditional_validation import run as public_conditional_validation
    random_work=tmp_path/'random-outcomes/random_structure'
    # The generator itself must emit the selected public path; no benchmark-only
    # conversion of its preparation manifest is permitted here.
    diagnostic_args=Namespace(manifest=random_work/'prepare.json',
        simulation=tmp_path/'random-outcomes/simulation.json',setting='random_structure',
        out=tmp_path/'public-conditional',num_threads=threads,block_size=32,memory_gib=2)
    from scripts.epistasis import public_conditional_validation as validation_module
    original_diagnose=validation_module.diagnose;completed=[]
    def interrupted_diagnostics(*args):
        completed.extend(original_diagnose(*args))
        raise RuntimeError('stop after diagnostic solvers, before final report')
    with monkeypatch.context() as stop:
        stop.setattr(validation_module,'diagnose',interrupted_diagnostics)
        with pytest.raises(RuntimeError,match='stop after diagnostic solvers'):
            public_conditional_validation(diagnostic_args)
    assert (diagnostic_args.out/'failure.json').is_file()
    diagnostic_args.resume=True
    with monkeypatch.context() as recovered:
        recovered.setattr(validation_module,'cli',lambda *a,**k:pytest.fail('completed public fitting repeated'))
        diagnosed=public_conditional_validation(diagnostic_args)
        repeated=public_conditional_validation(diagnostic_args)
    assert repeated==diagnosed
    assert (diagnostic_args.out/'diagnostic_resources.1.json').is_file()
    # A completed development result cannot be relabeled through recovery.
    diagnostic_args.phase='confirmation'
    with pytest.raises(ValueError,match='validation restart definition changed'):
        public_conditional_validation(diagnostic_args)
    diagnostic_args.phase='development'
    for first,last in zip(completed,diagnosed):
        for field in ('beta','se','p','conditional_interaction_truth','known_conditional_se','conditional_mean_bias'):
            np.testing.assert_allclose(last[field],first[field],rtol=2e-10,atol=2e-12)
    phenotype_path=random_work/'phenotypes.tsv';original_bytes=phenotype_path.read_bytes()
    phenotype_path.write_bytes(original_bytes+b'\n')
    with pytest.raises(ValueError,match='diagnostic restart inputs or implementation changed'):
        public_conditional_validation(diagnostic_args)
    phenotype_path.write_bytes(original_bytes)
    assert len(diagnosed)==2
    assert [row['replicate'] for row in diagnosed]==[8,9]
    for item in diagnosed:
        assert item['biological_null'] and item['conditional_interaction_truth']==0
        assert set(item['nuisance_prediction']['predictors'])=={'additive_pgs','conditional_prediction'}
        assert item['fitted_operator_interaction_response']==0
        np.testing.assert_allclose(item['feature_response'],1,atol=2e-7)
        assert item['known_conditional_se']>0
        assert set(item['comparisons'])=={'burden','oracle'}
        assert set(item['baselines'])=={'local','finite_varying_mean'}
        assert item['baselines']['local']['fixed_rank']<item['baselines']['finite_varying_mean']['fixed_rank']
        for comparison in item['comparisons'].values():
            assert comparison['known_conditional_se']>0 and comparison['conditional_interaction_truth']==0
            assert comparison['outside_confirmation_design']
        assert 0<=item['conditional_rejection_probability']['0.005']<=item['conditional_rejection_probability']['0.05']<=1
    from scripts.epistasis.full_polygenic_stress import run as stress_outcomes
    stress_outcomes(Namespace(reference=tmp_path/'covariance-reference',signal_reference=tmp_path/'ref',
        training=random_work/'train.json',out=tmp_path/'stress-outcomes',settings='fixed_sparse,fixed_heavy,fixed_structure_mixed',
        seed=162738,replicates=2,replicate_start=4,signal_multiplier=np.sqrt(.2)))
    stress_outcomes(Namespace(reference=tmp_path/'covariance-reference',signal_reference=tmp_path/'ref',
        training=random_work/'train.json',out=tmp_path/'stress-split',settings='fixed_heavy',
        seed=162738,replicates=1,replicate_start=5,signal_multiplier=np.sqrt(.2)))
    together=pd.read_csv(tmp_path/'stress-outcomes/fixed_heavy/phenotypes.tsv',sep='\t')
    split=pd.read_csv(tmp_path/'stress-split/fixed_heavy/phenotypes.tsv',sep='\t')
    np.testing.assert_array_equal(together['rep005'],split['rep005'])
    stress=tmp_path/'stress-outcomes/fixed_heavy'
    public_cli(['train-direction',str(stress/'train.json'),'--out',str(stress/'trained'),
        '--num-threads',str(threads)])
    # The new matched phenotype must pass through the actual reuse command,
    # reusing only genotype moments and estimating its own covariance.
    reuse=json.loads((stress/'prepare.json').read_text())
    reuse['kind']='summit.epistasis.prepare_traits'
    reuse['inference']['reference']=str(tmp_path/'public-conditional/prepared/cohort-reference.npz')
    (stress/'reuse.json').write_text(json.dumps(reuse))
    invoked=[]
    def record_public_command(argv):
        invoked.append(argv[0])
        return public_cli(argv)
    with monkeypatch.context() as reuse_command:
        reuse_command.setattr(validation_module,'cli',record_public_command)
        diagnosed=public_conditional_validation(Namespace(manifest=stress/'reuse.json',
            simulation=tmp_path/'stress-outcomes/simulation.json',setting='fixed_heavy',
            out=tmp_path/'stress-conditional',num_threads=threads,block_size=32,memory_gib=2,
            phase='confirmation'))
    assert json.loads((tmp_path/'stress-conditional/design.json').read_text())['phase']=='confirmation'
    assert invoked==['prepare-traits','fit','fit']
    assert [row['replicate'] for row in diagnosed]==[4,5]
    preparation=json.loads((tmp_path/'stress-conditional/prepared/preparation.json').read_text())
    assert preparation['reference_reused'] and not preparation['training_covariance_reused']
    early=json.loads((tmp_path/'stress-conditional/primary_diagnostics.json').read_text())
    assert early['scheduled']==2 and early['preparation_identity']==preparation['identity']
    assert not (tmp_path/'public-conditional/primary_diagnostics.json').exists()
    for primary,complete in zip(early['records'],diagnosed):
        for field in ('beta','se','p','conditional_interaction_truth','known_conditional_se','conditional_mean_bias'):
            np.testing.assert_allclose(primary[field],complete[field],rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(primary['noninteraction_mean_contribution']+
            primary['realized_training_noise_contribution'],complete['conditional_mean_bias'],atol=2e-12)
    for item in diagnosed:
        assert item['biological_null'] and item['conditional_interaction_truth']==0
        assert 'fixed-operator' in item['diagnostic_target']
        assert max(item['tail_numerical_error'].values())<4e-9
    # A local cause outside the fitted window must be absent from both the
    # additive learner and covariance, while its generating effect is retained.
    full_axis=json.loads((random_work/'train.json').read_text())
    full_axis['variants']=str(tmp_path/'ref/variants.txt')
    (random_work/'full-axis-train.json').write_text(json.dumps(full_axis))
    stress_outcomes(Namespace(reference=tmp_path/'covariance-reference',signal_reference=tmp_path/'ref',
        training=random_work/'full-axis-train.json',out=tmp_path/'withheld-outcomes',
        settings='fixed_local_withheld,fixed_local_withheld_mixed',num_threads=threads,
        seed=278193,replicates=2,signal_multiplier=np.sqrt(.2)))
    hidden=tmp_path/'withheld-outcomes/fixed_local_withheld'
    hidden_train=json.loads((hidden/'train.json').read_text())
    assert 'v7' not in Path(hidden_train['variants']).read_text().splitlines()
    assert 'v7' not in hidden_train['local_variants']+hidden_train['dominance_variants']
    excluded=json.loads((tmp_path/'withheld-outcomes/prediction_exclusion.json').read_text())
    assert excluded['original_fitted_marker_count']==95 and excluded['fitted_marker_count']==94
    with np.load(hidden/'diagnostic_truth.npz') as null, \
         np.load(tmp_path/'withheld-outcomes/fixed_local_withheld_mixed/diagnostic_truth.npz') as alt, \
         np.load(tmp_path/'ref/reference.npz') as teacher:
        center=np.nanmean(g[:,7]);scale=np.sqrt(center*(1-center/2))
        local=np.nan_to_num((g[:,7]-center)/scale)
        np.testing.assert_allclose(null['mean'],teacher['means'][:,0]+.8*local,atol=2e-13)
        np.testing.assert_allclose(alt['mean']-null['mean'],alt['signal'],atol=2e-13)
        assert not np.any(null['signal'])
    public_cli(['train-direction',str(hidden/'train.json'),'--out',str(hidden/'trained'),
        '--num-threads',str(threads)])
    hidden_args=Namespace(manifest=hidden/'prepare.json',
        simulation=tmp_path/'withheld-outcomes/simulation.json',setting='fixed_local_withheld',
        out=tmp_path/'withheld-conditional',num_threads=threads,block_size=32,memory_gib=2)
    from scripts.epistasis import conditional_comparators
    def stop_before_comparators(*args,**kwargs):
        assert (hidden_args.out/'primary_diagnostics.json').is_file()
        raise RuntimeError('primary complete before comparators')
    with monkeypatch.context() as stopped:
        stopped.setattr(conditional_comparators,'compare_directions',stop_before_comparators)
        with pytest.raises(RuntimeError,match='primary complete before comparators'):
            public_conditional_validation(hidden_args)
    primary_bytes=(hidden_args.out/'primary_diagnostics.json').read_bytes()
    hidden_args.resume=True
    with monkeypatch.context() as recovered:
        recovered.setattr(validation_module,'cli',lambda *a,**k:pytest.fail('completed public fitting repeated'))
        hidden_diagnosed=public_conditional_validation(hidden_args)
    assert (hidden_args.out/'primary_diagnostics.json').read_bytes()==primary_bytes
    assert len(hidden_diagnosed)==2 and all(r['conditional_interaction_truth']==0 for r in hidden_diagnosed)
    with np.load(tmp_path/'withheld-conditional/prepared/cohort-reference.npz') as fitted_reference:
        assert 7 not in fitted_reference['variants']
    from summit.prediction.checkpoint import SolverCheckpoint
    save=SolverCheckpoint.save
    def interrupt_checkpoint(self,state):
        save(self,state)
        if self.path.name=='outcome_feature.npz' and state['iteration']>=2:
            raise RuntimeError('bounded checkpoint interruption')
    args=Namespace(reference=tmp_path / "covariance-reference",input=tmp_path / "primary-only",
        out=tmp_path / "conditional-interrupted",settings="local_withheld,local_withheld_mixed",
        num_threads=threads,memory_gib=2)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(SolverCheckpoint,'save',interrupt_checkpoint)
        with pytest.raises(RuntimeError,match='checkpoint interruption'):
            conditional_fit(args)
    failed=json.loads((args.out/'failure.json').read_text())
    assert failed['scheduled']==4 and failed['failed_or_unexecuted']==4
    args.recover_from=args.out;args.out=tmp_path / "conditional-fit"
    conditional_fit(args)
    conditional=[json.loads(line) for line in (tmp_path / "conditional-fit/replicates.jsonl").read_text().splitlines()]
    assert len(conditional)==4 and all(not r['failed'] for r in conditional)
    assert all(r['public_feature_relative_error']<1e-8 for r in conditional)
    for record in conditional:
        np.testing.assert_allclose(record['error'],record['beta']-record['interaction_response'])
        if record['biological_null']:
            assert record['interaction_response']==0.
            assert record['coverage']==(record['p']>=.05)
    assert any(abs(r['conditional_fixed_architecture_mean']-r['interaction_response'])>1e-6
        for r in conditional)
    from scripts.epistasis.full_matched_population import run as population_run
    population_run(Namespace(input=tmp_path / "batch-driver", out=tmp_path / "population",
        samples=256, draws=3, methods="joint", settings="local_withheld,local_withheld_mixed", seed=137779,
        phase="development", num_threads=threads, memory_gib=2))
    population = json.loads((tmp_path / "population/results.json").read_text())
    assert len(population) == 4
    assert all(r["numerical_failures"] == 0 and r["unsupported"] == 3 for r in population)
    batch_run(Namespace(reference=tmp_path / "ref", out=tmp_path / "smaller-confirmation",
        training_samples=128, confirmation_samples=128, panel_seed=6327,
        seed=297191, replicates=1, batch_size=1, settings="dense",
        structure_covariates="PC1", phase="development", primary="learned", methods="learned",
        sampling_model="iid_population_projection", signal_multiplier=1.,
        num_threads=threads, memory_gib=2, interrupt_training=False))
    from scripts.epistasis.full_matched_donors import run as expand_donors
    expand_donors(Namespace(input=tmp_path / "smaller-confirmation", out=tmp_path / "all-donors",
        settings="dense",num_threads=threads,memory_gib=2))
    expanded = json.loads((tmp_path / "all-donors/design.json").read_text())
    assert expanded["arguments"]["confirmation_samples"] == 256
    assert expanded["intended_confirmation_n"] == 128
    assert expanded["independent_new_learning_models"] == 0
    population_run(Namespace(input=tmp_path / "all-donors", out=tmp_path / "expanded-population",
        samples=128,draws=2,methods="learned",settings="dense",seed=137791,
        phase="development",num_threads=threads,memory_gib=2))
    result = json.loads((tmp_path / "expanded-population/results.json").read_text())
    assert len(result)==1 and result[0]["numerical_failures"]==0
    with pytest.raises(ValueError,match="retain the selected confirmation sample size"):
        population_run(Namespace(input=tmp_path / "all-donors", out=tmp_path / "wrong-sample-size",
            samples=256,draws=2,methods="learned",settings="dense",seed=137791,
            phase="development",num_threads=threads,memory_gib=2))


def _check_multiple_targets(tmp_path, first, threads):
    """Different target axes/models must retain their separate confirmation means."""
    from copy import deepcopy
    from scripts.epistasis.make_trans_inputs import build
    from summit.epistasis.cli import main
    from summit.epistasis.robust import load_robust_scores

    second = tmp_path / "constructed-second"
    build(Namespace(out=second, genotypes=str(tmp_path / "g.bed"),
        genome_build="GRCh37", target="v1", background_chromosome="5",
        training_samples=tmp_path / "fit/training.tsv", confirmation_samples=tmp_path / "fit/confirmation.tsv",
        phenotypes=tmp_path / "fit/dense_000/phenotype.tsv", phenotype_column="y", unit="fixed reference units",
        covariates=tmp_path / "ref/covariates.tsv", covariate_columns="", structure_covariates="PC1",
        local_window_bp=100000, minimum_cell_fraction=.02, num_threads=threads))
    for command, manifest, output in [("train-direction", "train.json", "trained"), ("prepare", "prepare.json", "prepared")]:
        main([command, str(second / manifest), "--out", str(second / output), "--num-threads", str(threads)])
    from scripts.epistasis.full_matched_multitarget import run as combine
    specs = [json.loads((root / "prepare.json").read_text()) for root in (first, second)]
    report = combine(Namespace(preparations=[first / "prepare.json", second / "prepare.json"],
        jobs=[s["jobs"][0]["id"] for s in specs], out=tmp_path / "multiple-targets",
        num_threads=threads, memory_gib=2))
    assert report["independent_new_learning_models"] == 0
    assert len(report["comparisons"]) == 2
    changed = deepcopy(specs[1])
    table = pd.read_csv(second / changed["covariates"]["file"],sep="\t")
    table["PC1"] += 1e-7
    table.to_csv(second / "changed-covariates.tsv",sep="\t",index=False)
    changed["covariates"]["file"] = "changed-covariates.tsv"
    (second / "changed-prepare.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="identical genotype, samples, covariates"):
        combine(Namespace(preparations=[first / "prepare.json",second / "changed-prepare.json"],
            jobs=[s["jobs"][0]["id"] for s in specs],out=tmp_path / "changed-multiple-targets",
            num_threads=threads,memory_gib=2))


def _check_public_batch(tmp_path, monkeypatch, threads):
    from summit.epistasis.cli import main as epistasis_main
    def main(arguments):
        return epistasis_main(arguments if arguments[0] == "fit" else [*arguments, "--num-threads", str(threads)])
    from summit.epistasis.robust import load_robust_scores
    from summit.prediction.artifacts import load_prediction_models
    from scripts.epistasis.full_matched import load_reference

    root = tmp_path / "fit"
    traits = ["dense", "mixed"]
    y = pd.read_csv(root / "dense_000/phenotype.tsv", sep="\t", dtype={"FID": str, "IID": str})
    y = y.rename(columns={"y": "dense"})
    y["mixed"] = pd.read_csv(root / "mixed_000/phenotype.tsv", sep="\t").y
    y.to_csv(root / "batch-y.tsv", sep="\t", index=False)
    train = json.loads((root / "dense_000/train.json").read_text())
    del train["phenotype"]
    train["phenotypes"] = dict(file="batch-y.tsv", columns=traits, unit="fixed reference units")
    (root / "batch-train.json").write_text(json.dumps(train))
    main(["train-direction", str(root / "batch-train.json"), "--out", str(root / "batch-trained")])
    main(["train-direction", str(root / "batch-train.json"), "--out", str(root / "batch-trained"), "--resume"])
    models = {m.key:m for m in load_prediction_models(root / "batch-trained/models")}
    for i, name in enumerate(traits):
        for single in load_prediction_models(root / f"{name}_000/trained/models"):
            np.testing.assert_allclose(models[(f"direction.{i}",single.model_id)].weights,
                single.weights, atol=2e-9, rtol=2e-8)
    prepare = json.loads((root / "dense_000/prepare.json").read_text())
    template = prepare["jobs"][0]
    prepare["phenotypes"] = train["phenotypes"]
    prepare["jobs"], prepare["frozen_scores"] = [], []
    for i, name in enumerate(traits):
        pgs, score = f"pgs{i}", f"score{i}"
        prepare["frozen_scores"].extend([
            dict(name=pgs, direction=f"batch-trained/direction.{i}.json", component=0),
            dict(name=score, direction=f"batch-trained/direction.{i}.json", component=1),
        ])
        prepare["jobs"].append(dict(template, id=name, phenotypes=[name], adjust_scores=[pgs],
            components=[dict(name="learned", frozen_score=score, background="target")]))
    (root / "batch-prepare.json").write_text(json.dumps(prepare))
    main(["prepare", str(root / "batch-prepare.json"), "--out", str(root / "batch-prepared")])
    for name in traits:
        batch = load_robust_scores(root / f"batch-prepared/{name}.robust-score.npz")
        single = load_robust_scores(root / f"{name}_000/prepared/learned.robust-score.npz")
        assert batch.trait_names == (name,)
        assert batch.metadata["complete_variant_count"] == 96
        assert len(batch.metadata["variants"]["ids"]) < 96
        for field in ("scores", "information", "score_covariance"):
            np.testing.assert_allclose(getattr(batch, field), getattr(single, field),
                                       rtol=5e-8, atol=1e-10)
    # Independently published bundles both use the default model key. Their
    # identities and phenotype-specific means must remain distinct in a batch.
    for i,name in enumerate(traits):
        for score in prepare["frozen_scores"][2*i:2*i+2]:
            score["direction"] = f"{name}_000/trained/direction.json"
    (root / "separate-models-prepare.json").write_text(json.dumps(prepare))
    main(["prepare",str(root / "separate-models-prepare.json"),"--out",str(root / "separate-models-prepared")])
    report = json.loads((root / "separate-models-prepared/preparation.json").read_text())
    names = report["frozen_score_report"]["frozen_model_names"]
    assert len(names)==4 and len({v["identity"] for v in names.values()})==4
    assert len({v["trait_id"] for v in names.values()})==1
    for name in traits:
        batch = load_robust_scores(root / f"separate-models-prepared/{name}.robust-score.npz")
        single = load_robust_scores(root / f"{name}_000/prepared/learned.robust-score.npz")
        for field in ("scores","information","score_covariance"):
            np.testing.assert_allclose(getattr(batch,field),getattr(single,field),rtol=5e-8,atol=1e-10)
    # Reusing a learned direction for a new trait is a separate frozen-direction
    # question. It must recompute nuisance coefficients and trait-specific meat.
    reuse = dict(kind="summit.epistasis.prepare_traits", schema_version=1,
        reference="batch-prepared/dense.cohort-reference.npz",
        samples=prepare["samples"], phenotypes=train["phenotypes"])
    (root / "reuse.json").write_text(json.dumps(reuse))
    import summit.prediction.genotype as genotype
    def unavailable(*a, **k):
        raise AssertionError("genotype access during summary/reuse")
    with monkeypatch.context() as patch:
        patch.setattr(genotype, "source_from_spec", unavailable)
        patch.setattr(genotype, "FileGenotypeSource", unavailable)
        main(["prepare-traits", str(root / "reuse.json"), "--out", str(root / "reused.npz")])
        main(["fit", str(root / "reused.npz"), "--out", str(root / "reused.fit.json")])
    reference = load_reference(root / "batch-prepared/dense.cohort-reference.npz")
    selected = pd.read_csv(prepare["samples"], sep="\t", dtype=str)
    yy = selected.merge(y, on=["FID", "IID"], validate="one_to_one")[traits].to_numpy()
    x = np.column_stack([reference.fixed_effects, reference.features])
    inverse = np.linalg.pinv(x, rcond=1e-11)
    coef = inverse @ yy
    error = yy - x @ coef
    leverage = np.sum(x * inverse.T, axis=1)
    fits = json.loads((root / "reused.fit.json").read_text())["fits"]
    for i, fit in enumerate(fits):
        se = np.linalg.norm(inverse[-1] * error[:, i] / (1 - leverage))
        np.testing.assert_allclose(fit["beta"], coef[-1, i], atol=1e-9)
        np.testing.assert_allclose(fit["standard_errors"], se, atol=1e-9)
    changed = dict(train, phenotypes=dict(train["phenotypes"], columns=traits[::-1]))
    (root / "changed-train.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inputs changed"):
        main(["train-direction", str(root / "changed-train.json"), "--out", str(root / "batch-trained"), "--resume"])


def _check_training_nuisance_invariance(root, threads):
    """Conditional inference requires the learner to use only C-free outcomes."""
    from summit.epistasis.cli import main as cli
    from summit.prediction.cli import _aligned_table
    from summit.prediction.artifacts import load_prediction_models
    from bed_reader import open_bed

    spec = json.loads((root / "train.json").read_text())
    original = load_prediction_models(root / "trained/models")
    phenotype = spec["phenotype"]
    table = pd.read_csv(root / phenotype["file"], sep="\t", dtype={"FID": str, "IID": str})
    samples = list(table[["FID", "IID"]].itertuples(index=False, name=None))
    cv = spec["covariates"]
    z = _aligned_table(root / cv["file"], samples)["PC1"].to_numpy(float)
    with open_bed(root / spec["genotypes"]["geno"]) as source:
        j = list(source.sid).index(spec["target"])
        source_samples = list(zip(source.fid, source.iid))
        order = [source_samples.index(s) for s in samples]
        x = source.read(index=np.s_[:, j:j+1], num_threads=threads)[order, 0]
    assert np.all(np.isfinite(x))
    # Every term belongs to the actual declared public training mean,
    # including dominance and covariate-dependent target main effects.
    table[phenotype["column"]] += .7*z + .25*z*z - .4*(x == 1) + .2*x*z
    table.to_csv(root / "mean-shifted.tsv", sep="\t", index=False)
    spec["phenotype"] = dict(phenotype, file="mean-shifted.tsv")
    (root / "mean-shifted.json").write_text(json.dumps(spec))
    cli(["train-direction", str(root / "mean-shifted.json"), "--out", str(root / "mean-shifted"),
         "--num-threads", str(threads)])
    changed = load_prediction_models(root / "mean-shifted/models")
    assert len(original) == len(changed) == 2
    for before, after in zip(original, changed):
        assert before.model_id == after.model_id
        np.testing.assert_allclose(after.weights, before.weights, atol=1e-9, rtol=1e-8)
    from summit.epistasis.conditional import frozen_local_mean
    from summit.prediction.genotype import FileGenotypeSource
    from summit.prediction.cli import _rows
    with FileGenotypeSource(root/spec['genotypes']['geno'],genome_build='GRCh37') as source:
        train=_rows(source,root/spec['samples'])
        test=_rows(source,root/'confirmation.tsv')
        mean=frozen_local_mean(source,np.sort(train),np.sort(test),spec,root,threads=threads)
    provenance=original[0].provenance['training'][original[0].trait_id]
    assert mean['metadata']['training_fixed_hash']==provenance['fixed']
    assert mean['metadata']['training_context_hash']==provenance['phi']


def test_nuisance_prediction_diagnostic_separates_alignment_and_shrinkage():
    from scripts.epistasis.public_conditional_validation import nuisance_prediction_quality
    rng=np.random.default_rng(9123)
    fixed=np.column_stack([np.ones(240),rng.normal(size=(240,2))])
    basis=np.linalg.qr(fixed)[0]
    mean=fixed@np.array([3.,-1.,2.])+rng.normal(size=240)
    predictions={'shrunk':.25*mean,'unrelated':rng.normal(size=240)}
    result=nuisance_prediction_quality(mean,predictions,basis)
    target=mean-fixed@np.linalg.lstsq(fixed,mean,rcond=None)[0]
    np.testing.assert_allclose(result['residual_mean_variance'],np.mean(target**2),atol=1e-13)
    for name,prediction in predictions.items():
        residual=prediction-fixed@np.linalg.lstsq(fixed,prediction,rcond=None)[0]
        np.testing.assert_allclose(result['predictors'][name]['squared_alignment'],
            np.corrcoef(target,residual)[0,1]**2,atol=1e-13)
        np.testing.assert_allclose(result['predictors'][name]['unit_scale_r2'],
            1.-np.sum((target-residual)**2)/np.sum(target**2),atol=1e-13)
    assert result['predictors']['shrunk']['squared_alignment']>1.-1e-12
    np.testing.assert_allclose(result['predictors']['shrunk']['unit_scale_r2'],1.-.75**2)
    empty=nuisance_prediction_quality(fixed[:,0],predictions,basis)
    assert all(v['squared_alignment'] is None and v['unit_scale_r2'] is None
        for v in empty['predictors'].values())


def test_varying_main_effect_span_and_interaction_identification():
    from summit.epistasis.nuisance import varying_main_effects
    rng = np.random.default_rng(773)
    z = rng.normal(size=(700, 2))
    x = z[:, 0] + rng.normal(size=700)
    e = z[:, 1] + rng.normal(size=700)
    fixed = np.column_stack([np.ones(700), z, x, e])
    expanded, record = varying_main_effects(fixed, z, np.column_stack([x, e]),
        covariate_names=["PC1", "PC2"], main_names=["target", "score"], memory_bytes=2**30)
    mean = 2*z[:, 0]**2 + 3*z[:, 0]*z[:, 1] - z[:, 1]**2 + z[:, 0]*x - 2*z[:, 1]*x + .5*z[:, 0]*e + z[:, 1]*e
    np.testing.assert_allclose(expanded @ np.linalg.lstsq(expanded, mean, rcond=1e-12)[0], mean, atol=1e-12)
    interaction = x*e
    remaining = interaction-expanded@np.linalg.lstsq(expanded, interaction, rcond=1e-12)[0]
    assert np.mean(remaining**2) > .7
    changed, _ = varying_main_effects(fixed, z@np.array([[2., .3], [.5, -1.]]), np.column_stack([x, e]),
        covariate_names=["A", "B"], main_names=["target", "score"], memory_bytes=2**30)
    np.testing.assert_allclose(changed @ np.linalg.lstsq(changed, expanded, rcond=1e-12)[0], expanded, atol=1e-12)
    assert record["added_columns"] == 7


def test_population_regression_truth_and_sampled_hc3():
    from scripts.epistasis.full_matched_population import PopulationRegression
    rng = np.random.default_rng(161823)
    z = rng.normal(size=(900, 3))
    c = np.column_stack([np.ones(len(z)), z, z[:, 0]**2, 2*z[:, 0]])
    f = np.column_stack([z[:, 0]*z[:, 1], z[:, 0]*z[:, 2]])
    mean = .3*z[:, 0] + .1*z[:, 2]**2 + f@np.array([.4,-.2])
    variance = .4 + z[:, 0]**2
    regression = PopulationRegression(c, f)
    truth, cov = regression.truth(mean, variance, 400)
    full = np.column_stack([c,f])
    inverse = np.linalg.pinv(full, rcond=1e-11)
    generating = inverse@mean
    leftover = mean-full@generating
    np.testing.assert_allclose(truth, generating[-2:], atol=1e-12)
    # Finite donor law sandwich includes random-design mean misspecification.
    expected = (inverse[-2:]*(variance+leftover**2))@inverse[-2:].T * len(c)/400
    np.testing.assert_allclose(cov, expected, atol=1e-12)
    donors = rng.integers(len(c), size=400)
    y = mean[donors]+np.sqrt(variance[donors])*rng.normal(size=400)
    fitted = regression.fit(donors, y)
    x = full[donors]
    inverse = np.linalg.pinv(x, rcond=1e-11)
    coef = inverse@y
    hat = np.sum(x*inverse.T,axis=1)
    influence = inverse[-2:] * ((y-x@coef)/(1-hat))
    np.testing.assert_allclose(fitted["beta"],coef[-2:],atol=1e-12)
    np.testing.assert_allclose(fitted["coefficient_covariance"],influence@influence.T,atol=1e-12)


def test_frozen_score_constituents_exclude_exact_zero_weights():
    from types import SimpleNamespace
    from summit.epistasis.models import target_design
    from summit.prediction.genotype import ArrayGenotypeSource
    from summit.prediction.spec import VariantAxis
    rng = np.random.default_rng(44197)
    g = rng.binomial(2,.4,size=(80,3)).astype(float)
    axis = VariantAxis(("target","a","b"),("1","2","2"),(1,2,3),("A",)*3,("C",)*3)
    source = ArrayGenotypeSource(g,[(str(i),str(i)) for i in range(len(g))],axis,hard_calls=True)
    scale = SimpleNamespace(mean=g.mean(0),inverse_scale=np.ones(3))
    frozen = dict(values=g[:,1]+2*g[:,2], identity="all weights and full axis identified",
        variants=axis.ids, nonzero_variants=["a","b"])
    arguments = dict(components=[dict(name="direction",frozen_score="score",background="T")],
        annotations=dict(all=np.ones(3),T=np.array([1.,0,0])),additive_annotations=["all"],
        local_variants=["target"],frozen_scores=dict(score=frozen))
    design = target_design(source,np.arange(len(g)),scale,**arguments)
    assert design["definitions"]["interactions"][0]["score_variants"] == ["a","b"]
    arguments["frozen_scores"] = dict(score=dict(frozen,nonzero_variants=list(axis.ids)))
    with pytest.raises(ValueError,match="overlaps"):
        target_design(source,np.arange(len(g)),scale,**arguments)
    # Old records without a nonzero-axis record remain conservative.
    arguments["frozen_scores"]["score"].pop("nonzero_variants")
    with pytest.raises(ValueError,match="overlaps"):
        target_design(source,np.arange(len(g)),scale,**arguments)


def test_prespecified_score_cache_uses_genotype_definition_only():
    from summit.epistasis.models import target_design
    from summit.prediction.genotype import ArrayGenotypeSource, estimate_scale
    from summit.prediction.spec import VariantAxis
    rng = np.random.default_rng(18457)
    raw = rng.binomial(2,.4,size=(90,5)).astype(float)
    raw[0,1] = np.nan
    axis = VariantAxis(tuple("tabcd"),("1","2","2","2","2"),tuple(range(1,6)),("A",)*5,("C",)*5)
    source = ArrayGenotypeSource(raw,[("f",str(i)) for i in range(len(raw))],axis,hard_calls=True)
    rows = np.arange(len(raw))
    scale = estimate_scale(source,rows,np.arange(5))
    arguments = dict(components=[dict(name="score",score=dict(a=1.,b=-.5),background="T")],
        annotations=dict(all=np.ones(5),T=np.array([1.,0,0,0,0])),additive_annotations=["all"],local_variants=["t"])
    cache = {}
    first = target_design(source,rows,scale,score_cache=cache,**arguments)
    second = target_design(source,rows,scale,score_cache=cache,**arguments)
    np.testing.assert_array_equal(first["modifiers"],second["modifiers"])
    assert first["definitions"]["targeted_variant_reads"] == 3
    assert second["definitions"]["targeted_variant_reads"] == 1
    assert len(cache) == 1
    changed = target_design(source,rows,scale,score_cache=cache,main_imputation={"a":1.8},**arguments)
    independent = target_design(source,rows,scale,main_imputation={"a":1.8},**arguments)
    np.testing.assert_array_equal(changed["modifiers"],independent["modifiers"])
    assert len(cache) == 2 and changed["modifiers"][0,1] != first["modifiers"][0,1]
    subset = rows[1:]
    smallscale = estimate_scale(source,subset,np.arange(5))
    changed = target_design(source,subset,smallscale,score_cache=cache,**arguments)
    independent = target_design(source,subset,smallscale,**arguments)
    np.testing.assert_array_equal(changed["modifiers"],independent["modifiers"])
    assert len(cache) == 3


def test_separable_population_restriction_does_not_need_the_true_mean():
    from itertools import product
    from summit.epistasis.nuisance import varying_main_effects
    from scipy.stats import binom
    rows, probability = [], []
    for z in (-1.,1.):
        # Dependence within each chromosome block; independence between blocks
        # conditional on Z is the explicit algebraic restriction in this test.
        for x,l,b,d in product(range(3),repeat=4):
            p = .5*binom.pmf(x,2,.35+.08*z)*binom.pmf(l,2,.2+.2*x)
            p *= binom.pmf(b,2,.4+.1*z)*binom.pmf(d,2,.15+.25*b)
            rows.append((z,x,l,b,d));probability.append(p)
    z,x,l,b,d = np.asarray(rows).T
    probability = np.asarray(probability)
    np.testing.assert_allclose(probability.sum(),1.)
    e = np.column_stack([b-.7*d,b+d])
    pgs = .4*x+.3*l+.6*b-.2*d
    c = np.column_stack([np.ones(len(x)),z,x,x==1,e,pgs])
    expanded,_ = varying_main_effects(c,z[:,None],np.column_stack([x,e,pgs]),
        covariate_names=["Z"],main_names=["x","e1","e2","PGS"],memory_bytes=2**30)
    mean = x*x+.6*l*l+.7*z*l + b*b-.5*d*d+z*d
    f = x[:,None]*e
    weight = np.sqrt(probability)
    design = np.column_stack([expanded,f])
    beta = np.linalg.lstsq(design*weight[:,None],mean*weight,rcond=1e-11)[0]
    np.testing.assert_allclose(beta[-2:],0.,atol=1e-12)
    assert np.linalg.norm((mean-design@beta)*weight) > .1  # Mean is not spanned.
    without = np.column_stack([c,f])
    naive = np.linalg.lstsq(without*weight[:,None],mean*weight,rcond=1e-11)[0]
    assert np.linalg.norm(naive[-2:]) > .01
    alternative = mean+.3*x*(b+2*d)
    beta = np.linalg.lstsq(design*weight[:,None],alternative*weight,rcond=1e-11)[0]
    expected = .3*np.linalg.solve(np.array([[1.,1.],[-.7,1.]]),np.array([1.,2.]))
    np.testing.assert_allclose(beta[-2:],expected,atol=1e-12)


def test_conditional_coverage_is_about_interaction_truth_not_displaced_mean():
    from scipy.stats import norm,t
    from scripts.epistasis.public_conditional_validation import outcome_reference
    i0=np.array([0,1]);i1=np.array([2,3,4])
    a=np.array([.5,0.,0.]);b=np.array([.2,-.1])
    signal=np.array([.3,-.2,2.,0.,0.]);mean=signal+np.array([.1,.2,.8,0.,0.])
    y0=signal[i0]+np.array([.4,-.2]);variance=np.ones(5)*4
    target=float(a@signal[i1]+b@signal[i0]);bias=.5*.8+.2*.4+.1*.2
    for law in ('gaussian','t5_unit_variance'):
        result=outcome_reference(a,b,-b,1.,signal=signal,mean=mean,variance=variance,
            y0=y0,i0=i0,i1=i1,definition=dict(sampling_law='fixed_architecture',error_law=law),se=1.1)
        np.testing.assert_allclose(result['conditional_interaction_truth'],target,atol=1e-15)
        np.testing.assert_allclose(result['conditional_mean_bias'],bias,atol=1e-15)
        for alpha in (.05,.005):
            threshold=norm.isf(alpha/2)*1.1
            cdf=norm.cdf if law=='gaussian' else lambda value:t.cdf(value/np.sqrt(3/5),df=5)
            expected_coverage=cdf(threshold-bias)-cdf(-threshold-bias)
            expected_power=1-cdf(threshold-target-bias)+cdf(-threshold-target-bias)
            np.testing.assert_allclose(result['conditional_coverage_probability'][str(1-alpha)],expected_coverage,atol=3e-9)
            np.testing.assert_allclose(result['conditional_rejection_probability'][str(alpha)],expected_power,atol=3e-9)
            assert expected_coverage<cdf(threshold)-cdf(-threshold)-1e-5
            assert abs(expected_power-(1-expected_coverage))>.01


def test_population_failure_denominators_do_not_report_spurious_coverage():
    from scripts.epistasis.full_matched_population import aggregate_models
    common = dict(setting="dense",method="learned",scheduled_draws=128)
    records = [dict(common,replicate=0,failed=True,failure_stage="parent_pipeline"),
        dict(common,replicate=1,failed=True,failure_stage="population_reference"),
        dict(common,replicate=2,failed=False,numerical_failures=128,unsupported=0,
            denominator_all_numerical=0,truth=[.2],known_population_covariance=[[1.]])]
    result = aggregate_models(records)[0]
    assert result["scheduled_learning_models"]==3 and result["scheduled_conditional_draws"]==384
    assert result["unavailable_models"]==2 and result["numerical_failures"]==128
    assert result["failure_stages"]==dict(parent_pipeline=1,population_reference=1)
    assert result["models_with_coverage"]==0 and result["mean_projection_coverage"] is None
    assert result["mean_rate_05"] is None and result["model_mc_interval_05"] is None
