import importlib.util
from pathlib import Path
import numpy as np
from types import SimpleNamespace

from summit.context.cross_trait_fit import cross_trait_derived


def test_joint_gaussian_truth_is_positive_and_zero_program_is_centered():
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_simulation.py'
    spec=importlib.util.spec_from_file_location('cross_trait_simulation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mx=np.array([.13,-.08]);my=np.array([-.09,.17])
    models,psi=module.truth_covariances(mx,my)
    assert np.linalg.eigvalsh(models).min()>0 and np.linalg.eigvalsh(psi).min()>0
    assert not np.allclose(models[0,:3,3:],models[0,:3,3:].T)
    for i,model in enumerate(models):
        np.testing.assert_allclose(module.psd_root(model)@module.psd_root(model).T,model,atol=1e-15)
        derived=cross_trait_derived(model[:3,3:],model[:3,:3],model[3:,3:],
            mean_x=mx,mean_y=my,context_covariance=np.eye(2))
        assert derived['centered_baseline_rg']>0
        expected=np.array([[.020,.009],[-.006,.018]]) if i==0 else np.zeros((2,2))
        np.testing.assert_allclose(derived['h_xy'],expected,atol=1e-16)
    assert np.any(psi[:3,3:][1:,1:]!=0)


def test_simulation_pipeline_small_bed(tmp_path):
    from bed_reader import to_bed
    from summit import gxeldcore
    from summit.context.spec import ContextPairIndex
    from summit.context.cross_trait_zpass import load_array_artifact
    path=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_simulation.py'
    spec=importlib.util.spec_from_file_location('simulation_pipeline',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(1723);n,m=141,179
    raw=rng.binomial(2,rng.uniform(.2,.8,m),size=(n,m)).astype(float)
    bed=tmp_path/'geno';to_bed(str(bed)+'.bed',raw)
    phi=np.c_[np.ones(n),rng.normal(size=(n,2))];u=np.linalg.qr(phi)[0]
    d=np.column_stack([phi[:,p.q]*phi[:,p.r] for p in ContextPairIndex(3).entries])
    axes=tmp_path/'axes.npz'
    np.savez(axes,sample_ids=np.array(module.read_plink_axes(bed).sample_ids),basis=phi,fixed_basis=u,
        residual_basis=d,residual_names=np.array([str(i) for i in range(6)]),
        affine_mean=raw.mean(0),affine_inverse_scale=1/raw.std(0,ddof=1))
    args=SimpleNamespace(axes=axes,bed_prefix=bed,output=tmp_path/'ref',blocks=5,probes=32,
        seed=2901,threads=int(gxeldcore.build_info()['blas_runtime_threads']),memory_gib=1,width=31,
        replicates=2,gram_mode='factorized')
    args.output.mkdir();module.reference_run(args)
    import json
    args.reference=Path(json.loads((args.output/'COMPLETE.json').read_text())['reference'])
    # The current reference is the authoritative floating-point affine scale.
    ref=module.load_generalized_gxe_variant_reference_v1(args.reference)
    # The same variant probes and guarded two-pass estimator must agree when
    # pass 2 groups more probes into each matrix product.
    tiled=SimpleNamespace(**vars(args));tiled.output=tmp_path/'ref_tiled';tiled.probe_tile_width=16
    tiled.output.mkdir();module.reference_run(tiled)
    other=module.load_generalized_gxe_variant_reference_v1(
        json.loads((tiled.output/'COMPLETE.json').read_text())['reference'])
    for name in ('block_directed_numerator','same_person','affine_mean','affine_inverse_scale'):
        np.testing.assert_allclose(getattr(other,name),getattr(ref,name),rtol=1e-12,atol=1e-12)
    with np.load(axes) as z:values={key:z[key] for key in z.files}
    values.update(affine_mean=ref.affine_mean,affine_inverse_scale=ref.affine_inverse_scale)
    np.savez(axes,**values)
    args.output=tmp_path/'gen';args.output.mkdir();module.generate(args)
    args.generated=args.output/'generated.npz'
    args.output=tmp_path/'score';args.output.mkdir();module.score(args)
    args.scores=args.output/'scores.npz'
    args.output=tmp_path/'fit';args.output.mkdir();module.fit(args)
    result,_=load_array_artifact(args.output/'fits.npz',kind='summit.cross_trait.simulation_fit')
    assert result['omega'].shape==(2,2,3,3,3)
    assert result['loo_omega'].shape==(2,2,3,5,3,3)
    assert (args.output/'simulation_summary.tsv').exists()
