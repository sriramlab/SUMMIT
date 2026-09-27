"""Joint-Gaussian cross-trait simulation using the existing 50,112-row panel.

Generation has one genotype traversal for all replicates and scenarios. The
score pass is a second, single traversal for every requested pair. Reference
moments are reused, and genotype-independent fits need no further traversal.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import os
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from bed_reader import open_bed
from summit.context.cross_trait_zpass import write_array_artifact,load_array_artifact
from summit.context.reference_zpass_cli import file_sha256
from summit.context.spec import canonical_sha256
from summit.context.cross_trait_gram import chromosome_gram,orientation_matrix
from summit.context.cross_trait_fit import CrossTraitMomentPlan,fit_cross_trait,fit_cross_trait_rhs_batch,cross_trait_derived
from summit.context.cross_trait_uncertainty import derived_uncertainty
from summit.ldscore.generalized_gxe_reference_v1 import load_generalized_gxe_variant_reference_v1
from summit.ldscore.generalized_gxe_pass1 import ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import ProtectedTNOperator
from summit.ldscore.generalized_gxe_masked_batch import MaskedTraitBatch
from summit.ldscore.generalized_gxe_cross_trait_batch import CrossTraitBatch,CrossTraitScoreAccumulator
from workflow import read_plink_axes,run_reference,balanced_inference_block_ids


def reference_run(args):
    """Rebuild obsolete simulation references through the current native path."""
    from summit import gxeldcore
    with np.load(args.axes,allow_pickle=False) as z:
        phi,u,names,ids=(z[k] for k in ('basis','fixed_basis','residual_names','sample_ids'))
    axes=read_plink_axes(args.bed_prefix)
    np.testing.assert_array_equal(np.asarray(axes.sample_ids),ids)
    groups,labels=balanced_inference_block_ids(axes.m,args.blocks)
    result=run_reference(axes=axes,basis=phi,basis_names=('intercept','environment_1','environment_2'),
        fixed=u,annotations=np.ones((axes.m,1)),annotation_names=('all_variants',),
        inference_block_ids=groups,inference_block_labels=labels,residual_names=tuple(names.astype(str)),
        probes=args.probes,seed=args.seed,threads=args.threads,memory_bytes=args.memory_gib*2**30,
        native_module=gxeldcore,output=args.output/'reference',include_directional_panel=False,
        mode='summary',variant_block_width=args.width,
        probe_tile_width=getattr(args,'probe_tile_width',4),
        source_probe_tile_width=args.probes)
    artifact=result.artifact_path
    with (args.output/'COMPLETE.json').open('x') as f:
        json.dump(dict(reference=str(artifact),sha256=file_sha256(artifact),
            axes_sha256=file_sha256(args.axes),script_sha256=file_sha256(__file__),
            ledger=result.artifact.manifest['performance_ledger']),f,indent=2)


def truth_covariances(mean_x,mean_y):
    """Two prespecified positive-definite models, centered at the study means."""
    q=3;baseline=np.array([[.30,.16],[.16,.32]])
    hx=np.array([[.040,.006],[.006,.035]]);hy=np.array([[.045,-.004],[-.004,.040]])
    hxy=np.array([[.020,.009],[-.006,.018]])
    load=np.eye(6);load[1:3,0]=[.2,-.1];load[4:6,3]=[-.1,.25]
    cx=np.eye(q);cy=np.eye(q);cx[0,1:]=mean_x;cy[0,1:]=mean_y
    undo=np.zeros((6,6));undo[:3,:3]=np.linalg.inv(cx);undo[3:,3:]=np.linalg.inv(cy)
    models=[]
    for cross in (hxy,np.zeros((2,2))):
        core=np.zeros((6,6));core[np.ix_([0,3],[0,3])]=baseline
        core[1:3,1:3]=hx;core[4:6,4:6]=hy;core[1:3,4:6]=cross;core[4:6,1:3]=cross.T
        model=undo@load@core@load.T@undo.T
        if np.linalg.eigvalsh(model).min()<=0:raise ValueError('simulation genetic truth is not positive definite')
        models.append(model)
    sd=np.sqrt([.55,.025,.020,.52,.023,.022]);corr=np.eye(6)
    for i,j,rho in ((0,3,.3),(1,4,.25),(1,5,.15),(2,4,-.1),(2,5,.2),(0,1,.1),(3,5,-.1)):
        corr[i,j]=corr[j,i]=rho
    psi=corr*sd[:,None]*sd[None]
    assert np.linalg.eigvalsh(psi).min()>0
    return np.stack(models),psi


def psd_root(value):
    ev,u=np.linalg.eigh(value)
    if ev.min()<-1e-12:raise ValueError('indefinite generating covariance')
    return (u*np.sqrt(np.maximum(ev,0)))@u.T


def load_design(args):
    reference=load_generalized_gxe_variant_reference_v1(args.reference)
    with np.load(args.axes,allow_pickle=False) as z:
        arrays={key:z[key] for key in ('sample_ids','basis','fixed_basis','residual_basis','residual_names',
                                      'affine_mean','affine_inverse_scale')}
    panel=read_plink_axes(args.bed_prefix)
    np.testing.assert_array_equal(np.asarray(panel.sample_ids),arrays['sample_ids'])
    for name in ('affine_mean','affine_inverse_scale'):
        np.testing.assert_allclose(arrays[name],getattr(reference,name),rtol=1e-10,atol=1e-12)
        # Use the current reference's exact affine handoff, including rounding.
        arrays[name]=getattr(reference,name)
    n,q=arrays['basis'].shape
    if n!=reference.n_samples or panel.m!=reference.n_variants or q!=3:
        raise ValueError('simulation panel and reference axes differ')
    groups=np.asarray(reference.manifest['axes']['jackknife_blocks']['variant_block_ids'],dtype=int)
    nx=int(n/1.4);rows_x=np.arange(nx);rows_y=np.arange(n-nx,n)
    return reference,arrays,panel,groups,rows_x,rows_y


def genotype_tiles(args,arrays,m):
    with open_bed(str(args.bed_prefix)+'.bed',num_threads=args.threads) as bed:
        for begin in range(0,m,args.width):
            end=min(begin+args.width,m)
            g=bed.read(index=np.s_[:,begin:end],dtype='float64',order='F')
            g-=arrays['affine_mean'][begin:end];g*=arrays['affine_inverse_scale'][begin:end]
            np.nan_to_num(g,copy=False,nan=0.)
            yield begin,end,g


def base_provenance(args):
    return dict(reference_sha256=file_sha256(args.reference),axes_sha256=file_sha256(args.axes),
        script_sha256=file_sha256(__file__),seed=args.seed,replicates=args.replicates,
        genotype_files={s:file_sha256(str(args.bed_prefix)+s) for s in ('.bim','.fam')},
        bed_size=Path(str(args.bed_prefix)+'.bed').stat().st_size,threads=args.threads,width=args.width)


def generate(args):
    start=time.monotonic();reference,arrays,panel,groups,x,y=load_design(args)
    provenance=base_provenance(args);phi,u=arrays['basis'],arrays['fixed_basis'];n,q=phi.shape;m=panel.m
    omega,psi=truth_covariances(phi[x,1:].mean(0),phi[y,1:].mean(0))
    latent=np.zeros((n,args.replicates*2*q));diagonal=np.zeros((q,q,n))
    effects=np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[0])
    operator=ProtectedNNOperator(threads=args.threads);operator.begin_execution()
    build=dict(operator._module.build_info())
    if not build['gemm_integrity_enabled'] or not build['gemm_checksum_enabled']:
        raise RuntimeError('simulation generation requires guarded native products')
    telemetry=[];visits=0
    for begin,end,g in genotype_tiles(args,arrays,m):
        innovations=np.asfortranarray(effects.standard_normal((end-begin,args.replicates*2*q))/np.sqrt(m))
        latent+=operator.matmul(g,innovations)
        # Exact reference-person diagonals, collected during generation.
        f=np.stack([phi[:,a,None]*g for a in range(q)])
        f-=np.einsum('ic,acj->aij',u,np.einsum('ic,aij->acj',u,f,optimize=True),optimize=True)
        diagonal+=np.einsum('aij,bij->abi',f,f,optimize=True)/m
        telemetry.append(canonical_sha256(operator.finish_execution()));visits=end
        if (begin//args.width)%64==0 or end==m:
            print(json.dumps(dict(phase='generate',variants=end,total=m,seconds=time.monotonic()-start)),flush=True)
    latent=latent.reshape(n,args.replicates,2*q)
    noise_rng=np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[1])
    noise=noise_rng.standard_normal(latent.shape)@psd_root(psi).T
    residual=np.einsum('nq,nrtq->nrt',phi,noise.reshape(n,args.replicates,2,q),optimize=True)
    phenotypes=np.stack([np.einsum('nq,nrtq->nrt',phi,(latent@psd_root(model).T).reshape(n,args.replicates,2,q),
                                   optimize=True)+residual for model in omega])
    if visits!=m:raise RuntimeError('simulation generation traversal incomplete')
    saved_diagonal=orientation_matrix(q)@diagonal.reshape(q*q,n)
    diagonal_error=np.linalg.norm(saved_diagonal@saved_diagonal.T-reference.same_person)/np.linalg.norm(reference.same_person)
    if diagonal_error>1e-9:
        raise RuntimeError(f'generated reference diagonals disagree with the sealed exact same-person term: {diagonal_error}')
    provenance.update(seconds=time.monotonic()-start,genotype_traversals=1,variant_visits=visits,native_build=build,
        telemetry_sha256=telemetry,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scenario_names=['shared_nonsymmetric_program','zero_orthogonal_program'],same_person_relative_error=float(diagonal_error),
        truth_basis='master basis; zero H is specified after centering separately at the two study means')
    write_array_artifact(args.output/'generated.npz',kind='summit.cross_trait.simulation',arrays=dict(
        phenotypes=phenotypes,omega=omega,psi=psi,rows_x=x,rows_y=y,
        reference_diagonal=saved_diagonal),provenance=provenance)


def score(args):
    start=time.monotonic();reference,arrays,panel,groups,x,y=load_design(args)
    generated,gprov=load_array_artifact(args.generated,kind='summit.cross_trait.simulation')
    if gprov['reference_sha256']!=file_sha256(args.reference) or gprov['axes_sha256']!=file_sha256(args.axes):
        raise ValueError('simulation generation identity differs')
    phi,u,d=arrays['basis'],arrays['fixed_basis'],arrays['residual_basis'];n,q=phi.shape
    ys=generated['phenotypes'];scenarios,n0,reps,two=ys.shape
    if (n0,reps,two)!=(n,args.replicates,2):raise ValueError('simulation phenotype dimensions differ')
    masks=(x,y);fixed=(np.linalg.qr(u[x])[0],np.linalg.qr(u[y])[0]);labels=np.unique(groups)
    weights=np.zeros((n,scenarios*reps*2*q));normalized=[];scales=np.empty((scenarios,reps,2))
    names=[];requested=[]
    for scenario in range(scenarios):
        for rep in range(reps):
            base=(scenario*reps+rep)*2
            requested.extend(((base,base),(base+1,base+1),(base,base+1)))
            for trait in range(2):
                idx=masks[trait];ut=fixed[trait];value=ys[scenario,idx,rep,trait]
                value=value-ut@(ut.T@value);scale=np.sqrt((len(idx)-ut.shape[1])/(value@value))
                value*=scale;scales[scenario,rep,trait]=scale;normalized.append(value)
                names.append(f'{scenario}:{rep}:{trait}');column=(base+trait)*q
                weights[idx,column:column+q]=phi[idx]*value[:,None]
    mask_batch=MaskedTraitBatch(basis=phi,fixed_basis=u,residual_basis=d,traits=[dict(name=str(t),indices=masks[t],
        fixed_basis=fixed[t],phenotype=normalized[t]) for t in range(2)])
    geometry=CrossTraitBatch(mask_batch,block_ids=labels,annotation_names=('all_variants',),pairs=[(0,0),(1,1),(0,1)])
    accumulator=CrossTraitScoreAccumulator(trait_names=names,num_basis=q,num_annotations=1,block_ids=labels,pairs=requested)
    overlap,lx,ly=np.intersect1d(x,y,return_indices=True)
    rrhs=[]
    for scenario in range(scenarios):
        for rep in range(reps):
            base=(scenario*reps+rep)*2;zx,zy=normalized[base:base+2]
            rrhs.extend((d[x].T@(zx*zx),d[y].T@(zy*zy),d[overlap].T@(zx[lx]*zy[ly])))
    operator=ProtectedTNOperator(threads=args.threads);operator.begin_execution();weights=np.asfortranarray(weights)
    telemetry=[];visits=0
    for begin,end,g in genotype_tiles(args,arrays,panel.m):
        scores=operator.matmul_tn(g,weights).reshape(end-begin,len(names),q)
        a=np.ones((end-begin,1));accumulator.add(scores,a,groups[begin:end])
        for _ in geometry.block(g.T,a,groups[begin:end]):pass
        telemetry.append(canonical_sha256(operator.finish_execution()));visits=end
        if (begin//args.width)%64==0 or end==panel.m:
            print(json.dumps(dict(phase='score',variants=end,total=panel.m,seconds=time.monotonic()-start)),flush=True)
    if visits!=panel.m:raise RuntimeError('simulation scoring traversal incomplete')
    provenance=base_provenance(args)
    provenance.update(generated_sha256=file_sha256(args.generated),genotype_traversals=1,variant_visits=visits,
        seconds=time.monotonic()-start,telemetry_sha256=telemetry,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_array_artifact(args.output/'scores.npz',kind='summit.cross_trait.simulation_scores',arrays=dict(
        block_rhs=accumulator.rhs,block_masses=accumulator.masses,block_ids=labels,
        block_genetic_residual=geometry.genetic_residual,residual_gram=geometry.residual_gram,
        residual_rhs=np.asarray(rrhs),scales=scales),provenance=provenance)


def fit(args):
    start=time.monotonic();reference,arrays,panel,groups,x,y=load_design(args)
    gen,gprov=load_array_artifact(args.generated,kind='summit.cross_trait.simulation')
    scores,sprov=load_array_artifact(args.scores,kind='summit.cross_trait.simulation_scores')
    if sprov['generated_sha256']!=file_sha256(args.generated):raise ValueError('simulation score identity differs')
    np.testing.assert_array_equal(scores['block_masses'],reference.block_annotation_mass)
    q=3;phi=arrays['basis'];masks=(x,y);types=((0,0),(1,1),(0,1))
    ref=SimpleNamespace(n_samples=len(phi),num_basis=q,annotation_names=('all_variants',),
        residual_rank=reference.residual_rank,block_masses=reference.block_annotation_mass,
        block_directed=reference.block_directed_numerator)
    grams=[chromosome_gram(ref,gen['reference_diagonal'],phi,masks[i],masks[j],global_masses=[panel.m],
                          mode=args.gram_mode) for i,j in types]
    estimate=[];deleted=[];raw_deleted=[]
    for t,(i,j) in enumerate(types):
        record=dict(block_ids=scores['block_ids'],block_masses=scores['block_masses'],
            block_rhs=scores['block_rhs'][:,t],
            block_genetic_residual=scores['block_genetic_residual'][:,t],gram=grams[t])
        plan=CrossTraitMomentPlan([record],residual_gram=scores['residual_gram'][t],
            residual_rhs=scores['residual_rhs'][t],num_basis=q,annotation_names=('all_variants',),
            deletion_method=args.deletion_method)
        batch_rhs=np.moveaxis(scores['block_rhs'][:,t::3],1,-1)
        result=fit_cross_trait_rhs_batch(plan,[batch_rhs],scores['residual_rhs'][t::3].T,
            restore_mass=args.deletion_method=='legacy' and not args.unrestored_deletions)
        scale=(scores['scales'][...,i]*scores['scales'][...,j]).reshape(-1)
        estimate.append(result['omega_xy'][:,0]/scale[:,None,None])
        deleted.append(result['loo_omega_xy'][:,:,0]/scale[:,None,None,None])
        raw_deleted.append(result['raw_loo_coefficients'][:,:,:q*q].reshape(-1,len(scores['block_ids']),q,q)
            /scale[:,None,None,None])
    estimate=np.stack(estimate,axis=1).reshape(2,args.replicates,3,q,q)
    deleted=np.stack(deleted,axis=1).reshape(2,args.replicates,3,len(scores['block_ids']),q,q)
    raw_deleted=np.stack(raw_deleted,axis=1).reshape(deleted.shape)
    s=np.cov(phi[:,1:].T,bias=True);means=(phi[x,1:].mean(0),phi[y,1:].mean(0))
    rows=[];replicate_rows=[]
    for sc in range(2):
        truth=gen['omega'][sc]
        kwargs=dict(mean_x=means[0],mean_y=means[1],context_covariance=s)
        actual=cross_trait_derived(estimate[sc,:,2],estimate[sc,:,0],estimate[sc,:,1],**kwargs)
        loo=cross_trait_derived(deleted[sc,:,2],deleted[sc,:,0],deleted[sc,:,1],**kwargs)
        target=cross_trait_derived(truth[:3,3:],truth[:3,:3],truth[3:,3:],**kwargs)
        actual['omega_xy']=estimate[sc,:,2];loo['omega_xy']=deleted[sc,:,2];target['omega_xy']=truth[:3,3:]
        uncertainty=[derived_uncertainty(estimate[sc,r,2],estimate[sc,r,0],estimate[sc,r,1],
            deleted[sc,r,2],deleted[sc,r,0],deleted[sc,r,1],method=args.uncertainty_method,**kwargs)
            for r in range(args.replicates)]
        for name in ('omega_xy','h_xy','baseline_rg','centered_baseline_rg','response_rg','orthogonal_rg','orthogonal_trace'):
            points=actual[name].reshape(args.replicates,-1);replicates=loo[name].reshape(args.replicates,len(scores['block_ids']),-1)
            truths=np.asarray(target[name]).ravel();jackknife_ses=np.sqrt((len(scores['block_ids'])-1)*np.var(replicates,axis=1))
            ses=(jackknife_ses if name=='omega_xy' else np.array([
                np.sqrt(np.maximum(0,np.diag(u[name+'_covariance']))) for u in uncertainty]))
            for j,value in enumerate(truths):
                valid=np.isfinite(points[:,j])&np.isfinite(ses[:,j]);errors=points[:,j]-value
                coverage=np.mean(np.abs(errors[valid])<=1.96*ses[valid,j]) if valid.any() else np.nan
                count=int(valid.sum());sd=np.std(points[valid,j],ddof=1) if count>1 else np.nan
                mean_se=np.mean(ses[valid,j]) if count else np.nan
                rows.append(dict(scenario=gprov['scenario_names'][sc],quantity=name,entry=j,truth=value,
                    valid_replicates=count,bias=np.mean(errors[valid]) if count else np.nan,
                    rmse=np.sqrt(np.mean(errors[valid]**2)) if count else np.nan,
                    empirical_sd=sd,mean_standard_error=mean_se,uncertainty_method=args.uncertainty_method,
                    rms_se_calibration=np.sqrt(np.mean(ses[valid,j]**2))/sd if sd>0 else np.nan,
                    mean_jackknife_se=np.mean(jackknife_ses[valid,j]) if count else np.nan,
                    se_calibration=mean_se/sd if sd>0 else np.nan,coverage_95=coverage,
                    coverage_acceptance=bool(valid.sum()==args.replicates and .90<=coverage<=.98)))
                for rep in range(args.replicates):
                    replicate_rows.append(dict(scenario=gprov['scenario_names'][sc],quantity=name,entry=j,replicate=rep,
                        truth=value,estimate=points[rep,j],standard_error=ses[rep,j],jackknife_se=jackknife_ses[rep,j]))
    for name,data in (('simulation_summary.tsv',rows),('simulation_replicates.tsv',replicate_rows)):
        with (args.output/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t');writer.writeheader();writer.writerows(data)
    provenance=dict(generated_sha256=file_sha256(args.generated),scores_sha256=file_sha256(args.scores),
        script_sha256=file_sha256(__file__),
        implementation_sha256={name:file_sha256(ROOT/'src/summit/context'/name) for name in
            ('cross_trait_fit.py','target_jackknife.py','cross_trait_uncertainty.py','cross_trait_gram.py')},
        deletion_method=args.deletion_method,uncertainty_method=args.uncertainty_method,
        deleted_genetic_mass_restored=args.deletion_method=='legacy' and not getattr(args,'unrestored_deletions',False),
        gram_mode=args.gram_mode,seconds=time.monotonic()-start,all_coverage_gates_pass=all(row['coverage_acceptance'] for row in rows),
        files={p.name:file_sha256(p) for p in args.output.glob('*.tsv')})
    write_array_artifact(args.output/'fits.npz',kind='summit.cross_trait.simulation_fit',
        arrays=dict(omega=estimate,loo_omega=deleted,raw_loo_omega=raw_deleted,truth=gen['omega'],
            block_ids=scores['block_ids'],loo_mass_restoration=result['loo_mass_restoration']),provenance=provenance)
    with (args.output/'COMPLETE.json').open('x') as f:json.dump(provenance,f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['reference','generate','score','fit'])
    for name in ('axes','bed-prefix','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--generated',type=Path);p.add_argument('--scores',type=Path)
    p.add_argument('--replicates',type=int,default=100);p.add_argument('--seed',type=int,default=2026092301)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--width',type=int,default=1024)
    p.add_argument('--probes',type=int,default=1024);p.add_argument('--blocks',type=int,default=200)
    p.add_argument('--probe-tile-width',type=int,default=4,
        help='Pass-2 probe tile width; changes tiling, not the probes or estimator')
    p.add_argument('--memory-gib',type=int,default=24)
    p.add_argument('--gram-mode',choices=['factorized','factorized_plus_residual','legacy_transport'],default='factorized')
    p.add_argument('--deletion-method',choices=['target_moments','legacy'],default='target_moments')
    p.add_argument('--uncertainty-method',choices=['delta','jackknife'],default='delta')
    p.add_argument('--unrestored-deletions',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    {'reference':reference_run,'generate':generate,'score':score,'fit':fit}[args.mode](args)


if __name__=='__main__':main()
