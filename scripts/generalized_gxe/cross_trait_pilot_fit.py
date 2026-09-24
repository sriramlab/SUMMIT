"""Fit the prespecified common-bin eight-trait pilot without genotypes.

Cross summaries come from one shared traversal per chromosome. Within-trait
scores reuse authenticated saved study summaries, reduced to the same common
annotation and fitted through the same ordered normal equations. All four
arms share paired target deletions and frozen overlap-person diagonals.
"""
from pathlib import Path
import argparse
import itertools
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from summit.context.cross_trait_gram import chromosome_gram,orientation_matrix,same_person
from summit.context.cross_trait_fit import CrossTraitMomentPlan,fit_cross_trait,write_cross_trait_fit,cross_trait_derived
from summit.context.cross_trait_zpass import load_array_artifact,repair_single_annotation
from summit.context.reference_zpass_cli import file_sha256,authenticated_reference
from summit.ldscore.generalized_gxe_chromosome import load_chromosome_moments,combine_chromosome_annotations
from cross_trait_refit import authenticate,write_table

TRAITS=('ldl_raw','apo_b_raw','cholesterol_raw','non_hdl_cholesterol_raw','hba1c_raw',
        'diastolic_blood_pressure_raw','height_raw','platelet_count_raw')
MODES=('factorized','factorized_plus_residual','legacy_transport','legacy_transport_exact')


def ordered_within_record(study):
    """Expand symmetric score/R moments; each off-diagonal was stored twice."""
    q=study.num_basis;b=orientation_matrix(q);expand=b.T/np.sum(b*b,axis=1)
    return dict(block_ids=study.block_ids,block_masses=study.block_masses,
        block_rhs=(study.block_genetic_rhs[...,0]@expand.T).reshape(-1,1,q,q),
        block_genetic_residual=(expand@study.block_genetic_residual)[:,None])


def fit_pilot(args):
    start=time.monotonic();base=args.base;refroot=base/'shared_reference_full_20260916'
    master=base/'full_cohort_inputs_20260916/height_raw.npz'
    with np.load(master,allow_pickle=False) as z:master_rows,phi,basis_names=z['rows'],z['phi'],z['basis_names']
    n,q=phi.shape;p=q*(q+1)//2;s=np.cov(phi[:,1:].T,bias=True)
    expanded=base/'imputed_expansion_20260917'
    manifests={r:json.loads((r/'MANIFEST.json').read_text()) for r in (refroot,expanded)}
    roots={t:r for r in manifests for t in manifests[r]['traits']};roots['height_raw']=refroot
    rows={};input_hashes={};completion_hashes={}
    for name in TRAITS:
        r=roots[name];ip=(base/'full_cohort_inputs_20260916' if r==refroot else expanded/'inputs')/f'{name}.npz'
        digest=file_sha256(ip)
        if digest!=manifests[r]['trait_sources'][name]:raise ValueError(f'trait identity differs: {ip}')
        input_hashes[str(ip)]=digest
        with np.load(ip,allow_pickle=False) as z:
            rows[name]=np.searchsorted(master_rows,z['rows']);np.testing.assert_array_equal(master_rows[rows[name]],z['rows'])
    refs=[];diagonals=[];cross=[];repaired=[];within={t:[] for t in TRAITS};commons={}
    select=np.array([[0.],[0.],[1.]])
    # Read masses from validated artifacts; no manifest key guessing.
    for ch in args.chromosomes:
        manifest,panel,r,record=authenticated_reference(refroot/'MANIFEST.json',refroot,ch,master)
        chunk,_=load_chromosome_moments(r/'reference.npz')
        common_name=(chunk.annotation_names[2],)
        refs.append(combine_chromosome_annotations(chunk,select,common_name))
        with np.load(r/'reference_aux.npz',allow_pickle=False) as z:diagonals.append(z['global_kernel_diagonal'][2*p:3*p])
        cp=args.study_root/f'chr{ch}/cross_trait_summary.npz'
        data,provenance=load_array_artifact(cp,kind='summit.cross_trait.summary')
        if (provenance['reference_files']!=record['files'] or not provenance['common_only']
                or provenance['genotype_traversals']!=1 or provenance['variant_visits']!=panel['m']
                or tuple(data['trait_names'].astype(str))!=TRAITS):raise ValueError(f'cross-study design differs: {cp}')
        identities=lambda values:{Path(k).name:v for k,v in values.items()}
        if identities(provenance['input_sha256'])!=identities(input_hashes):
            raise ValueError('cross study input identities differ')
        np.testing.assert_array_equal(data['block_ids'],refs[-1].block_ids)
        np.testing.assert_array_equal(data['block_masses'],refs[-1].block_masses)
        cross.append(data);completion_hashes[str(cp)]=file_sha256(cp)
        if 'legacy_transport_exact' in args.modes:
            zp=args.z_root/f'zpass_chr{ch}.npz';z,zm=load_array_artifact(zp,kind='summit.cross_trait.z_moments')
            if zm['reference_files']!=record['files'] or zm['master_input_sha256']!=file_sha256(master):
                raise ValueError(f'Z pass identity differs: {zp}')
            if zm['execution_ledger']['retained_variant_visits']!=panel['m']:
                raise ValueError('Z pass chromosome is incomplete')
            np.testing.assert_array_equal(z['block_ids'],refs[-1].block_ids)
            repaired.append(z);completion_hashes[str(zp)]=file_sha256(zp)
        for r0 in {roots[t] for t in TRAITS}:
            authenticate(r0/f'chr{ch}','STUDY_COMPLETE.json',file_sha256(r0/'MANIFEST.json'))
            marker=r0/f'chr{ch}/STUDY_COMPLETE.json';completion_hashes[str(marker)]=file_sha256(marker)
        for name in TRAITS:
            r0=roots[name]/f'chr{ch}';study,_=load_chromosome_moments(r0/f'{name}.npz')
            study=combine_chromosome_annotations(study,select,common_name)
            within[name].append(ordered_within_record(study))
            with np.load(r0/f'{name}_common.npz',allow_pickle=False) as z:
                current={key:z[key] for key in ('residual_gram','residual_rhs')}
            if name in commons:
                for key in current:np.testing.assert_allclose(current[key],commons[name][key],rtol=1e-13,atol=1e-10)
            commons[name]=current
    masses=sum(ref.block_masses.sum(0) for ref in refs)
    # Stored diagonals use genome-wide masses even for a chromosome timing fit.
    global_masses=np.asarray(manifests[refroot]['global_masses'],dtype=float)[2:3]
    for i in range(len(diagonals)):diagonals[i]=diagonals[i]*(global_masses[0]/masses[0])
    diagonal_total=sum(diagonals)
    pair_order=[(i,i) for i in range(len(TRAITS))]+list(itertools.combinations(range(len(TRAITS)),2))
    full_person={(i,j):same_person(diagonal_total,rows[TRAITS[i]],rows[TRAITS[j]],q=q)
                 for i,j in pair_order}
    repaired_blocks=[]
    if repaired:
        for ref,z in zip(refs,repaired):
            tr=ref.block_directed*ref.residual_rank**2/masses[0]**2
            repaired_blocks.append(repair_single_annotation(tr,z['block_products'][:,2],z['global_products'][2],mass=masses[0]))
    modes={};table=[];gram_rows=[]
    provenance=dict(input_sha256=input_hashes,source_sha256=completion_hashes,script_sha256=file_sha256(__file__),
        implementation_sha256={name:file_sha256(ROOT/'src/summit/context'/name) for name in
            ('cross_trait_fit.py','cross_trait_gram.py','cross_trait_zpass.py')},
        chromosomes=list(args.chromosomes),annotation=refs[0].annotation_names[0],
        context_metric='master cohort population covariance',genotype_traversals=0,
        same_person_assembly='sum_chromosome_diagonals_before_Gram; frozen full-profile deletion adjustment',
        interpretation='exploratory common-bin-only model; omitted lower-frequency effects may confound estimates',
        prespecified_hypothesis='Shared age-BMI response direction in lipid-glycaemic-BP cluster; absent in height and platelets')
    for mode in args.modes:
        fits={};matrices={};diagnostics={}
        pairs=pair_order
        for ix,iy in pairs:
            x,y=TRAITS[ix],TRAITS[iy];records=[];factorization=[];shares=[]
            for i,ref in enumerate(refs):
                gram=chromosome_gram(ref,diagonals[i],phi,rows[x],rows[y],global_masses=masses,
                    mode=mode,repaired_blocks=repaired_blocks[i] if repaired_blocks else None)
                factorization.append(gram.diagnostics['factorization_residual']);shares.append(gram.diagnostics['same_person_share'])
                if ix==iy:
                    record=dict(within[x][i]);rr=commons[x]['residual_gram'];rrhs=commons[x]['residual_rhs'].ravel()
                else:
                    data=cross[i];found=np.flatnonzero(np.all(data['pairs']==[ix,iy],axis=1))
                    if len(found)!=1:raise ValueError('missing or duplicated trait pair')
                    j=found[0]
                    record=dict(block_ids=data['block_ids'],block_masses=data['block_masses'],
                        block_rhs=data['block_rhs'][:,j],block_genetic_residual=data['block_genetic_residual'][:,j])
                    rr=data['residual_gram'][j];rrhs=data['residual_rhs'][j]
                if records:
                    np.testing.assert_allclose(rr,previous_rr,rtol=1e-13,atol=1e-10)
                    np.testing.assert_allclose(rrhs,previous_rhs,rtol=1e-13,atol=1e-10)
                previous_rr,previous_rhs=rr,rrhs;record['gram']=gram;records.append(record)
            plan=CrossTraitMomentPlan(records,residual_gram=rr,residual_rhs=rrhs,num_basis=q,
                annotation_names=refs[0].annotation_names,reference_n=n,n_x=len(rows[x]),n_y=len(rows[y]),
                full_same_person=full_person[ix,iy])
            fit=fit_cross_trait(plan);fits[ix,iy]=fit;matrices[ix,iy]=plan.equations().equations.matrix
            fit.update(factorization_residual=np.concatenate(factorization),same_person_share=np.concatenate(shares),
                basis_names=basis_names,annotation_names=np.array(refs[0].annotation_names))
            diagnostics[ix,iy]=dict(rank=int(fit['rank']),condition=float(fit['condition_number']))
        for ix,iy in pairs:
            if ix==iy:continue
            x,y=TRAITS[ix],TRAITS[iy];fit=fits[ix,iy]
            cohort=dict(n_x=len(rows[x]),n_y=len(rows[y]),
                n_overlap=len(np.intersect1d(rows[x],rows[y],assume_unique=True)))
            wx=dict(omega=fits[ix,ix]['omega_xy'],loo=fits[ix,ix]['loo_omega_xy'],block_ids=fit['block_ids'])
            wy=dict(omega=fits[iy,iy]['omega_xy'],loo=fits[iy,iy]['loo_omega_xy'],block_ids=fit['block_ids'])
            kwargs=dict(mean_x=phi[rows[x],1:].mean(0),mean_y=phi[rows[y],1:].mean(0),context_covariance=s)
            write_cross_trait_fit(args.output/f'{x}__{y}__{mode}.npz',fit,
                provenance=dict(provenance,gram_mode=mode,trait_x=x,trait_y=y,
                    reference_n=n,**cohort),within_x=wx,within_y=wy,**kwargs)
            point=cross_trait_derived(fit['omega_xy'],wx['omega'],wy['omega'],**kwargs)
            loo=cross_trait_derived(fit['loo_omega_xy'],wx['loo'],wy['loo'],**kwargs)
            point['omega_xy']=fit['omega_xy'];loo['omega_xy']=fit['loo_omega_xy']
            for quantity in ('omega_xy','h_xy','response_rg','baseline_rg','orthogonal_rg','orthogonal_trace',
                             'orthogonal_minus_baseline_rg','response_minus_baseline_rg'):
                se=np.sqrt((len(fit['block_ids'])-1)*np.var(loo[quantity],axis=0)).ravel()
                for j,value in enumerate(point[quantity].ravel()):
                    if quantity=='omega_xy':ex,ey=basis_names[j//q],basis_names[j%q]
                    elif quantity=='h_xy':ex,ey=basis_names[1+j//(q-1)],basis_names[1+j%(q-1)]
                    elif quantity in ('response_rg','response_minus_baseline_rg'):ex=ey=basis_names[j+1]
                    else:ex=ey='intercept' if quantity=='baseline_rg' else 'context_weighted'
                    table.append(dict(trait_x=x,trait_y=y,**cohort,mode=mode,quantity=quantity,entry=j,
                        exposure_x=str(ex),exposure_y=str(ey),estimate=value,jackknife_se=se[j],
                        lower_95=value-1.96*se[j],upper_95=value+1.96*se[j]))
            print(json.dumps(dict(mode=mode,pair=[x,y],**diagnostics[ix,iy])),flush=True)
        modes[mode]=matrices
    default={(r['trait_x'],r['trait_y'],r['quantity'],r['entry']):r for r in table if r['mode']=='factorized'}
    for row in table:
        d=default[row['trait_x'],row['trait_y'],row['quantity'],row['entry']]
        row['shift_from_default_se']=(row['estimate']-d['estimate'])/d['jackknife_se'] if d['jackknife_se']>0 else np.nan
    for ix,iy in itertools.combinations(range(len(TRAITS)),2):
        for mode in args.modes:
            gram_rows.append(dict(trait_x=TRAITS[ix],trait_y=TRAITS[iy],mode=mode,
                normal_matrix_difference_2norm=np.linalg.norm(modes[mode][ix,iy]-modes['factorized'][ix,iy],2)))
    write_table(args.output/'pilot_estimates.tsv',table);write_table(args.output/'pilot_gram_mode_differences.tsv',gram_rows)
    shifts=[]
    for key,d in default.items():
        values=[abs(r['shift_from_default_se']) for r in table
                if (r['trait_x'],r['trait_y'],r['quantity'],r['entry'])==key and np.isfinite(r['shift_from_default_se'])]
        undefined=[r['mode'] for r in table if (r['trait_x'],r['trait_y'],r['quantity'],r['entry'])==key
                   and not np.isfinite(r['shift_from_default_se'])]
        shifts.append(dict(trait_x=key[0],trait_y=key[1],quantity=key[2],entry=key[3],
            maximum_mode_shift_se=max(values) if values else np.nan,valid_modes=len(values),
            undefined_shift_modes=','.join(undefined)))
    write_table(args.output/'pilot_maximum_mode_shifts.tsv',shifts)
    with (args.output/'COMPLETE.json').open('x') as f:
        json.dump(dict(provenance,seconds=time.monotonic()-start,tables={p.name:file_sha256(p) for p in args.output.glob('*.tsv')}),f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','study-root','z-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chromosomes',type=int,nargs='+',default=list(range(1,23)))
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=p.parse_args()
    if 'factorized' not in args.modes:p.error('factorized default is required for comparisons')
    args.output.mkdir(exist_ok=False);fit_pilot(args)


if __name__=='__main__':main()
