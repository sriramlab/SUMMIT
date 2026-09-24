"""Refit sealed within-trait summaries under six prespecified transport arms.

No genotypes are opened. Every reference and summary completion record and
array checksum is checked. Output paths are exclusive and legacy artifacts
remain unchanged. The cached Grams are reused for all paired SNP deletions.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from summit.context.cross_trait_gram import chromosome_gram,within_trait_equations
from summit.context.fit import solve_context_normal_equations
from summit.context.cross_trait_fit import restore_deleted_genetic_mass
from summit.context.annotations import _jackknife_covariance
from summit.context.oracle import coefficients_to_omegas
from summit.context.spec import ContextComponentIndex,ContextPairIndex
from summit.context.reference_zpass_cli import file_sha256
from summit.context.cross_trait_zpass import write_array_artifact
from summit.ldscore.generalized_gxe_chromosome import load_chromosome_moments,combine_chromosome_annotations


def authenticate(root,marker,manifest_hash):
    record=json.loads((root/marker).read_text())
    if not record['passed'] or record['manifest_sha256']!=manifest_hash:
        raise ValueError(f'completion identity differs: {root/marker}')
    for name,digest in record['files'].items():
        if file_sha256(root/name)!=digest:
            raise ValueError(f'checksum differs: {root/name}')
    return record


def quantities(omega,mean,s):
    q=omega.shape[-1];c=np.eye(q);c[0,1:]=mean
    centered=c@omega@c.T
    with np.errstate(divide='ignore',invalid='ignore'):
        h=centered[...,1:,1:]-centered[...,1:,0,None]*centered[...,None,0,1:]/centered[...,0,0,None,None]
    output={}
    for pair in ContextPairIndex(q).entries:
        output[f'omega:{pair.q},{pair.r}']=omega[...,pair.q,pair.r]
    output['centered_baseline']=centered[...,0,0]
    for a in range(q-1):
        for b in range(a,q-1):
            output[f'h:{a+1},{b+1}']=h[...,a,b]
            if a!=b:
                with np.errstate(invalid='ignore',divide='ignore'):
                    output[f'orthogonal_response_correlation:{a+1},{b+1}']=np.where(
                        (h[...,a,a]>0)&(h[...,b,b]>0),h[...,a,b]/np.sqrt(h[...,a,a]*h[...,b,b]),np.nan)
    output['response_trace']=np.einsum('ij,...ji->...',s,centered[...,1:,1:])
    output['orthogonal_response_trace']=np.einsum('ij,...ji->...',s,h)
    return output


def write_table(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--traits',nargs='*')
    p.add_argument('--unrestored-deletions',action='store_true',
        help='reproduce raw deleted-system coefficients before study-collector mass restoration')
    p.add_argument('--diagnostics-only',action='store_true',help='publish shared per-block reference diagnostics without fitting')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    started=time.monotonic();refroot=a.base/'shared_reference_full_20260916'
    expansion=a.base/'imputed_expansion_20260917';sources=(refroot,expansion)
    manifests={root:json.loads((root/'MANIFEST.json').read_text()) for root in sources}
    hashes={root:file_sha256(root/'MANIFEST.json') for root in sources}
    trait_root={name:root for root in sources for name in manifests[root]['traits']}
    # The expansion's height is a cohort anchor; use its original completed summary.
    trait_root['height_raw']=refroot
    selected=sorted(trait_root) if not a.traits else a.traits
    with np.load(a.base/'full_cohort_inputs_20260916/height_raw.npz') as z:
        master_rows,phi=z['rows'],z['phi']
    ref=[];diagonals={};refcommon=None;diagonal_sum=None
    completion_hashes={}
    for ch in range(1,23):
        root=refroot/f'chr{ch}'
        authenticate(root,'REFERENCE_COMPLETE.json',hashes[refroot])
        completion_hashes[str(root/'REFERENCE_COMPLETE.json')]=file_sha256(root/'REFERENCE_COMPLETE.json')
        chunk,_=load_chromosome_moments(root/'reference.npz');ref.append(chunk)
        with np.load(root/'reference_aux.npz') as z:
            d=z['global_kernel_diagonal'];diagonals[str(ch)]=d
            diagonal_sum=d.copy() if diagonal_sum is None else diagonal_sum+d
            current={name:z[name] for name in ('residual_gram','residual_traces','residual_names')}
        if refcommon is not None:
            for name in current:np.testing.assert_array_equal(current[name],refcommon[name])
        refcommon=current
        for source in sources:
            authenticate(source/f'chr{ch}','STUDY_COMPLETE.json',hashes[source])
            marker=source/f'chr{ch}/STUDY_COMPLETE.json';completion_hashes[str(marker)]=file_sha256(marker)
    same=diagonal_sum@diagonal_sum.T
    masses=sum(c.block_masses.sum(0) for c in ref);q=ref[0].num_basis
    components=ContextComponentIndex(ref[0].annotation_names,ContextPairIndex(q))
    blocks=np.unique(np.concatenate([c.block_ids for c in ref]))
    if len(blocks)!=200:
        raise ValueError('paper refit requires all 200 target blocks')
    options=dict(reference_residual_gram=refcommon['residual_gram'],reference_residual_traces=refcommon['residual_traces'],
        reference_same_person=same,residual_names=tuple(refcommon['residual_names'].astype(str)),expected_chromosomes=range(1,23))
    modes=[(mode,sp) for mode in ('legacy_transport','factorized','factorized_plus_residual') for sp in ('scaled','own_rows')]
    table=[];diagnostics=[];failures=[]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=dict(branch='feat/cross-trait-response-covariance',commit=commit,script_sha256=file_sha256(__file__),
        deleted_genetic_mass_restored=not a.unrestored_deletions,
        source_completion_sha256=completion_hashes,manifest_sha256={str(k):v for k,v in hashes.items()},
        uncertainty='paired 200-block frozen-source mass-restored deletion; frozen same-person',genotype_traversals=0)
    # These reference diagnostics do not depend on the trait mask or fit mode.
    # Publish once and authenticate the common file in every fit's provenance.
    reference_diagnostics={key:[] for key in ('chromosome','block_id','factorization_residual','same_person_share','ld_scalar')}
    for c in ref:
        g=chromosome_gram(c,diagonals[c.chromosome],phi,np.arange(len(phi)),global_masses=masses,ordered=False)
        reference_diagnostics['chromosome'].extend([c.chromosome]*len(c.block_ids))
        reference_diagnostics['block_id'].extend(c.block_ids)
        for key in ('factorization_residual','same_person_share','ld_scalar'):
            reference_diagnostics[key].extend(g.diagnostics[key])
    diagnostic_path=a.output/'reference_diagnostics.npz'
    write_array_artifact(diagnostic_path,kind='summit.cross_trait.reference_diagnostics',
        arrays={key:np.asarray(value) for key,value in reference_diagnostics.items()},provenance=provenance)
    provenance['reference_diagnostics']=dict(file=diagnostic_path.name,sha256=file_sha256(diagnostic_path))
    if a.diagnostics_only:selected=[]
    for name in selected:
        root=trait_root[name];inputroot=(a.base/'full_cohort_inputs_20260916' if root==refroot else expansion/'inputs')
        inputpath=inputroot/f'{name}.npz'
        if file_sha256(inputpath)!=manifests[root]['trait_sources'][name]:
            raise ValueError(f'trait input checksum differs: {inputpath}')
        with np.load(inputpath) as z:rows=np.searchsorted(master_rows,z['rows']);np.testing.assert_array_equal(master_rows[rows],z['rows'])
        mean=phi[rows,1:].mean(0);s=np.cov(phi[rows,1:].T,bias=True)
        own_same=diagonal_sum[:,rows]@diagonal_sum[:,rows].T
        study=[];common=None
        for ch in range(1,23):
            chunk,_=load_chromosome_moments(root/f'chr{ch}/{name}.npz');study.append(chunk)
            with np.load(root/f'chr{ch}/{name}_common.npz') as z:
                current={key:z[key] for key in ('residual_gram','residual_rhs','residual_traces')}
            if common is not None:
                for key in current:np.testing.assert_allclose(current[key],common[key],rtol=1e-13,atol=1e-10)
            common=current
        results={}
        for mode,sp in modes:
            key=f'{mode}__{sp}';tick=time.monotonic()
            try:
                prepared=None if (mode,sp)==('legacy_transport','scaled') else {
                    c.chromosome:chromosome_gram(c,diagonals[c.chromosome],phi,rows,global_masses=masses,
                        mode=mode,same_person_mode=sp,ordered=False) for c in ref}
                def eq(deleted=()):
                    return within_trait_equations(ref,study,reference_diagonals=None,phi=phi,rows=rows,
                        mode=mode,same_person_mode=sp,prepared_grams=prepared,deleted_blocks=deleted,
                        full_same_person=own_same if sp=='own_rows' else len(rows)/ref[0].n_samples*same,**options,**common)
                full=eq();fit=solve_context_normal_equations(full)
                deleted_equations=[eq((b,)) for b in blocks]
                raw_loo=np.array([solve_context_normal_equations(e).coefficients for e in deleted_equations])
                retained=np.array([e.annotation_masses for e in deleted_equations])
                loo=(raw_loo.copy() if a.unrestored_deletions else
                     restore_deleted_genetic_mass(raw_loo,masses,retained,len(ContextPairIndex(q))))
                omega=coefficients_to_omegas(fit.coefficients[:len(components)],components)
                loomega=np.array([coefficients_to_omegas(row[:len(components)],components) for row in loo])
                values=quantities(omega,mean,s);deleted=quantities(loomega,mean,s)
                results[key]=(values,deleted,full.matrix)
                arrays=dict(omega=omega,loo_omega=loomega,coefficients=fit.coefficients,loo_coefficients=loo,
                    covariance=_jackknife_covariance(loo),normal_matrix=full.matrix,normal_rhs=full.rhs,
                    raw_loo_coefficients=raw_loo,loo_annotation_masses=retained,
                    loo_mass_restoration=masses[None]/retained,
                    loo_genetic_mass_restored=np.array(not a.unrestored_deletions),
                    same_person_gram=own_same if sp=='own_rows' else len(rows)/ref[0].n_samples*same,
                    block_ids=blocks,environment_mean=mean,environment_covariance=s)
                write_array_artifact(a.output/f'{name}__{key}.npz',kind='summit.cross_trait.within_refit',arrays=arrays,
                    provenance=dict(provenance,trait=name,mode=mode,same_person_mode=sp,input_sha256=file_sha256(inputpath),
                        same_person_assembly='sum_chromosome_diagonals_before_Gram; frozen full-profile deletion adjustment'))
                diagnostics.append(dict(trait=name,mode=mode,same_person_mode=sp,rank=fit.rank,
                    condition_number=fit.condition_number,relative_residual=fit.relative_residual,
                    minimum_eigenvalue=fit.minimum_gram_eigenvalue,seconds=time.monotonic()-tick))
                print(json.dumps(diagnostics[-1]),flush=True)
            except (ValueError,np.linalg.LinAlgError) as exc:
                failures.append(dict(trait=name,mode=mode,same_person_mode=sp,error=str(exc)))
                print(json.dumps(failures[-1]),flush=True)
        legacy=results.get('legacy_transport__scaled');default=results.get('factorized__own_rows')
        for key,(point,deleted,matrix) in results.items():
            for quantity,value in point.items():
                se=np.sqrt((len(blocks)-1)*np.var(deleted[quantity],axis=0))
                legacy_value=legacy[0][quantity] if legacy else np.full_like(value,np.nan)
                default_value=default[0][quantity] if default else np.full_like(value,np.nan)
                default_se=np.sqrt((len(blocks)-1)*np.var(default[1][quantity],axis=0)) if default else se*np.nan
                for k,annotation in enumerate(ref[0].annotation_names):
                    shift=(value[k]-default_value[k])/default_se[k] if default_se[k]>0 else np.nan
                    table.append(dict(trait=name,n=len(rows),annotation=annotation,quantity=quantity,arm=key,
                        estimate=value[k],jackknife_se=se[k],shift_from_legacy=value[k]-legacy_value[k],
                        shift_from_default_se=shift,flag_shift_gt_one_se=bool(abs(shift)>1),
                        normal_matrix_difference_2norm=float(np.linalg.norm(matrix-default[2],2)) if default else np.nan))
    if table:write_table(a.output/'within_trait_mode_comparison.tsv',table)
    if diagnostics:write_table(a.output/'within_trait_fit_diagnostics.tsv',diagnostics)
    result=dict(provenance,traits=selected,failures=failures,seconds=time.monotonic()-started,
        tables={str(path):file_sha256(path) for path in a.output.glob('*.tsv')})
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
