"""Restore study-collector deletion masses in saved within-trait fits, without solving.

Input artifacts remain unchanged. The exact raw deleted-system coefficients
are retained beside restored coefficients in a new authenticated output tree.
"""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from summit.context.annotations import _jackknife_covariance
from summit.context.cross_trait_fit import restore_deleted_genetic_mass
from summit.context.cross_trait_zpass import load_array_artifact,write_array_artifact
from summit.context.oracle import coefficients_to_omegas
from summit.context.reference_zpass_cli import file_sha256
from summit.context.spec import ContextComponentIndex,ContextPairIndex
from summit.ldscore.generalized_gxe_chromosome import load_chromosome_moments
from cross_trait_refit import authenticate,quantities,write_table


def restore(source,reference,diagnostics,output):
    receipt=json.loads((source/'COMPLETE.json').read_text())
    if receipt['failures']:raise ValueError('source refits contain failures')
    for name,digest in receipt['tables'].items():
        if file_sha256(source/name)!=digest:raise ValueError(f'source checksum differs: {name}')
    manifest=file_sha256(reference/'MANIFEST.json')
    if receipt['manifest_sha256'][str(reference)]!=manifest:raise ValueError('reference identity differs')
    diagnostic=diagnostics/'reference_diagnostics.npz'
    _,meta=load_array_artifact(diagnostic,kind='summit.cross_trait.reference_diagnostics')
    for key in ('manifest_sha256','source_completion_sha256'):
        if meta[key]!=receipt[key]:raise ValueError('reference diagnostic identities differ')
    chunks=[]
    for ch in range(1,23):
        folder=reference/f'chr{ch}';authenticate(folder,'REFERENCE_COMPLETE.json',manifest)
        chunk,_=load_chromosome_moments(folder/'reference.npz');chunks.append(chunk)
    blocks=np.unique(np.concatenate([c.block_ids for c in chunks]));k=len(chunks[0].annotation_names)
    group_mass=np.zeros((len(blocks),k))
    for chunk in chunks:np.add.at(group_mass,np.searchsorted(blocks,chunk.block_ids),chunk.block_masses)
    mass=group_mass.sum(0);retained=mass-group_mass;q=chunks[0].num_basis
    components=ContextComponentIndex(chunks[0].annotation_names,ContextPairIndex(q))
    output.mkdir(exist_ok=False)
    read=lambda name:list(csv.DictReader((source/name).open(),delimiter='\t'))
    old_table=read('within_trait_mode_comparison.tsv');table=[];fit_hashes={}
    correction=dict(source_refit_completion_sha256=file_sha256(source/'COMPLETE.json'),
        script_sha256=file_sha256(__file__),commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        implementation_sha256=file_sha256(ROOT/'src/summit/context/cross_trait_fit.py'),
        deleted_genetic_mass_restored=True,genotype_traversals=0,normal_equation_solves=0)
    for trait in receipt['traits']:
        old_rows=[r for r in old_table if r['trait']==trait];results={}
        for arm in sorted({r['arm'] for r in old_rows}):
            name=f'{trait}__{arm}.npz';arrays,provenance=load_array_artifact(
                source/name,kind='summit.cross_trait.within_refit')
            np.testing.assert_array_equal(arrays['block_ids'],blocks)
            if bool(arrays.get('loo_genetic_mass_restored',False)):
                raise ValueError('input is already restored')
            raw=arrays['loo_coefficients'];loo=restore_deleted_genetic_mass(raw,mass,retained,len(ContextPairIndex(q)))
            omega=arrays['omega'];loomega=np.array([
                coefficients_to_omegas(row[:len(components)],components) for row in loo])
            # Compare independently with the original study collector ordering.
            expected=arrays['loo_omega']*(mass/retained)[:,:,None,None]
            np.testing.assert_array_equal(loomega,expected)
            mean,s=arrays['environment_mean'],arrays['environment_covariance']
            values=quantities(omega,mean,s);deleted=quantities(loomega,mean,s)
            results[arm]=(values,{name:np.sqrt((len(blocks)-1)*np.var(value,axis=0))
                                 for name,value in deleted.items()})
            arrays.update(raw_loo_coefficients=raw,loo_coefficients=loo,loo_omega=loomega,
                covariance=_jackknife_covariance(loo),loo_annotation_masses=retained,
                loo_mass_restoration=mass/retained,loo_genetic_mass_restored=np.array(True))
            write_array_artifact(output/name,kind='summit.cross_trait.within_refit',arrays=arrays,
                provenance=dict({**provenance,**correction},mass_restoration_correction=correction,
                    reference_diagnostics=dict(file=str(diagnostic),sha256=file_sha256(diagnostic)),
                    original_fit_script_sha256=provenance['script_sha256'],
                    raw_source_fit_sha256=file_sha256(source/name)))
            fit_hashes[name]=file_sha256(output/name)
        default=results['factorized__own_rows']
        for old in old_rows:
            row=dict(old);arm=row['arm'];quantity=row['quantity'];j=chunks[0].annotation_names.index(row['annotation'])
            value=float(results[arm][0][quantity][j]);se=float(results[arm][1][quantity][j])
            np.testing.assert_allclose(value,float(row['estimate']),rtol=0,atol=0,equal_nan=True)
            ds=float(default[1][quantity][j]);shift=(value-default[0][quantity][j])/ds if ds>0 else np.nan
            row.update(jackknife_se=se,shift_from_default_se=shift,flag_shift_gt_one_se=bool(abs(shift)>1))
            table.append(row)
    write_table(output/'within_trait_mode_comparison.tsv',table)
    diagnostics=read('within_trait_fit_diagnostics.tsv')
    for row in diagnostics:row['correction_only_no_refit']=True
    write_table(output/'within_trait_fit_diagnostics.tsv',diagnostics)
    result=dict(correction,traits=receipt['traits'],failures=[],source=str(source),
        tables={str(p):file_sha256(p) for p in output.glob('*.tsv')},fit_sha256=fit_hashes,
        manifest_sha256=receipt['manifest_sha256'],source_completion_sha256=receipt['source_completion_sha256'],
        reference_diagnostics_sha256=file_sha256(diagnostic),
        flagged_rows=sum(r['flag_shift_gt_one_se'] for r in table),
        flagged_traits=sorted({r['trait'] for r in table if r['flag_shift_gt_one_se']}))
    with (output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key!='fit_sha256'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','reference','diagnostics','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();restore(a.source,a.reference,a.diagnostics,a.output)


if __name__=='__main__':main()
