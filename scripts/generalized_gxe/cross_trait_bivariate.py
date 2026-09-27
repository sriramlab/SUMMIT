"""Compare the pilot with ordinary bivariate SUMMIT without another genotype pass.

The ordinary model has only a baseline genetic kernel. The context model's
master and centered baseline entries need not estimate the same marginal
quantity under GxE. Report discrepancies, never force their agreement.
"""
from pathlib import Path
import argparse
import itertools
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from summit.context.cross_trait_bivariate import BivariateBlockAdapter,baseline_reference_block_ld
from summit.context.cross_trait_zpass import load_array_artifact,write_array_artifact
from summit.context.reference_zpass_cli import authenticated_reference,file_sha256
from summit.ldscore.generalized_gxe_chromosome import load_chromosome_moments,combine_chromosome_annotations
from cross_trait_pilot_fit import TRAITS,MODES
from cross_trait_refit import authenticate,write_table


def run(args):
    base=args.base;refroot=base/'shared_reference_full_20260916'
    expanded=base/'imputed_expansion_20260917';master=base/'full_cohort_inputs_20260916/height_raw.npz'
    manifests={r:json.loads((r/'MANIFEST.json').read_text()) for r in (refroot,expanded)}
    roots={t:r for r in manifests for t in manifests[r]['traits']};roots['height_raw']=refroot
    input_hashes={}
    for t in TRAITS:
        ip=(base/'full_cohort_inputs_20260916' if roots[t]==refroot else expanded/'inputs')/f'{t}.npz'
        input_hashes[ip.name]=file_sha256(ip)
        if input_hashes[ip.name]!=manifests[roots[t]]['trait_sources'][t]:raise ValueError('trait input identity differs')
    refs=[];studies={t:[] for t in TRAITS};cross=[];hashes={}
    select=np.array([[0.],[0.],[1.]])
    for ch in args.chromosomes:
        _,panel,r,record=authenticated_reference(refroot/'MANIFEST.json',refroot,ch,master)
        ref,_=load_chromosome_moments(r/'reference.npz');names=(ref.annotation_names[2],)
        refs.append(combine_chromosome_annotations(ref,select,names))
        hashes[str(r/'reference.npz')]=record['files']['reference.npz']
        for root in {roots[t] for t in TRAITS}:
            authenticate(root/f'chr{ch}','STUDY_COMPLETE.json',file_sha256(root/'MANIFEST.json'))
            marker=root/f'chr{ch}/STUDY_COMPLETE.json';hashes[str(marker)]=file_sha256(marker)
        for t in TRAITS:
            chunk,_=load_chromosome_moments(roots[t]/f'chr{ch}/{t}.npz')
            studies[t].append(combine_chromosome_annotations(chunk,select,names))
        cp=args.study_root/f'chr{ch}/cross_trait_summary.npz'
        d,p=load_array_artifact(cp,kind='summit.cross_trait.summary')
        if (p['reference_files']!=record['files'] or not p['common_only']
                or p['genotype_traversals']!=1 or p['variant_visits']!=panel['m']
                or tuple(d['trait_names'].astype(str))!=TRAITS
                or {Path(k).name:v for k,v in p['input_sha256'].items()}!=input_hashes):
            raise ValueError('cross summary identity differs')
        np.testing.assert_array_equal(d['block_ids'],refs[-1].block_ids)
        np.testing.assert_array_equal(d['block_masses'],refs[-1].block_masses)
        cross.append(d);hashes[str(cp)]=file_sha256(cp)
    ids,counts,ld=baseline_reference_block_ld(refs)
    adapter=BivariateBlockAdapter(ids,counts,ld);within={}
    for t,chunks in studies.items():
        sums=np.zeros(len(ids));first=chunks[0]
        for c in chunks:
            if (c.n_samples,c.residual_rank)!=(first.n_samples,first.residual_rank):raise ValueError('trait cohort differs')
            np.add.at(sums,np.searchsorted(ids,c.block_ids),c.block_genetic_rhs[:,0,0])
        within[t]=adapter.within(sums,n=first.n_samples,rank=first.residual_rank)
    arrays=dict(block_ids=ids,block_counts=counts,block_ld_sums=ld);table=[]
    for ix,iy in itertools.combinations(range(len(TRAITS)),2):
        x,y=TRAITS[ix],TRAITS[iy];sums=np.zeros(len(ids));rhs=None
        for d in cross:
            matches=np.flatnonzero(np.all(d['pairs']==[ix,iy],axis=1))
            if len(matches)!=1:raise ValueError('missing or duplicated trait pair')
            j=matches[0];np.add.at(sums,np.searchsorted(ids,d['block_ids']),d['block_rhs'][:,j,0,0,0])
            current=float(d['residual_rhs'][j,0])
            if rhs is not None:np.testing.assert_allclose(current,rhs,rtol=1e-13)
            rhs=current
        fitted=adapter.cross(sums,left=within[x],right=within[y],overlap_rhs=rhs)
        ordinary=fitted.rg_reps[:,0];key=f'{x}__{y}';arrays[key+'__rg_replicates']=ordinary
        for mode in args.modes:
            path=args.pilot_fits/f'{key}__{mode}.npz'
            d,p=load_array_artifact(path,kind='summit.cross_trait.fit')
            if p['chromosomes']!=args.chromosomes or p['gram_mode']!=mode:raise ValueError('pilot fit design differs')
            np.testing.assert_array_equal(d['block_ids'],ids);hashes[str(path)]=file_sha256(path)
            for quantity in ('baseline_rg','centered_baseline_rg'):
                point=float(d[quantity].item());loo=d['loo_'+quantity].reshape(len(ids))
                se=float(np.sqrt((len(ids)-1)*np.var(loo)))
                paired_se=float(np.sqrt((len(ids)-1)*np.var(loo-ordinary[:-1])))
                difference=point-ordinary[-1]
                table.append(dict(trait_x=x,trait_y=y,mode=mode,quantity=quantity,
                    context_rg=point,ordinary_rg=float(ordinary[-1]),context_jackknife_se=se,
                    ordinary_jackknife_se=float(fitted.rg[0,1]),difference=difference,
                    difference_in_context_se=difference/se if se>0 else np.nan,
                    paired_difference_se=paired_se,
                    within_one_context_se=bool(np.isfinite(difference) and se>0 and abs(difference)<=se)))
    provenance=dict(input_sha256=input_hashes,source_sha256=hashes,chromosomes=args.chromosomes,
        script_sha256=file_sha256(__file__),adapter_sha256=file_sha256(ROOT/'src/summit/context/cross_trait_bivariate.py'),
        genotype_traversals=0,ordinary_model='single baseline kernel; reference chromosome constant-residual profile',
        block_adapter='exact HE block sums expanded as block means; no per-SNP observation reconstruction',
        uncertainty='ordinary SUMMIT frozen-source block jackknife; paired with context deletion IDs',
        criterion='absolute point difference <= one context-model jackknife SE; both baseline centerings reported')
    write_array_artifact(args.output/'ordinary_bivariate.npz',kind='summit.cross_trait.bivariate_regression',arrays=arrays,provenance=provenance)
    write_table(args.output/'baseline_rg_regression.tsv',table)
    with (args.output/'COMPLETE.json').open('x') as f:
        json.dump(dict(provenance,tables={'baseline_rg_regression.tsv':file_sha256(args.output/'baseline_rg_regression.tsv')}),f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','study-root','pilot-fits','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--chromosomes',nargs='+',type=int,default=list(range(1,23)))
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=p.parse_args();args.output.mkdir(exist_ok=False);run(args)


if __name__=='__main__':main()
