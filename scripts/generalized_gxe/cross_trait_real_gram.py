"""Dense common-variant chr22 validation on two real masks, with bounded RAM.

Genotypes are read once. Ordered kernels are constructed in sample-row tiles;
the full collection of 25 dense N by N kernels is never retained.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import ctypes
import json
import os
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from bed_reader import open_bed
from summit.context.cross_trait_gram import chromosome_gram,orientation_matrix,saved
from summit.context.cross_trait_zpass import repair_single_annotation
from summit.context.cross_trait_fit import assemble_cross_trait_normal_equations,solve_cross_trait_normal_equations
from summit.context.reference_zpass_cli import authenticated_reference,file_sha256


def tiled_moments(kernel,phi_x,phi_y,ux,uy,*,shared_x,shared_y,yx=None,yy=None,tile=256):
    q=phi_x.shape[1];nx,ny=kernel.shape;c=ux.shape[1]
    right=np.stack([kernel@(phi_y[:,b,None]*uy) for b in range(q)])
    left=np.stack([(ux*phi_x[:,a,None]).T@kernel for a in range(q)])
    middle=np.einsum('ia,ic,bid->abcd',phi_x,ux,right,optimize=True)
    gram=np.zeros((q*q,q*q));rhs=np.zeros(q*q);diag=np.empty((q*q,len(shared_x)))
    for begin in range(0,nx,tile):
        end=min(begin+tile,nx);ix=slice(begin,end);block=np.empty((q*q,end-begin,ny))
        for a in range(q):
            for b in range(q):
                value=(phi_x[ix,a,None]*kernel[ix])*phi_y[None,:,b]
                value-=ux[ix]@(left[a]*phi_y[None,:,b])
                value-=(right[b,ix]*phi_x[ix,a,None])@uy.T
                value+=ux[ix]@middle[a,b]@uy.T
                block[a*q+b]=value
        flat=block.reshape(q*q,-1);gram+=flat@flat.T
        shared=(shared_x>=begin)&(shared_x<end)
        diag[:,shared]=block[:,shared_x[shared]-begin,shared_y[shared]]
        if yx is not None:rhs+=np.einsum('i,aij,j->a',yx[ix],block,yy,optimize=True)
    return gram,diag,rhs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--bed-prefix',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=20000);parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic()
    threadpool_limits(limits=args.threads)
    libc=ctypes.CDLL(None);assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1
    master=args.base/'full_cohort_inputs_20260916/height_raw.npz';refroot=args.base/'shared_reference_full_20260916'
    manifest,panel,refdir,record=authenticated_reference(refroot/'MANIFEST.json',refroot,22,master)
    with np.load(master) as z:rows,phi,fixed=z['rows'],z['phi'],z['fixed']
    selected=np.sort(np.random.default_rng(1).choice(len(rows),args.n,replace=False))
    fam_rows=rows[selected];phi=phi[selected];fixed=fixed[selected];u=np.linalg.qr(fixed)[0];q=phi.shape[1]
    weights=np.load(args.annotations);common=np.flatnonzero(weights[:,2]>0)
    from summit.context.spec import array_sha256
    assert array_sha256(weights)==panel['annotation_sha256']
    with np.load(refdir/'reference_aux.npz') as z:mean,inverse=z['affine_mean'][common],z['affine_inverse_scale'][common]
    prefix=str(args.bed_prefix)
    assert file_sha256(prefix+'.bim')==panel['bim_sha256'] and file_sha256(prefix+'.fam')==panel['fam_sha256']
    assert Path(prefix+'.bed').stat().st_size==panel['bed_size']
    with open_bed(prefix+'.bed',num_threads=args.threads) as bed:
        raw=bed.read(index=np.s_[fam_rows,common],dtype='float32')
    missing=np.isnan(raw);g=raw.astype(float);del raw
    g-=mean;g*=inverse;g[missing]=0;del missing
    mass=len(common);kernel=g@g.T/mass
    # Independent direct genotype dot-products qualify the dense GRM product.
    check=np.linspace(0,len(g)-1,19,dtype=int)
    np.testing.assert_allclose(kernel[np.ix_(check,check)],g[check]@g[check].T/mass,rtol=1e-12,atol=1e-12)
    del g
    print(json.dumps(dict(phase='dense_grm',n=args.n,m=mass,seconds=time.monotonic()-start)),flush=True)
    tr,dr,_=tiled_moments(kernel,phi,phi,u,u,shared_x=np.arange(args.n),shared_y=np.arange(args.n))
    rhs=(phi[:,:,None]*u[:,None,:]).reshape(args.n,-1)
    w=(rhs.T@kernel@rhs).reshape(q,u.shape[1],q,u.shape[1]).transpose(0,2,1,3)
    repaired=repair_single_annotation(saved(tr,q),w,w,mass=1)
    repair_error=float(np.linalg.norm(repaired-tr)/np.linalg.norm(tr))
    if repair_error>1e-12:raise AssertionError(('Z repair',repair_error))
    ref=SimpleNamespace(n_samples=args.n,num_basis=q,annotation_names=('common',),residual_rank=args.n-u.shape[1],
        block_masses=np.array([[mass]],dtype=float),block_directed=saved(tr,q)[None]*mass**2/(args.n-u.shape[1])**2)
    diagonal=orientation_matrix(q)@dr
    traits=[];input_hashes={}
    for name,path in (
        ('fev1',args.base/'full_cohort_inputs_20260916/fev1_best_acceptable_litres.npz'),
        ('ldl',args.base/'imputed_expansion_20260917/inputs/ldl_raw.npz')):
        with np.load(path) as z:
            present=np.isin(fam_rows,z['rows']);idx=np.flatnonzero(present)
            positions=np.searchsorted(z['rows'],fam_rows[idx]);np.testing.assert_array_equal(z['rows'][positions],fam_rows[idx])
            ut=np.linalg.qr(fixed[idx])[0];yt=z['y'][positions].reshape(-1);yt=yt-ut@(ut.T@yt)
            yt*=np.sqrt((len(idx)-ut.shape[1])/(yt@yt))
        traits.append((name,idx,ut,yt));input_hashes[str(path)]=file_sha256(path)
    tables=[];arrays=dict(reference_gram=tr,repaired_reference_gram=repaired,reference_diagonal=dr)
    # Both selected masks individually, followed by their cross-trait overlap.
    for ix,iy in ((0,0),(1,1),(0,1)):
        nx,x,ux,yx=traits[ix];ny,y,uy,yy=traits[iy]
        overlap,lx,ly=np.intersect1d(x,y,return_indices=True)
        true,diag,g_rhs=tiled_moments(kernel[np.ix_(x,y)],phi[x],phi[y],ux,uy,
            shared_x=lx,shared_y=ly,yx=yx,yy=yy)
        d=np.column_stack([phi[overlap,a]*phi[overlap,b]*(1 if a==b else 2)
                          for a in range(q) for b in range(a,q)])
        xo,yo=ux[lx],uy[ly]
        cross=np.stack([xo.T@(d[:,h,None]*yo) for h in range(d.shape[1])])
        rr=d.T@(d*(1-np.sum(xo*xo,1)-np.sum(yo*yo,1))[:,None])+np.einsum('hij,kij->hk',cross,cross)
        gr=diag@d;r_rhs=d.T@(yx[lx]*yy[ly])
        def solve(gram):
            eq=assemble_cross_trait_normal_equations(genetic_gram=gram,genetic_rhs=g_rhs,genetic_residual=gr,
                residual_gram=rr,residual_rhs=r_rhs,num_basis=q,annotation_masses=[mass])
            return solve_cross_trait_normal_equations(eq).coefficients[:q*q]
        coefficients=solve(true);label=f'{nx}__{ny}'
        arrays[label+'_true']=true
        for mode in ('factorized','factorized_plus_residual','legacy_transport','legacy_transport_exact'):
            assembled=chromosome_gram(ref,diagonal,phi,x,y,global_masses=[mass],mode=mode,
                repaired_blocks=repaired[None] if mode=='legacy_transport_exact' else None)
            estimate=assembled.matrix;error=estimate-true
            rowscale=np.sqrt(np.maximum(np.diag(true),0))
            scale=rowscale[:,None]*rowscale[None,:]
            delta=solve(estimate)-coefficients
            tables.append(dict(pair=label,n_x=len(x),n_y=len(y),n_overlap=len(overlap),mode=mode,
                gram_relative_frobenius=float(np.linalg.norm(error)/np.linalg.norm(true)),
                maximum_entry_error_correlation_scale=float(np.max(np.abs(error)/np.maximum(scale,1e-30))),
                coefficient_relative_error=float(np.linalg.norm(delta)/max(np.linalg.norm(coefficients),1e-30)),
                same_person_relative_error=float(np.linalg.norm(assembled.same_person-diag@diag.T)/np.linalg.norm(diag@diag.T))))
            arrays[label+'_'+mode]=estimate
        print(json.dumps(dict(phase='mask_done',pair=label,seconds=time.monotonic()-start)),flush=True)
    path=args.output/'gram_mode_validation.tsv'
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]),delimiter='\t');writer.writeheader();writer.writerows(tables)
    with (args.output/'dense_moments.npz').open('xb') as f:np.savez(f,**arrays)
    result=dict(n=args.n,m=mass,seed=1,Z_repair_relative_error=repair_error,seconds=time.monotonic()-start,
        reference_source_files=record['files'],input_sha256=input_hashes,script_sha256=file_sha256(__file__),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpus=sorted(os.sched_getaffinity(0)),
        common_genotype_scale='sealed full-cohort affine, no subsample renormalization',
        files={p.name:file_sha256(p) for p in args.output.iterdir()})
    with (args.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
