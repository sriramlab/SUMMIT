"""Replay confirmation phenotypes against independent bounded sample-space algebra."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scripts.epistasis.confirm_inference import panel
from summit.epistasis.quadratic import quadratic_sf


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--real-genotypes',required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]/'benchmarks/epistasis'
    saved=pd.read_csv(root/'confirmed_linear_real_identifiable_v2_20261001/replicates.csv')
    records=[]
    for panel_index,seed in enumerate((1763011,1773019)):
        x,h,_=panel(args.real_genotypes,seed,n=384,m=96)
        fixed=np.column_stack([np.ones(384),x,h[:,:8]])
        rotation=null_space(fixed.T);r=rotation.shape[1]
        rng=np.random.default_rng(2627301+panel_index)
        mean=x@rng.normal(size=96)/np.sqrt(96)+h[:,:8]@rng.normal(size=8)/3
        a=np.zeros(96);a[:4]=1;b=np.zeros(96);b[4:8]=1
        overlap=b.copy();overlap[2:4]=1
        for name,right in (('cross',b),('overlap',overlap),('within',a),('remainder',(a==0).astype(float))):
            # Independent construction of unique unordered products and masses.
            columns=[];weights=[]
            for i in range(96):
                for j in range(i+1,96):
                    weight=a[i]*right[j]+(0 if name=='within' else a[j]*right[i])
                    if weight>0:columns.append(x[:,i]*x[:,j]);weights.append(weight)
            f=np.column_stack(columns)*np.sqrt(np.array(weights)/sum(weights))
            if f.shape[1]>=r:
                f=f@rng.normal(size=(f.shape[1],32))/np.sqrt(32)
                name+='_fixed_sketch32'
            reduced=rotation.T@f
            coefficient=rng.normal(size=f.shape[1])
            coefficient*=np.sqrt(.03*r/np.sum((reduced@coefficient)**2))
            for signal in (False,True):
                y=mean[:,None]+signal*(f@coefficient)[:,None]+rng.normal(size=(384,100))
                if signal:continue
                z=rotation.T@y
                k=reduced@reduced.T
                eigenvalues=np.linalg.eigvalsh(k)
                ratio=np.einsum('it,ij,jt->t',z,k,z)/np.sum(z*z,axis=0)
                ps=np.array([quadratic_sf(0,eigenvalues-q,atol=1e-10)['p'] for q in ratio])
                setting=f'real{panel_index}_{name}_null'
                original=saved.loc[(saved.setting==setting)&(saved.method=='kernel'),'p'].to_numpy(float)
                error=float(np.max(abs(ps-original)))
                if error>1e-8:raise AssertionError((setting,error))
                records.append(dict(setting=setting,replayed_phenotypes=100,residual_rank=r,
                    maximum_p_difference=error,rejections=int(np.sum(ps<=.05)),minimum_p=float(ps.min())))
    (args.out/'audit.json').write_text(json.dumps(dict(records=records,
        contract='Same saved confirmation seeds and phenotypes, not new replicates. Independent SciPy SVD residual basis, explicit pair enumeration and dense residual-space kernel; shared scalar quadratic-tail routine already checked separately.'),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
