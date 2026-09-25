"""Context-specific genetic correlation curves from saved covariance estimates.

Offsets use the master-standardized exposure units and are added to each
trait's own context mean. These are fitted cross-sectional response surfaces,
not longitudinal predictions or causal intervention effects.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/generalized_gxe')]
import numpy as np
from summit.context.cross_trait_uncertainty import paired_delta_covariances
from summit.context.cross_trait_zpass import load_array_artifact
from summit.context.reference_zpass_cli import file_sha256
from cross_trait_biology import LABELS,bh,write


def profiles(primitive,*,mean_x,mean_y,exposure,offsets):
    xx,yy,xy=(primitive[...,i,:,:] for i in range(3))
    px=np.tile(np.r_[1.,mean_x],(len(offsets),1));py=np.tile(np.r_[1.,mean_y],(len(offsets),1))
    px[:,exposure]+=offsets;py[:,exposure]+=offsets
    bilinear=lambda a,left,right:np.einsum('ki,...ij,kj->...k',left,a,right)
    vx,vy=bilinear(xx,px,px),bilinear(yy,py,py)
    with np.errstate(divide='ignore',invalid='ignore'):
        rg=np.where((vx.real>0)&(vy.real>0),bilinear(xy,px,py)/np.sqrt(vx*vy),np.nan)
    return dict(context_rg=rg,high_minus_low_rg=rg[...,-1]-rg[...,0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    offsets=np.linspace(-1,1,9);rows=[];changes=[];sources={};seen=set();blocks=None
    for root in args.fits:
        complete=json.loads((root/'COMPLETE.json').read_text())
        sources[str(root/'COMPLETE.json')]=file_sha256(root/'COMPLETE.json')
        for name,digest in complete['tables'].items():
            if file_sha256(root/name)!=digest:raise ValueError('fit table checksum differs')
        for path in sorted(root.glob('*__factorized.npz')):
            fit,meta=load_array_artifact(path,kind='summit.cross_trait.fit')
            if meta['chromosomes']!=list(range(1,23)) or str(fit['deletion_method'])!='target_moments':
                raise ValueError('complete genome and corrected deletion units required')
            if blocks is None:blocks=fit['block_ids']
            np.testing.assert_array_equal(blocks,fit['block_ids'])
            if len(blocks)!=200:raise ValueError('200 paired blocks required')
            x,y=meta['trait_x'],meta['trait_y'];key=frozenset((x,y))
            if key in seen:raise ValueError('duplicate pair')
            seen.add(key);sources[str(path)]=file_sha256(path)
            names=list(fit['basis_names'].astype(str))
            point=np.stack([fit['omega_'+k][0] for k in ('xx','yy','xy')])
            deleted=np.stack([fit['loo_omega_'+k][:,0] for k in ('xx','yy','xy')],axis=1)
            for exposure in ('age','bmi_raw'):
                kwargs=dict(mean_x=fit['mean_x'],mean_y=fit['mean_y'],exposure=names.index(exposure),offsets=offsets)
                function=lambda value:profiles(value,**kwargs)
                values=function(point);covariances=paired_delta_covariances(function,point,deleted)
                for i,offset in enumerate(offsets):
                    value=float(values['context_rg'][i]);se=float(np.sqrt(np.maximum(0,covariances['context_rg'][i,i])))
                    rows.append(dict(trait_x=x,trait_y=y,exposure=exposure,offset_master_sd=float(offset),
                        rg=value,delta_se=se,lower_95=value-1.96*se,upper_95=value+1.96*se,
                        admissible=bool(np.isfinite(value) and abs(value)<=1),exploratory=True))
                value=float(values['high_minus_low_rg']);se=float(np.sqrt(np.maximum(0,covariances['high_minus_low_rg'].item())))
                changes.append(dict(trait_x=x,trait_y=y,exposure=exposure,high_minus_low_rg=value,
                    paired_delta_se=se,lower_95=value-1.96*se,upper_95=value+1.96*se,
                    contrast='own mean +1 master SD minus own mean -1 master SD',exploratory=True))
    from scipy.stats import norm
    pvalues=[2*norm.sf(abs(r['high_minus_low_rg']/r['paired_delta_se']))
        if np.isfinite(r['paired_delta_se']) and r['paired_delta_se']>0 else np.nan for r in changes]
    for record,pvalue,fdr in zip(changes,pvalues,bh(pvalues)):
        record.update(p=float(pvalue),fdr=float(fdr),fdr_family='all trait-pair age and BMI endpoint contrasts')
    write(args.output/'context_profiles.tsv',rows);write(args.output/'paired_context_changes.tsv',changes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=['diastolic_blood_pressure_raw','hba1c_raw','height_raw','platelet_count_raw']
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True)
    for col,target in enumerate(targets):
        for row,(exposure,label) in enumerate([('age','Age'),('bmi_raw','BMI')]):
            selected=sorted([r for r in rows if {r['trait_x'],r['trait_y']}=={'ldl_raw',target} and r['exposure']==exposure],key=lambda r:r['offset_master_sd'])
            ax=axes[row,col]
            if selected:
                grid=np.array([r['offset_master_sd'] for r in selected]);values=np.array([r['rg'] for r in selected])
                low=np.array([r['lower_95'] for r in selected]);high=np.array([r['upper_95'] for r in selected])
                ax.plot(grid,values,color='#0072B2');ax.fill_between(grid,low,high,color='#0072B2',alpha=.18)
            ax.axhline(0,color='#bbbbbb',lw=.6);ax.axvline(0,color='#bbbbbb',lw=.6,ls=':')
            ax.set_xlabel(label+' offset (master SD)');ax.set_xticks([-1,0,1]);ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_title('LDL – '+LABELS[target])
            if col==0:ax.set_ylabel('Context-specific genetic rg')
    fig.suptitle('Fitted genetic sharing across age and BMI',fontsize=14)
    fig.text(.02,.015,'Offsets are relative to each trait’s mean; all other contexts remain at that mean. Shading: pointwise 95% delta intervals.\n'
        'Common-bin, cross-sectional model estimates. These curves do not predict causal effects of aging, weight loss or treatment.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.94))
    for suffix in ('pdf','png'):fig.savefig(args.output/f'context_specific_genetic_correlations.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    files={path.name:file_sha256(path) for path in args.output.iterdir()}
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(sources=sources,script_sha256=file_sha256(__file__),
        files=files,context_units='master-standardized offset from trait-specific means'),indent=2)+'\n')


if __name__=='__main__':main()
