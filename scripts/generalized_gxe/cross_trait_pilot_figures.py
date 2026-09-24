"""Plot the audited common-bin pilot without refitting or genotype access."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from summit.context.cross_trait_zpass import load_array_artifact

LABELS={
    'ldl_raw':'LDL','apo_b_raw':'ApoB','cholesterol_raw':'Total cholesterol',
    'non_hdl_cholesterol_raw':'Non-HDL','hba1c_raw':'HbA1c',
    'diastolic_blood_pressure_raw':'DBP','height_raw':'Height',
    'platelet_count_raw':'Platelets',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fits','report','audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    audit=json.loads((a.audit/'COMPLETE.json').read_text())
    assert audit['paired_blocks']==200 and audit['chromosomes']==list(range(1,23))
    assert audit['verified_pair_mode_artifacts']==112
    for root in (a.fits,a.report):
        marker=root/'COMPLETE.json'
        assert audit['source_completion_sha256'][str(marker)]==sha(marker)
    complete=json.loads((a.report/'COMPLETE.json').read_text())
    for name,digest in complete['tables'].items():assert sha(a.report/name)==digest
    with (a.report/'baseline_vs_orthogonal_response.tsv').open() as f:
        contrasts=list(csv.DictReader(f,delimiter='\t'))
    assert len(contrasts)==28
    contrasts.sort(key=lambda row:row['pair_category']!='cardiometabolic')
    traits=list(LABELS);n=len(traits)
    covariance=np.full((n,n),np.nan);se=np.full((n,n),np.nan)
    self_entries={}
    for path in sorted(a.fits.glob('*__factorized.npz')):
        assert audit['fit_sha256'][path.name]==sha(path)
        fit,provenance=load_array_artifact(path,kind='summit.cross_trait.fit')
        assert np.array_equal(fit['block_ids'],audit['block_ids'])
        names=list(fit['basis_names']);age=names.index('age')-1;bmi=names.index('bmi_raw')-1
        ix,iy=[traits.index(provenance[key]) for key in ('trait_x','trait_y')]
        for i,j,q,r in ((ix,iy,age,bmi),(iy,ix,bmi,age)):
            covariance[i,j]=fit['h_xy'][0,q,r]
            se[i,j]=np.sqrt(199*np.var(fit['loo_h_xy'][:,0,q,r]))
        for i,side in ((ix,'x'),(iy,'y')):
            value=float(fit['h_'+side+side][0,age,bmi])
            error=float(np.sqrt(199*np.var(fit['loo_h_'+side+side][:,0,age,bmi])))
            if i in self_entries:np.testing.assert_allclose([value,error],self_entries[i],rtol=1e-12,atol=1e-14)
            self_entries[i]=(value,error);covariance[i,i]=value;se[i,i]=error
    assert np.isfinite(covariance).all() and np.isfinite(se).all()
    # Import plotting only after input and paired-block authentication.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    a.output.mkdir(exist_ok=False)
    def save(fig,name):
        fig.savefig(a.output/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(a.output/(name+'.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,9))
    colors={'cardiometabolic':'#176b91','control_involving':'#b26b28'}
    undefined=[]
    for y,row in enumerate(contrasts):
        point=float(row['orthogonal_minus_baseline_rg_estimate'])
        low=float(row['orthogonal_minus_baseline_rg_lower_95'])
        high=float(row['orthogonal_minus_baseline_rg_upper_95'])
        if np.isfinite([point,low,high]).all():
            ax.errorbar(point,y,xerr=[[point-low],[high-point]],fmt='o',ms=4,
                        color=colors[row['pair_category']],lw=1,capsize=2)
        else:
            undefined.append(y)
    for y in undefined:ax.text(.02,y,'undefined',transform=ax.get_yaxis_transform(),color='#666666',va='center')
    ax.set_yticks(range(28),[LABELS[r['trait_x']]+' / '+LABELS[r['trait_y']] for r in contrasts],fontsize=8)
    ax.invert_yaxis();ax.axvline(0,color='#666666',ls='--',lw=.8)
    ax.axhline(14.5,color='#aaaaaa',lw=.6)
    ax.set_xlabel('Orthogonal-response rg minus baseline rg\nPaired nominal 95% interval, 200 deletion blocks')
    ax.set_title('Exploratory common-bin pilot\nFactorized Gram with same-person terms on actual rows',loc='left')
    ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=color,label=label) for label,color in
        [('Cardiometabolic pair',colors['cardiometabolic']),('Control-involving pair',colors['control_involving'])]],
        loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,ncol=2)
    fig.text(.02,-.025,'Unconstrained estimates; undefined ratios are labeled. No multiplicity adjustment.',fontsize=8)
    save(fig,'paired_response_minus_baseline')
    fig,ax=plt.subplots(figsize=(8,7))
    limit=float(np.max(abs(covariance)))
    im=ax.imshow(covariance,cmap='RdBu_r',vmin=-limit if limit else -1,vmax=limit or 1)
    ax.set_xticks(range(n),list(LABELS.values()),rotation=45,ha='right')
    ax.set_yticks(range(n),list(LABELS.values()))
    ax.set_xlabel('Trait with BMI response');ax.set_ylabel('Trait with age response')
    ax.set_title('Baseline-orthogonal age–BMI response covariance\nExploratory common-bin pilot, factorized / actual rows',loc='left')
    significant=abs(covariance)>1.96*se
    ys,xs=np.nonzero(significant);ax.scatter(xs,ys,c='#111111',s=12)
    ax.axhline(5.5,color='#555555',lw=1);ax.axvline(5.5,color='#555555',lw=1)
    bar=fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    bar.set_label('Covariance in standardized phenotype and exposure bases')
    fig.text(.02,-.08,'Dots: paired nominal 95% intervals exclude zero; no multiplicity adjustment.\n'
        'Diagonal cells use within-trait H. A nonsignificant control does not establish absence.',fontsize=8)
    save(fig,'age_bmi_cross_trait_covariance')
    result=dict(script_sha256=sha(Path(__file__)),audit_completion_sha256=sha(a.audit/'COMPLETE.json'),
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        source_completion_sha256=audit['source_completion_sha256'],chromosomes=audit['chromosomes'],
        paired_blocks=200,gram_mode='factorized',exploratory=True,
        genotype_traversals=0,normal_equation_solves=0,
        undefined_paired_contrasts=len(undefined),
        figures={p.name:sha(p) for p in sorted(a.output.iterdir())})
    with (a.output/'COMPLETE.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
