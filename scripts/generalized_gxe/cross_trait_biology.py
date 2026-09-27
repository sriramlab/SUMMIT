"""Paired biological contrasts and publication figures from authenticated fits.

Uses all supplied default-mode pairs; no selection by significance. External
comparisons use the reported single-exposure interaction correlation, without
own-baseline orthogonalization. No genotype or phenotype files are opened.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.meta_path[:]=[f for f in sys.meta_path if type(f).__module__!='_gwldcore_editable']
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.stats import norm
from summit.context.annotations import _jackknife_covariance
from summit.context.cross_trait_fit import cross_trait_derived
from summit.context.cross_trait_zpass import load_array_artifact
from summit.context.reference_zpass_cli import file_sha256

LABELS={'ldl_raw':'LDL','apo_b_raw':'ApoB','cholesterol_raw':'Total cholesterol',
    'non_hdl_cholesterol_raw':'Non-HDL','hba1c_raw':'HbA1c','diastolic_blood_pressure_raw':'DBP',
    'height_raw':'Height','platelet_count_raw':'Platelets','triglycerides_log':'Triglycerides',
    'hdl_raw':'HDL','aspartate_aminotransferase_log':'AST','alanine_aminotransferase_log':'ALT',
    'gamma_glutamyltransferase_log':'GGT','creatinine_log':'Creatinine','urea_log':'Urea',
    'white_blood_cell_count_log':'WBC','bp_systolic_raw':'SBP','glucose_log':'Glucose'}
EXTERNAL={'TC':'cholesterol_raw','LDL-C':'ldl_raw','TG':'triglycerides_log','HDL-C':'hdl_raw',
    'AST':'aspartate_aminotransferase_log','ALT':'alanine_aminotransferase_log',
    'GGT':'gamma_glutamyltransferase_log','sCr':'creatinine_log','BUN':'urea_log',
    'Plt':'platelet_count_raw','WBC':'white_blood_cell_count_log',
    'SBP':'bp_systolic_raw','DBP':'diastolic_blood_pressure_raw'}
EXPOSURES={'Age':'age','Sex':'sex','Ever-smoking':'ever_smoked'}


def analysis_family(x,y):
    """Trait-set families fixed before the new chromosome scoring results."""
    pair=frozenset((x,y))
    if pair.issubset(tuple(LABELS)[:8]):return 'pilot'
    if pair in {frozenset(('glucose_log',t)) for t in
            ('hba1c_raw','ldl_raw','apo_b_raw','diastolic_blood_pressure_raw')}:
        return 'glycaemic_followup'
    external=(('triglycerides_log','hdl_raw'),('triglycerides_log','ldl_raw'),
        ('aspartate_aminotransferase_log','alanine_aminotransferase_log'),
        ('aspartate_aminotransferase_log','gamma_glutamyltransferase_log'),
        ('creatinine_log','urea_log'),('platelet_count_raw','white_blood_cell_count_log'),
        ('diastolic_blood_pressure_raw','bp_systolic_raw'))
    return 'external_followup' if pair in {frozenset(p) for p in external} else 'other_exploratory'


def quantities(xy,xx,yy,*,mean_x,mean_y,context_covariance,age,bmi):
    base=cross_trait_derived(xy,xx,yy,mean_x=mean_x,mean_y=mean_y,context_covariance=context_covariance)
    h,hx,hy=(base[k] for k in ('h_xy','h_xx','h_yy'))
    vx=np.diagonal(hx,axis1=-2,axis2=-1);vy=np.diagonal(hy,axis1=-2,axis2=-1)
    def ratio(c,x,y):
        with np.errstate(invalid='ignore',divide='ignore'):
            return np.where((x.real>0)&(y.real>0),c/np.sqrt(x*y),np.nan)
    r=ratio(h,vx[..., :,None],vy[...,None,:])
    result={name:base[name] for name in ('baseline_rg','centered_baseline_rg','orthogonal_rg','response_rg',
        'orthogonal_minus_baseline_rg','response_minus_baseline_rg','h_xy')}
    result.update(orthogonal_exposure_rg=np.diagonal(r,axis1=-2,axis2=-1),
        orthogonal_minus_centered_baseline_rg=base['orthogonal_rg']-base['centered_baseline_rg'],
        age_bmi_rg=r[...,age,bmi],bmi_age_rg=r[...,bmi,age],
        age_bmi_asymmetry=h[...,age,bmi]-h[...,bmi,age],
        age_bmi_rg_asymmetry=r[...,age,bmi]-r[...,bmi,age])
    q=xy.shape[-1];cx=np.eye(q);cy=np.eye(q);cx[0,1:]=mean_x;cy[0,1:]=mean_y
    x=cx@xx@cx.T;y=cy@yy@cy.T;z=cx@xy@cy.T
    # Regress both response vectors on the two-dimensional baseline vector.
    baseline=np.stack([np.stack([x[...,0,0],z[...,0,0]],axis=-1),
                       np.stack([z[...,0,0],y[...,0,0]],axis=-1)],axis=-2)
    valid=np.linalg.eigvalsh(baseline.real).min(axis=-1)>1e-10*np.maximum(1,np.linalg.norm(baseline.real,axis=(-2,-1)))
    safe=np.where(valid[...,None,None],baseline,np.eye(2))
    bx=np.stack([x[...,1:,0],z[...,1:,0]],axis=-1)
    by=np.stack([z[...,0,1:],y[...,1:,0]],axis=-1)
    inverse=np.linalg.inv(safe)
    joint=z[...,1:,1:]-bx@inverse@by.swapaxes(-1,-2)
    jx=x[...,1:,1:]-bx@inverse@bx.swapaxes(-1,-2)
    jy=y[...,1:,1:]-by@inverse@by.swapaxes(-1,-2)
    trace=lambda a:np.einsum('ij,...ji->...',context_covariance,a)
    jr=np.where(valid,ratio(trace(joint),trace(jx),trace(jy)),np.nan)
    result.update(joint_baseline_response_rg=jr,joint_minus_own_baseline_rg=jr-base['orthogonal_rg'])
    with np.errstate(invalid='ignore',divide='ignore'):
        denom=np.sqrt(trace(hx)*trace(hy))
        result['trace_contribution']=(.5*np.einsum('ij,...ij->...i',context_covariance,h+h.swapaxes(-1,-2))
            /np.where((trace(hx).real>0)&(trace(hy).real>0),denom,np.nan)[...,None])
    return result


def paired_values(fit):
    names=list(fit['basis_names'].astype(str));q=len(names);p=q*q
    kwargs=dict(mean_x=fit['mean_x'],mean_y=fit['mean_y'],context_covariance=fit['context_covariance'],
        age=names.index('age')-1,bmi=names.index('bmi_raw')-1)
    xx,yy,xy=[fit[k][0] for k in ('omega_xx','omega_yy','omega_xy')]
    lx,ly,lz=[fit[k][:,0] for k in ('loo_omega_xx','loo_omega_yy','loo_omega_xy')]
    point=quantities(xy,xx,yy,**kwargs);loo=quantities(lz,lx,ly,**kwargs)
    flat=np.r_[xx.ravel(),yy.ravel(),xy.ravel()];step=1e-30
    perturbed=(flat[None]+1j*step*np.eye(3*p)).reshape(3*p,3,q,q)
    values=quantities(perturbed[:,2],perturbed[:,0],perturbed[:,1],**kwargs)
    covariance=_jackknife_covariance(np.c_[lx.reshape(len(lx),-1),ly.reshape(len(ly),-1),lz.reshape(len(lz),-1)])
    errors={};jackknife={};influences={}
    primitive=np.c_[lx.reshape(len(lx),-1),ly.reshape(len(ly),-1),lz.reshape(len(lz),-1)]
    primitive-=primitive.mean(0)
    for name,value in values.items():
        j=np.where(np.isfinite(point[name]).ravel()[:,None],value.imag.reshape(3*p,-1).T/step,np.nan)
        errors[name]=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',j,covariance,j))).reshape(point[name].shape)
        jackknife[name]=np.sqrt((len(lx)-1)*np.var(loo[name],axis=0))
        influences[name]=(primitive@j.T).reshape((len(lx),)+point[name].shape)
    return point,errors,jackknife,names[1:],influences


def bh(values):
    values=np.asarray(values);result=np.full(values.shape,np.nan);valid=np.flatnonzero(np.isfinite(values))
    order=valid[np.argsort(values[valid])]
    if len(order):result[order]=np.minimum(1,np.minimum.accumulate((values[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return result


def assign_fdr(rows):
    """Keep pilot inference unchanged when adding prespecified follow-up sets."""
    groups={}
    for row in rows:
        family=analysis_family(row['trait_x'],row['trait_y'])
        row['analysis_family']=family
        groups.setdefault((family,row['quantity']),[]).append(row)
    for (family,name),selected in groups.items():
        for row,fdr in zip(selected,bh([r['p'] for r in selected])):
            row['fdr_family']=family+':'+name;row['fdr']=float(fdr)


def write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)


def glycaemic_specificity(lookup):
    """Paired HbA1c-minus-glucose sharing, not a difference of significance."""
    rows=[]
    for anchor in ('ldl_raw','apo_b_raw','diastolic_blood_pressure_raw'):
        left=lookup.get(frozenset((anchor,'hba1c_raw')))
        right=lookup.get(frozenset((anchor,'glucose_log')))
        if left is None or right is None:continue
        if left[2]!=right[2]:raise ValueError('exposure order differs between paired fits')
        for quantity in ('baseline_rg','orthogonal_rg','joint_baseline_response_rg',
                         'response_rg','orthogonal_exposure_rg'):
            value=np.asarray(left[0][quantity])-np.asarray(right[0][quantity])
            influence=np.asarray(left[5][quantity])-np.asarray(right[5][quantity])
            b=len(influence)
            if b<2:raise ValueError('paired influence covariance needs at least two blocks')
            error=np.sqrt((b-1)/b*np.sum(influence**2,axis=0))
            for i,(point,se) in enumerate(zip(value.ravel(),error.ravel())):
                rows.append(dict(anchor=anchor,comparison='HbA1c sharing minus glucose sharing',
                    quantity=quantity,entry=i,exposure=left[2][i] if value.size==len(left[2]) else 'all',
                    estimate=float(point),paired_delta_se=float(se),
                    lower_95=float(point-1.96*se),upper_95=float(point+1.96*se),
                    p=float(2*norm.sf(abs(point/se))) if np.isfinite(point) and np.isfinite(se) and se>0 else np.nan,
                    exploratory=True))
    for quantity in dict.fromkeys(r['quantity'] for r in rows):
        selected=[r for r in rows if r['quantity']==quantity]
        for row,fdr in zip(selected,bh([r['p'] for r in selected])):
            row.update(fdr=float(fdr),fdr_family='glycaemic_specificity:'+quantity)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits',type=Path,nargs='+',required=True)
    parser.add_argument('--namba-table',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    rows=[];lookup={};sources={};blocks=None;cohorts={}
    for root in args.fits:
        complete=json.loads((root/'COMPLETE.json').read_text())
        sources[str(root/'COMPLETE.json')]=file_sha256(root/'COMPLETE.json')
        for name,digest in complete['tables'].items():
            if file_sha256(root/name)!=digest:raise ValueError('fit table checksum differs')
        for path in sorted(root.glob('*__factorized.npz')):
            fit,meta=load_array_artifact(path,kind='summit.cross_trait.fit')
            if meta['chromosomes']!=list(range(1,23)) or str(fit['deletion_method'])!='target_moments':
                raise ValueError('biology requires complete genome and corrected deletion fits')
            if blocks is None:blocks=fit['block_ids']
            np.testing.assert_array_equal(fit['block_ids'],blocks)
            if len(blocks)!=200:raise ValueError('200 paired blocks required')
            x,y=meta['trait_x'],meta['trait_y'];key=frozenset((x,y))
            if key in lookup:raise ValueError('duplicate pair across fit roots')
            point,errors,jackknife,names,influences=paired_values(fit)
            lookup[key]=(point,errors,names,x,y,influences)
            sources[str(path)]=file_sha256(path)
            cohorts[x,y]=(meta['n_x'],meta['n_y'],meta['n_overlap'])
            for name,values in point.items():
                for i,value in enumerate(np.asarray(values).ravel()):
                    se=errors[name].ravel()[i];jk=jackknife[name].ravel()[i]
                    if name=='h_xy':ex,ey=names[i//len(names)],names[i%len(names)]
                    elif np.size(values)==len(names):ex=ey=names[i]
                    else:ex=ey='all' if 'age_bmi' not in name and 'bmi_age' not in name else 'age,BMI'
                    p=2*norm.sf(abs(value/se)) if np.isfinite(value) and np.isfinite(se) and se>0 else np.nan
                    rows.append(dict(trait_x=x,trait_y=y,quantity=name,entry=i,exposure_x=ex,exposure_y=ey,
                        estimate=float(value),delta_se=float(se),jackknife_se=float(jk),
                        lower_95=float(value-1.96*se),upper_95=float(value+1.96*se),p=float(p),
                        admissible=bool(np.isfinite(value) and ('rg' not in name or 'minus' in name or 'asymmetry' in name or abs(value)<=1)),
                        exploratory=True))
    # Each named contrast family is specified before testing. These FDRs are
    # descriptive under the usual BH dependence assumptions, not replication.
    assign_fdr(rows)
    write(args.output/'biological_contrasts.tsv',rows)
    # Highly correlated lipid measurements are not independent replications.
    # Average the four estimands and their paired influences, not their SEs.
    cluster=[];lipids=list(LABELS)[:4];averages={}
    for target in list(LABELS)[4:8]:
        selected=[lookup.get(frozenset((lipid,target))) for lipid in lipids]
        if any(value is None for value in selected):continue
        for quantity in ('baseline_rg','orthogonal_rg','orthogonal_minus_baseline_rg',
                         'joint_baseline_response_rg'):
            value=float(np.mean([v[0][quantity] for v in selected]))
            influence=np.mean([v[5][quantity] for v in selected],axis=0)
            error=float(np.sqrt((len(blocks)-1)/len(blocks)*np.sum(influence**2)))
            averages[target,quantity]=(value,influence)
            cluster.append(dict(comparison='mean of four lipid correlations with '+LABELS[target],quantity=quantity,
                estimate=value,delta_se=error,lower_95=value-1.96*error,upper_95=value+1.96*error,
                interpretation='post-hoc exploratory average; correlated lipid measurements, not four replications'))
    for quantity in ('baseline_rg','orthogonal_rg','orthogonal_minus_baseline_rg'):
        if all((t,quantity) in averages for t in ('diastolic_blood_pressure_raw','hba1c_raw')):
            a,ia=averages['diastolic_blood_pressure_raw',quantity];b,ib=averages['hba1c_raw',quantity]
            error=float(np.sqrt((len(blocks)-1)/len(blocks)*np.sum((ia-ib)**2)))
            cluster.append(dict(comparison='lipid–DBP mean minus lipid–HbA1c mean',quantity=quantity,
                estimate=a-b,delta_se=error,lower_95=a-b-1.96*error,upper_95=a-b+1.96*error,
                interpretation='post-hoc exploratory contrast with covariance across all eight pairs'))
    if cluster:write(args.output/'paired_lipid_group_contrasts.tsv',cluster)
    specificity=glycaemic_specificity(lookup)
    if specificity:write(args.output/'glycaemic_specificity_contrasts.tsv',specificity)
    external=[];unavailable=[]
    for row in json.loads(args.namba_table.read_text())[2:]:
        x,y=EXTERNAL.get(row['B']),EXTERNAL.get(row['C']);exposure=EXPOSURES.get(row['D'])
        key=frozenset((x,y))
        if exposure is None or key not in lookup:
            unavailable.append(dict(cohort=row['A'],trait_x=row['B'],trait_y=row['C'],exposure=row['D'],
                reason='exposure not harmonized' if exposure is None else 'pair unavailable in supplied completed fits'))
            continue
        point,se,names,_,_,_=lookup[key];e=names.index(exposure)
        external.append(dict(cohort=row['A'],trait_x=x,trait_y=y,exposure=exposure,
            namba_rg=float(row['E']),namba_se=float(row['F']),summit_rg=float(point['response_rg'][e]),
            summit_se=float(se['response_rg'][e]),
            comparison='overlapping UKB; not independent replication' if row['A']=='UKB1' else 'different population; independent cohort',
            source_table='Namba et al. 2026 Supplementary Table S22 (significant entries only)'))
    if external:write(args.output/'namba_comparisons.tsv',external)
    if unavailable:write(args.output/'namba_unavailable.tsv',unavailable)
    plot(args.output,rows,external)
    if specificity:plot_glycaemic_specificity(args.output,specificity)
    receipt=dict(script_sha256=file_sha256(__file__),source_sha256=sources,
        literature_sha256=file_sha256(args.namba_table),paired_blocks=200,genotype_traversals=0,
        uncertainty='full-point delta with corrected paired coefficient covariance',
        exploratory=True,pairs=len(lookup),external_entries=len(external),
        files={p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()})
    with (args.output/'COMPLETE.json').open('x') as f:json.dump(receipt,f,indent=2)


def plot_glycaemic_specificity(output,rows):
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['quantity']=='orthogonal_rg']
    fig,ax=plt.subplots(figsize=(8,3.5))
    for i,row in enumerate(selected):
        if np.isfinite(row['estimate']) and np.isfinite(row['paired_delta_se']):
            ax.errorbar(row['estimate'],i,xerr=1.96*row['paired_delta_se'],fmt='o',color='#0072B2',capsize=3)
        else:ax.text(.03,i,'undefined',transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(selected)),[LABELS[r['anchor']] for r in selected]);ax.invert_yaxis()
    ax.axvline(0,color='#999999',lw=.8);ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Orthogonal-response rg with HbA1c − rg with glucose')
    ax.set_title('Does response sharing differ between HbA1c and glucose?',loc='left')
    fig.text(.02,.015,'Paired nominal 95% delta intervals; covariance across both fitted pairs is retained.\n'
        'Exploratory common-bin contrast. A difference does not identify erythrocyte, treatment or causal mechanisms.',fontsize=8)
    fig.tight_layout(rect=(0,.16,1,1))
    for suffix in ('pdf','png'):fig.savefig(output/f'glycaemic_specificity.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(output,rows,external):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    colors={'baseline_rg':'#666666','orthogonal_rg':'#0072B2','response_rg':'#009E73'}
    def save(fig,name):
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    pair_rows=[r for r in rows if r['quantity']=='baseline_rg']
    pair_rows.sort(key=lambda r:(r['trait_x'] in ('height_raw','platelet_count_raw') or
        r['trait_y'] in ('height_raw','platelet_count_raw'),list(LABELS).index(r['trait_x']),list(LABELS).index(r['trait_y'])))
    pairs=[(r['trait_x'],r['trait_y']) for r in pair_rows]
    index={(r['trait_x'],r['trait_y'],r['quantity'],r['entry']):r for r in rows}
    labels=[LABELS[x]+' – '+LABELS[y] for x,y in pairs]
    fig,axes=plt.subplots(1,2,figsize=(10,max(6,len(pairs)*.25)),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    for i,(x,y) in enumerate(pairs):
        for name,offset in [('baseline_rg',-.13),('orthogonal_rg',.13)]:
            r=index[x,y,name,0]
            axes[0].errorbar(r['estimate'],i+offset,xerr=1.96*r['delta_se'],fmt='o',ms=3,
                color=colors[name],lw=.8,capsize=1.5)
        r=index[x,y,'orthogonal_minus_baseline_rg',0]
        axes[1].errorbar(r['estimate'],i,xerr=1.96*r['delta_se'],fmt='o',ms=3,
            color='#0072B2' if r['fdr']<.05 else '#aaaaaa',lw=.8,capsize=1.5)
    axes[0].set_yticks(range(len(pairs)),labels,fontsize=8);axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0,color='#aaaaaa',lw=.7);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('Genetic correlation');axes[1].set_xlabel('Orthogonal-response rg − baseline rg')
    axes[0].set_title('Baseline and aggregate response sharing',loc='left',pad=28);axes[1].set_title('Paired difference',loc='left',pad=28)
    axes[0].legend(handles=[Line2D([],[],color=colors[k],marker='o',ls='',label=l) for k,l in
        [('baseline_rg','Baseline'),('orthogonal_rg','Baseline-orthogonal response')]],
        loc='lower left',bbox_to_anchor=(0,1.002),ncol=2,fontsize=8,frameon=False)
    fig.text(.02,-.045,'Exploratory common-bin fits. Bars: nominal 95% delta intervals.\n'
        'Blue paired differences: BH FDR < 0.05 within the recorded pilot, external or glycaemic analysis family.',fontsize=8)
    fig.tight_layout();save(fig,'baseline_and_orthogonal_response')
    # Same-exposure response slopes are the immediate single-exposure analog.
    fig,axes=plt.subplots(1,4,figsize=(14,max(6,len(pairs)*.25)),sharey=True)
    for e,title in enumerate(['Age','BMI','Sex','Ever smoking']):
        ax=axes[e]
        for i,(x,y) in enumerate(pairs):
            r=index[x,y,'response_rg',e];base=index[x,y,'baseline_rg',0]
            if np.isfinite(r['estimate']) and np.isfinite(r['delta_se']):
                value=r['estimate'];low,high=value-1.96*r['delta_se'],value+1.96*r['delta_se']
                left,right=-1.2,1.2
                if max(left,low)<=min(right,high):
                    ax.hlines(i,max(left,low),min(right,high),color='#009E73',lw=.8)
                if left<=value<=right:
                    ax.plot(value,i,'o',ms=3,color='#009E73')
                else:
                    boundary=left if value<left else right
                    ax.plot(boundary,i,'<' if value<left else '>',ms=5,color='#009E73',clip_on=False)
                    ax.text(boundary+(.06 if value<left else -.06),i-.18,f'{value:.2f}',
                        ha='left' if value<left else 'right',fontsize=6,color='#007554')
                if low<left:ax.plot(left,i,'<',ms=4,color='#009E73',clip_on=False)
                if high>right:ax.plot(right,i,'>',ms=4,color='#009E73',clip_on=False)
                ax.plot(base['estimate'],i,'|',color='#777777',ms=7)
            else:ax.text(.02,i,'undefined',transform=ax.get_yaxis_transform(),fontsize=6,color='#777777')
        ax.axvline(0,color='#bbbbbb',lw=.7);ax.set_title(title);ax.set_xlabel('Response-slope rg');ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(-1.2,1.2);ax.set_xticks([-1,-.5,0,.5,1]);ax.grid(axis='x',alpha=.12)
    axes[0].set_yticks(range(len(pairs)),labels,fontsize=8);axes[0].invert_yaxis()
    fig.text(.02,-.025,'Green: same-exposure slope correlation (no baseline orthogonalization); grey tick: baseline rg.\n'
        'Nominal 95% delta intervals. Triangles mark off-scale intervals or estimates; off-scale point values are printed.\n'
        'Undefined denominators are labeled. Full untruncated estimates and intervals are in the source table.',fontsize=8)
    fig.tight_layout();save(fig,'single_exposure_response_correlations')
    pilot=list(LABELS)[:8];matrix=np.full((8,8),np.nan)
    for (x,y) in pairs:
        if x in pilot and y in pilot:
            i,j=pilot.index(x),pilot.index(y)
            matrix[i,j]=index[x,y,'age_bmi_rg',0]['estimate']
            matrix[j,i]=index[x,y,'bmi_age_rg',0]['estimate']
    fig,ax=plt.subplots(figsize=(9,8));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
    # Unconstrained moment ratios outside [-1,1] are retained in the table,
    # but must not look like stronger biological correlations in a heatmap.
    displayed=np.where(abs(matrix)<=1,matrix,np.nan)
    im=ax.imshow(displayed,cmap=cmap,vmin=-1,vmax=1)
    for i in range(8):
        for j in range(8):
            if np.isfinite(matrix[i,j]):
                label=f'{matrix[i,j]:.2f}'
                if abs(matrix[i,j])>1:
                    label='unstable\n'+label
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='#cccccc',hatch='///',lw=0))
                ax.text(j,i,label,ha='center',va='center',fontsize=8)
    ax.set_xticks(range(8),[LABELS[t] for t in pilot],rotation=45,ha='right')
    ax.set_yticks(range(8),[LABELS[t] for t in pilot])
    ax.set_xlabel('Trait with BMI response');ax.set_ylabel('Trait with age response')
    ax.set_title('Age response of one trait versus BMI response of another\nBoth responses orthogonal to their own baseline',loc='left')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.04,label='Response correlation')
    fig.subplots_adjust(left=.20,right=.88,bottom=.25,top=.86)
    fig.text(.02,.015,'Ordered cross-exposure correlations need not be symmetric. Grey diagonal: within-trait entries omitted.\n'
        'Hatched cells: inadmissible moment ratios outside [−1, 1]; do not interpret these as correlations.\n'
        'Exploratory common-bin estimates; full intervals and paired asymmetry tests are in biological_contrasts.tsv.',fontsize=8)
    save(fig,'age_bmi_ordered_response')
    if external:
        fig,ax=plt.subplots(figsize=(8,max(3,len(external)*.5)))
        for i,row in enumerate(external):
            for prefix,offset,color in [('namba',-.12,'#D55E00'),('summit',.12,'#0072B2')]:
                if np.isfinite(row[prefix+'_rg']) and np.isfinite(row[prefix+'_se']):
                    ax.errorbar(row[prefix+'_rg'],i+offset,xerr=1.96*row[prefix+'_se'],fmt='o',ms=4,color=color,capsize=2)
        ax.set_yticks(range(len(external)),[f"{LABELS[r['trait_x']]} – {LABELS[r['trait_y']]} | {r['exposure']} | {r['cohort']}" for r in external])
        ax.invert_yaxis();ax.axvline(0,color='#aaaaaa',lw=.7);ax.set_xlabel('Single-exposure response rg, nominal 95% interval')
        ax.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for c,l in [('#D55E00','Namba et al.'),('#0072B2','SUMMIT')]],frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
        ax.set_title('External single-exposure comparison',loc='left',pad=32);ax.spines[['top','right']].set_visible(False)
        fig.text(.01,-.10,'UKB comparisons overlap samples; BBJ differs in ancestry and ascertainment. Exposure sets, transformations and SNP panels differ.\n'
            'Namba Table S22 contains significant entries only. This is a concordance check, not an independent replication test.',fontsize=8)
        save(fig,'namba_single_exposure_comparison')


if __name__=='__main__':main()
