"""Plot an authenticated conditional assessment without changing its decisions.

Observed outcome counts and generating-law conditional diagnostics are distinct.
The latter do not increase the number of independently fitted learners.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def intervals(ax, positions, values, bounds, *, offset, label, color):
    values=np.asarray(values,float);bounds=np.asarray(bounds,float)
    ax.errorbar(values,np.asarray(positions)+offset,
        xerr=np.maximum(0,np.vstack([values-bounds[:,0],bounds[:,1]-values])),
        fmt="o",markersize=4,capsize=2,label=label,color=color)


def label(group):
    return group["id"].replace("chr12_","12 / ").replace("chr16_","16 / ").replace("_"," ")


def save(fig,root,name):
    fig.savefig(root/(name+".png"),dpi=160,bbox_inches="tight")
    fig.savefig(root/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def report(assessment,root,resource_path=None):
    raw=assessment.read_bytes();data=json.loads(raw)
    if data["method"]!="conditional_polygenic_mean_tangent_v1":
        raise ValueError("conditional mean assessment required")
    groups=data["groups"]
    complete=[g for g in groups if g["records"]]
    for g in complete:
        if len({r["biological_null"] for r in g["records"]})!=1:
            raise ValueError("do not pool null and alternative outcomes")
    nulls=[g for g in complete if g["records"][0]["biological_null"]]
    alternatives=[g for g in complete if not g["records"][0]["biological_null"]]
    root.mkdir(parents=True,exist_ok=False)
    scheduled=sum(g["summary"]["scheduled"] for g in groups)
    completed=sum(g["summary"]["completed"] for g in groups)
    phases=sorted({g["phase"] for g in groups})
    footer=(f"{', '.join(phases)}; {completed}/{scheduled} scheduled fits complete. "
        "Conditional diagnostics add zero learners. No final qualification.")
    rows=[]
    for g in groups:
        s=g["summary"]
        row=dict(group=g["id"],biological_null=(g["records"][0]["biological_null"] if g["records"] else None),phase=g["phase"],scheduled=s["scheduled"],completed=s["completed"],
            missing=s["unexecuted_or_incomplete"],unsupported=s["unsupported"],
            bias=s.get("bias"),error_sd=s.get("error_sd"),mean_se=s.get("mean_se"),
            rms_se=s.get("rms_se"),coverage=s.get("coverage"))
        for alpha in ("0.05","0.005"):
            stat=s.get(alpha,{})
            for key in ("rejected","observed_rate","analytic_conditional_rate","material_size_screen_excludes_tolerance"):
                row[alpha+"_"+key]=(None if key=="material_size_screen_excludes_tolerance" and not row["biological_null"] else stat.get(key))
        rows.append(row)
    with (root/"summary.csv").open("x") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axs=plt.subplots(1,3,figsize=(17,6),layout="constrained")
    for ax,alpha in zip(axs[:2],("0.05","0.005")):
        y=np.arange(len(nulls));stats=[g["summary"][alpha] for g in nulls]
        intervals(ax,y,[s["observed_rate"] for s in stats],[s["observed_mc95"] for s in stats],
            offset=-.12,label="Observed, binomial MC95",color="#455a64")
        intervals(ax,y,[s["analytic_conditional_rate"] for s in stats],[s["learner_bootstrap_mc95"] for s in stats],
            offset=.12,label="Conditional, learner-bootstrap MC95",color="#1565c0")
        ax.axvline(float(alpha),color="black",linestyle="--",linewidth=1,label="Nominal")
        ax.axvline(.075 if alpha=="0.05" else .01,color="#c62828",linestyle=":",linewidth=1,label="Material-inflation screen")
        ax.set(yticks=y,yticklabels=[label(g) for g in nulls],xlabel="Biological-null rejection probability",title="Null, nominal "+alpha,xlim=(-.015,.52))
        ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
    ax=axs[2];y=np.arange(len(alternatives))
    for method,offset,color in (("learned",-.18,"#1565c0"),("burden",0,"#6a1b9a"),("oracle",.18,"#2e7d32")):
        selected=[];positions=[]
        for k,g in enumerate(alternatives):
            s=g["summary"] if method=="learned" else g["summary"]["comparisons"][method]
            if s.get("completed"):
                selected.append(s["0.005"]);positions.append(k)
        intervals(ax,positions,[s["observed_rate"] for s in selected],[s["observed_mc95"] for s in selected],
            offset=offset,label=method,color=color)
    ax.set(yticks=y,yticklabels=[label(g) for g in alternatives],xlabel="Observed power, binomial MC95",title="Matched alternatives, nominal .005",xlim=(-.02,1.04))
    ax.invert_yaxis();ax.grid(axis="x",alpha=.15);ax.legend(loc="lower left",fontsize=8)
    axs[0].legend(loc="lower right",fontsize=7)
    fig.suptitle(footer,fontsize=11)
    save(fig,root,"calibration_power")
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    ax=axs[0];y=np.arange(len(complete));stats=[g["summary"] for g in complete]
    intervals(ax,y,[s["coverage"] for s in stats],[s["coverage_mc95"] for s in stats],offset=-.12,
        label="Observed, binomial MC95",color="#455a64")
    coverage=[(k,s["0.05"]) for k,s in enumerate(stats) if "analytic_conditional_coverage" in s["0.05"]]
    intervals(ax,[k for k,s in coverage],[s["analytic_conditional_coverage"] for k,s in coverage],
        [s["coverage_learner_bootstrap_mc95"] for k,s in coverage],offset=.12,
        label="Conditional, learner-bootstrap MC95",color="#1565c0")
    ax.axvline(.95,color="black",linestyle="--",linewidth=1)
    ax.set(yticks=y,yticklabels=[label(g)+(" *" if "analytic_conditional_coverage" not in g["summary"]["0.05"] else "") for g in complete],xlabel="Coverage of each learner's interaction target",title="95% coefficient intervals",xlim=(.45,1.02))
    ax.invert_yaxis();ax.legend(loc="lower left",fontsize=8);ax.grid(axis="x",alpha=.15)
    ax=axs[1]
    if resource_path:
        resource=json.loads(resource_path.read_text())["hoffman"]
        records=resource["tasks"]
        y=np.arange(len(records));hours=[r["seconds"]/3600 for r in records]
        ax.barh(y,hours,color="#546e7a")
        for k,r in enumerate(records):
            ax.text(hours[k]+.1,k,f'{r["peak_rss_bytes"]/2**30:.2f} GiB; {r["cpu_seconds"]/r["seconds"]:.2f} CPU',va="center",fontsize=8)
        ax.set(yticks=y,yticklabels=["task "+str(r["task"]) for r in records],xlabel="Measured wall hours per twelve-trait batch",title="Hoffman 15047065: 8 reserved cores / task",xlim=(0,max(hours)+6))
        ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
    else:
        ax.axis("off");ax.text(.1,.5,"No resource record supplied")
    fig.suptitle(footer+"\n* Conditional coverage absent from older diagnostic records.",fontsize=11);save(fig,root,"coverage_resources")
    receipt=dict(assessment=str(assessment.resolve()),assessment_sha256=hashlib.sha256(raw).hexdigest(),
        method=data["method"],phases=phases,scheduled=scheduled,completed=completed,
        interpretation="Development evidence, including fixed-architecture stresses. Conditional rates use known generating laws only for diagnosis, never production fitting; no additional learners. Missing settings remain in summary.csv. Confidence intervals do not establish universal calibration.")
    if resource_path:
        receipt.update(resources=str(resource_path.resolve()),resources_sha256=hashlib.sha256(resource_path.read_bytes()).hexdigest())
    (root/"figure_sources.json").write_text(json.dumps(receipt,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("assessment",type=Path)
    p.add_argument("--resources",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();report(a.assessment,a.out,a.resources)


if __name__=="__main__":main()
