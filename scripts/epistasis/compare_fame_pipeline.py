"""Run the entire unmodified pinned preprocessing pipeline on bounded PLINK inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import numpy as np
from bed_reader import to_bed
from sklearn.linear_model import LinearRegression
from scripts.epistasis.compare_fame import REVISION,printed_equations
from summit.epistasis.oracle import selected_kernels,dense_summary
from summit.epistasis.cli import _jsonable


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if subprocess.check_output(["git","-C",str(args.source),"rev-parse","HEAD"],text=True).strip()!=REVISION:
        raise ValueError("incorrect upstream revision")
    if subprocess.check_output(["git","-C",str(args.source),"diff","--name-only"],text=True).strip():
        raise ValueError("upstream tracked modifications")
    args.out.mkdir(parents=True,exist_ok=False);records=[]
    for case,(n,m,ld) in enumerate(((192,128,False),(384,128,False),(192,384,False),(192,128,True))):
        rng=np.random.default_rng(78301+case)
        raw=rng.binomial(2,.3,(n,m)).astype(float)
        if ld:
            for j in range(1,m):
                if j%8:raw[:,j]=np.where(rng.random(n)<.7,raw[:,j-1],raw[:,j])
        mean=raw.mean(axis=0);x=(raw-mean)/np.sqrt(mean*(1-mean/2))
        cov=np.column_stack([np.ones(n),rng.normal(size=n)])
        y=x@rng.normal(size=m)/np.sqrt(m)+.25*x[:,2]*x[:,15]+rng.normal(size=n)+cov[:,1]
        with tempfile.TemporaryDirectory(prefix="summit-fame-pipeline-") as tmp:
            root=Path(tmp);ids=list(map(str,range(n)))
            to_bed(root/"input.bed",raw,properties=dict(fid=ids,iid=ids,sid=[f"v{i}" for i in range(m)],
                chromosome=["1"]*m,bp_position=np.arange(1,m+1),allele_1=["A"]*m,allele_2=["G"]*m))
            for script in ("generate_ld_annotations.py","linear_regression_annotation.py"):
                (root/script).symlink_to(args.source.resolve()/"pipeline"/script)
            (root/"trait.pheno").write_text("FID IID pheno\n"+"".join(f"{i} {i} {y[i]:.17g}\n" for i in range(n)))
            (root/"trait.covar").write_text("FID IID intercept c\n"+"".join(f"{i} {i} 1 {cov[i,1]:.17g}\n" for i in range(n)))
            (root/"pairs.txt").write_text("trait v2\n")
            (root/"ld.txt").write_text(f"chr start stop\n1 1 8\n1 9 {m}\n")
            cmd=["bash",str(args.source.resolve()/"pipeline/run_fame_pipeline.sh"),"0",str(root/"input"),str(root),
                 str(root/"pairs.txt"),str(root/"ld.txt"),str(args.binary.resolve())]
            started=time.perf_counter()
            run=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=120)
            if run.returncode or not (root/"results/trait-v2.res.out.txt").exists():
                raise RuntimeError(f"upstream pipeline failed: {run.stdout[-2000:]} {run.stderr[-2000:]}")
            bins=np.loadtxt(root/"annot/trait-v2.annot")
            local=np.column_stack([np.ones(n),raw[:,:8]])
            expected_residual=y-local@np.linalg.lstsq(local,y,rcond=None)[0]
            residual=np.loadtxt(root/"residualized_pheno/trait-v2.pheno",skiprows=1)[:,2]
            # bed_reader.read defaults to float32; sklearn fits/predicts in that
            # dtype. Preserve, and separately quantify, this upstream choice.
            local32=raw[:,:8].astype(np.float32)
            matched_residual=y-LinearRegression().fit(local32,y).predict(local32)
            yy=residual-cov@np.linalg.lstsq(cov,residual,rcond=None)[0]
            yy=(yy-yy.mean())/yy.std(ddof=1)
            weights=np.column_stack([bins,bins[:,1]])
            kernels,_=selected_kernels(x,np.column_stack([np.ones((n,2)),x[:,2]]),weights,np.empty((n,0)))
            summary=dense_summary(kernels,yy,component_names=("local","other","epistasis","residual"),
                trait_names=("y",),residual_rank=n,metadata={"FAME":REVISION})
            stdout=(root/"results/trait-v2.res.out.full.txt").read_text()
            t,q=printed_equations(stdout,4)
            observed=np.array([[float(a),float(b)] for a,b in re.findall(r"sigma\^2_\d+: ([^ ]+) se: ([^\n]+)",
                (root/"results/trait-v2.res.out.txt").read_text())])
            coefficients=np.linalg.solve(t,summary.rhs[:,0]);inverse=np.linalg.inv(t)
            covariance=inverse@(2*np.einsum("c,acb->ab",coefficients,summary.cubic[0]))@inverse.T
            # Additive-residual entries are set to N upstream, not measured.
            corrected=summary.matrix.copy();corrected[:2,-1]=n;corrected[-1,:2]=n
            records.append(dict(n=n,m=m,ld=ld,seed=78301+case,seconds=time.perf_counter()-started,
                command=[s.replace(str(root),"<temporary>") for s in cmd],pipeline_probes=100,
                preprocessing_residual_max_error=float(np.max(abs(residual-expected_residual))),
                preprocessing_matched_float32_error=float(np.max(abs(residual-matched_residual))),
                upstream_regression_dtype="float32 bed_reader default; phenotype residual stored float64",
                annotation_agreement=bool(np.array_equal(bins,np.column_stack([np.arange(m)<8,np.arange(m)>=8]))),
                rhs_max_error=float(np.max(abs(q-summary.rhs[:,0]))),
                coefficient_max_error_given_upstream_T=float(np.max(abs(coefficients-observed[:,0]))),
                se_max_error_given_upstream_T=float(np.max(abs(np.sqrt(np.diag(covariance))-observed[:,1]))),
                upstream_coefficients=observed[:,0],upstream_se=observed[:,1],
                exact_projected_kernel_rank=int(np.linalg.matrix_rank(kernels[2])),normal_condition=float(np.linalg.cond(summary.matrix)),
                stochastic_and_fixed_trace_relative_error=float(np.linalg.norm(t-summary.matrix)/np.linalg.norm(summary.matrix)),
                stochastic_error_after_additive_trace_convention=float(np.linalg.norm(t-corrected)/np.linalg.norm(corrected)),
                additive_trace_deviation_from_N=(summary.traces[:2]-n).tolist()))
    result=dict(revision=REVISION,binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),records=records,
        interpretation="Actual unmodified shell, annotations, local regression and executable. Time-seeded 100 probes; finite HWE trace shortcut retained only for comparator. Inputs synthetic without missingness; no biobank extrapolation.")
    (args.out/"comparison.json").write_text(json.dumps(_jsonable(result),indent=2,allow_nan=False)+"\n")
    print(json.dumps(_jsonable(result),indent=2))
    if any(r["preprocessing_matched_float32_error"]>1e-10 or not r["annotation_agreement"] for r in records):
        raise AssertionError("pipeline preprocessing discrepancy")


if __name__=="__main__":main()
