#!/usr/bin/env python3
"""Conditional null checks without fitted risks or liability approximations."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from compare_official_jackknife import null_expected_variance
from validate_pcgc import generate
from summit.sumstats.binary import prepare_binary_risk

URL = 'https://raw.githubusercontent.com/omerwe/PCGCs/fdc5089f485fe25c04a8972665fee4216570764f/deprecated/pcgcs_direct.py'
SHA = 'e84fcb690c4376b177672d9c28a68b48a746067c0bb8ee5b7c79a65353d8f26a'


def direct_function():
    source=urlopen(URL,timeout=30).read()
    if hashlib.sha256(source).hexdigest()!=SHA:
        raise RuntimeError('direct upstream code changed')
    # The old module is Python 2, but this isolated function is Python 3 syntax.
    selected=source.decode().split('def pcgc_jackknife_sig2g(',1)[1].split('def pcgc_jackknife_corr(',1)[0]
    namespace=dict(np=np,xrange=range)
    exec(compile('def pcgc_jackknife_sig2g('+selected,URL,'exec'),namespace)
    return namespace['pcgc_jackknife_sig2g']


def direct_check():
    rng=np.random.default_rng(620)
    n,m=40,120
    x=rng.normal(size=(n,m)); d=rng.uniform(.4,1.4,n); p=rng.uniform(.1,.9,n)
    y=rng.binomial(1,p); z=(y-p)/np.sqrt(p*(1-p)); v=d*z
    kernel=x@x.T/m; np.fill_diagonal(kernel,0)
    B=kernel*d[:,None]*d[None,:]
    h=np.sum(B*B); b=z@B@z
    hminus=h-2*np.sum(B*B,axis=1)
    loo=(b-2*z*(B@z))/hminus
    expected=np.sqrt((n-1)*np.var(loo))
    actual=direct_function()(x,z,b,h,u0=d,u1=np.zeros(n),window_size=7)
    np.testing.assert_allclose(actual,expected,rtol=1e-13)
    # Exact null covariance of the person-deleted quadratic forms. For i!=j,
    # intersect the retained ordered-pair sets; restore (i,j),(j,i) once.
    row=np.sum(B*B,axis=1)
    overlap=h-2*row[:,None]-2*row[None,:]+2*B*B
    np.fill_diagonal(overlap,hminus)
    cov=2*overlap/np.outer(hminus,hminus)
    expected_variance=(n-1)/n*(np.trace(cov)-cov.sum()/n)
    return dict(samples=n,variants=m,upstream_url=URL,upstream_sha256=SHA,
                independent_se=float(expected),upstream_se=float(actual),
                exact_null_variance=2/h,expected_person_jackknife_variance=float(expected_variance),
                expected_variance_ratio=float(expected_variance/(2/h)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=4000)
    parser.add_argument('--variants',type=int,default=4000)
    args=parser.parse_args()
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(41,1,0,0,0) or libc.prctl(42,0,0,0,0)!=1:
        raise RuntimeError('THP guard failed')
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),Path(__file__).with_name('compare_official_jackknife.py'),Path(__file__).with_name('validate_pcgc.py'),root/'src/summit/sumstats/binary.py']
    result=dict(source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                null_design='independent Bernoulli conditional on X and fixed risk vectors; not fixed case count',
                direct=direct_check(),snp=[])
    for seed in (827020,827065):
        data=generate(seed,'S7',args.samples,args.variants,1,40)
        risk=prepare_binary_risk(data['y'],.1,population_risk=data['k'],covariate_variance=1.)
        for blocks in (50,100):
            result['snp'].append(dict(seed=seed,blocks=blocks,**null_expected_variance(data['x'],data['annotations'],risk,blocks)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False); handle.write('\n')


if __name__=='__main__':
    main()
