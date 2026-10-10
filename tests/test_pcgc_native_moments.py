"""Independent checks of the production PCGC moment and trace kernels."""
import numpy as np
import pytest

from prediction_helpers import prediction_threads
from summit import gxeldcore
from summit.ldscore.matrix_products import MatrixProducts
from summit.pcgc.architecture import ArchitectureCovariance,architecture_trace_covariance
from summit.pcgc.gxe import context_pairs
from summit.pcgc.sampling import build_sampling_moments,center_strata,sampling_panel_bytes
from test_pcgc_gxe_sampling import fixture


@pytest.mark.parametrize('overlapping',[False,True])
def test_native_relatedness_accumulates_variant_tiles(overlapping):
    rng=np.random.default_rng(4447)
    n,m,l,k=31,37,9,3
    x=rng.normal(size=(n,m))
    annotations=(rng.uniform(.1,1,(m,k)) if overlapping else np.eye(k)[np.arange(m)%k])
    annotations/=annotations.sum(0)
    other=rng.integers(0,n-1,(n,l)); other+=other>=np.arange(n)[:,None]
    expected=np.stack([(x*ann)@x.T for ann in annotations.T],axis=-1)[np.arange(n)[:,None],other]
    result=np.zeros((n*l,k))
    for first in range(0,m,11):
        partition=gxeldcore.pcgc_accumulate_relatedness(np.ascontiguousarray(x[:,first:first+11]),
            np.asfortranarray(annotations[first:first+11]),other,result,prediction_threads())
        assert partition is (not overlapping)
    np.testing.assert_allclose(result.reshape(n,l,k),expected,rtol=3e-13,atol=5e-16)
    invalid=other.copy(); invalid[-1,0]=-1
    saved=result.copy()
    with pytest.raises(ValueError,match='partners'):
        gxeldcore.pcgc_accumulate_relatedness(x,np.asfortranarray(annotations),invalid,result,prediction_threads())
    np.testing.assert_array_equal(result,saved)
    shared=np.zeros(max(n*m,n*l*k))
    with pytest.raises(ValueError,match='overlap'):
        gxeldcore.pcgc_accumulate_relatedness(shared[:n*m].reshape(n,m),
            np.asfortranarray(annotations),other,shared[:n*l*k].reshape(n*l,k),prediction_threads())


@pytest.mark.parametrize('context_only',[False,True])
@pytest.mark.parametrize('importance',[False,True])
def test_native_pair_panels_match_explicit_products(context_only,importance):
    rng=np.random.default_rng(81933)
    n,l,k,q=21,5,2,3
    pairs=np.asarray(context_pairs(q),dtype=np.int64)
    p=len(pairs); c=k*p
    features=rng.normal(size=(n,q))
    relatedness=rng.normal(size=(n,l,k))
    response=rng.normal(size=n)
    other=rng.integers(0,n-1,(n,l)); other+=other>=np.arange(n)[:,None]
    probability=rng.uniform(.5,2,n); probability/=probability.sum()
    leverage=rng.uniform(.2,1.8,n)
    upper=np.column_stack(np.triu_indices(p if context_only else c)).astype(np.int64)
    first,last=2,15
    budget=sampling_panel_bytes(last-first,l,c,p,len(upper))
    args=(relatedness.reshape(n*l,k),other,features,response,
        probability if importance else np.empty(0),leverage,pairs,upper,first,last,context_only,budget,prediction_threads())
    actual=list(map(np.asarray,gxeldcore.pcgc_pair_panels(*args)))
    neighbors=other[first:last]
    f=features[first:last]
    context=np.stack([f[:,u,None]*features[neighbors,v] if u==v else
        f[:,u,None]*features[neighbors,v]+f[:,v,None]*features[neighbors,u] for u,v in pairs],axis=-1)
    kernels=(relatedness[first:last,:,:,None]*context[:,:,None,:]).reshape(-1,c)
    value=context.reshape(-1,p) if context_only else kernels
    square=value[:,upper[:,0]]*value[:,upper[:,1]]
    weight=((1-probability[first:last,None])/probability[neighbors] if importance else
        np.full((last-first,l),n-1.)).reshape(-1,1)
    score=kernels*(response[first:last,None]*response[neighbors]).reshape(-1,1)
    expected=[score,score*weight,square,square*weight,
        square*weight**2*np.repeat(leverage[first:last],l)[:,None],
        (square*weight).reshape(last-first,l,-1).mean(1)]
    for left,right in zip(actual,expected):
        np.testing.assert_allclose(left,right,rtol=3e-14,atol=2e-14)
    with pytest.raises(RuntimeError,match='workspace'):
        gxeldcore.pcgc_pair_panels(*args[:-2],1,args[-1])
    invalid=other.copy(); invalid[first,0]=first
    with pytest.raises(ValueError,match='partners'):
        gxeldcore.pcgc_pair_panels(args[0],invalid,*args[2:])


def test_native_stratum_centering_matches_direct_means():
    rng=np.random.default_rng(171)
    values=np.asfortranarray(rng.normal(size=(19,7)))
    cases=np.arange(19)%3==0
    expected=center_strata(values,cases)
    gxeldcore.pcgc_center_strata(values,cases.astype(np.uint8),prediction_threads())
    np.testing.assert_allclose(values,expected,rtol=2e-14,atol=2e-15)


@pytest.mark.parametrize('q',[1,2,3])
def test_native_architecture_trace_and_scalar_polynomial(q):
    rng=np.random.default_rng(7791+q)
    k,probes=2,3
    c=k*q*(q+1)//2; d=k*q*probes
    left=rng.normal(size=(c,d,d)); right=rng.normal(size=left.shape)
    omega=rng.normal(size=(k,q,q)); omega=(omega+omega.transpose(0,2,1))/2
    backend=MatrixProducts(native=True,threads=prediction_threads())
    evaluator=ArchitectureCovariance(left,right,q,probes,backend)
    np.testing.assert_allclose(evaluator.covariance(omega),
        architecture_trace_covariance(omega,left,right,probes),rtol=3e-12,atol=2e-12)
    rows=rng.normal(size=(5,c))
    directions=rng.normal(size=(5,k,q,q))
    directions=(directions+directions.transpose(0,1,3,2))/2
    coefficients=evaluator.scalar_polynomials(omega,rows,directions)
    for step in (-1.,0.,.31,2.):
        expected=[r@architecture_trace_covariance(omega+step*v,left,right,probes)@r
                  for r,v in zip(rows,directions)]
        np.testing.assert_allclose(coefficients@np.array([1.,step,step*step]),expected,rtol=4e-12,atol=2e-10)


def test_native_sampling_direction_polynomial():
    from test_pcgc_gxe_sampling import panel
    data,_,base=fixture(n=24,m=19)
    n=len(base)
    partners=np.array([np.delete(np.arange(n),i) for i in range(n)])
    moments=panel(data,base,partners,False)
    theta=np.array([.21,-.1,.08])
    rows=np.random.default_rng(153).normal(size=(3,7))
    products=MatrixProducts(native=True,threads=prediction_threads())
    terms=moments.component_direction_terms(theta,rows,products)
    for d,row in enumerate(rows):
        base=float(row@moments.covariance(theta)@row)
        for step in (-1.,.1,2.):
            changed=theta.copy(); changed[d]+=step
            expected=row@moments.covariance(changed)@row
            np.testing.assert_allclose(base+step*terms[d,0]+step**2*terms[d,1],expected,rtol=3e-12,atol=2e-15)


def test_missing_native_sampling_kernel_fails_before_genotype_reads(monkeypatch):
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
    from summit.pcgc.gxe import prepare_gxe_moments
    data,_,_=fixture()
    operator=ArraySequentialGenotypeOperator(data['x'])
    monkeypatch.delattr(gxeldcore,'pcgc_accumulate_relatedness')
    with pytest.raises(RuntimeError,match='rebuild SUMMIT'):
        prepare_gxe_moments(operator,np.ones((operator.num_variants,1)),data['risk'],data['contexts'],
            liability_sd=data['sd'],sampling_partners=8,threads=prediction_threads())
    assert operator.observed_passes==0


@pytest.mark.parametrize('native',[False,True])
def test_external_sampling_preserves_directional_annotation_gram(native):
    data,_,_=fixture(n=24,m=19)
    x=data.pop('x'); n=len(x)
    bases=np.stack((x@x.T/x.shape[1],x[:,:7]@x[:,:7].T/7),axis=-1)
    partners=np.array([np.delete(np.arange(n),i) for i in range(n)])
    features=data['features']; response=data['response']
    pairs=context_pairs(2); p=len(pairs); c=2*p
    context=np.stack([features[:,u,None]*features[partners,v] if u==v else
        features[:,u,None]*features[partners,v]+features[:,v,None]*features[partners,u] for u,v in pairs],axis=-1)
    base=bases[np.arange(n)[:,None],partners]
    kernels=(base[:,:,:,None]*context[:,:,None,:]).reshape(n,n-1,c)
    actions=(kernels*response[partners,None]).sum(1)
    gram=np.array([[.8,.2],[.3,.6]])
    square=np.einsum('ab,nlp,nlr->nlapbr',gram,context,context).reshape(n,n-1,c,c)
    execution={}
    actual=build_sampling_moments(**{key:value for key,value in data.items()
        if key not in ('kernel_actions','genotype_diagonal')},kernel_actions=actions,
        genotype_diagonal=np.stack([np.diag(bases[:,:,j]) for j in range(2)],axis=1),
        pair_kernels=base,partners=partners,sampled=False,reference_annotation_gram=gram,
        native=native,threads=prediction_threads(),execution=execution)
    for theta in (np.zeros(c),np.linspace(-.1,.2,c)):
        residual=kernels*(response[:,None]*response[partners])[:,:,None]-square@theta
        influence=2*(response[:,None]*actions-square.sum(1)@theta)/float(n*(n-1))
        influence=center_strata(influence,data['risk'].z>0)
        pair=2*np.einsum('nli,nlj->ij',residual,residual)/float(n*(n-1))**2
        np.testing.assert_allclose(actual.covariance(theta)[:c,:c],influence.T@influence-pair,rtol=5e-12,atol=3e-15)
        np.testing.assert_allclose(actual.pair_covariance(theta),pair,rtol=5e-12,atol=3e-15)
    assert execution['pair_feature_columns']==p*(p+1)//2
    assert execution['native']==native


def test_execution_evidence_is_bounded_and_retains_failures():
    from summit.ldscore.matrix_products import NativeExecutionEvidence
    class Runtime:
        def __init__(self):
            self.records=[]
            self.count=0
            self.failed=0
            self.dropped=0
        def gemm_telemetry_status(self):
            return dict(capacity=4,buffered_records=len(self.records),
                        captured_records=self.count,dropped_records=self.dropped)
        def native_gemm_output_numa_evidence_status(self):
            return dict(failed_calls=self.failed,dropped_records=0)
        def consume_gemm_telemetry(self):
            result,self.records=self.records,[]
            return result
    runtime=Runtime()
    evidence=NativeExecutionEvidence(runtime)
    evidence.record_limit=3
    for j in range(29):
        runtime.records.append(dict(operation='NN' if j%2 else 'TN',
                                    wall_seconds=.25,process_cpu_seconds=.5,flop_count=8))
        runtime.count+=1
        assert len(runtime.records)<=4
        evidence.check()
    result=evidence.finish()
    assert result['gemm_record_count']==29
    assert result['gemm_records_summarized']==26
    assert len(result['gemm_records'])==3
    assert sum(v['calls'] for v in result['gemm_summary'].values())==29
    assert sum(v['flop_count'] for v in result['gemm_summary'].values())==232
    assert evidence.finish()['gemm_record_count']==0
    runtime.failed=1
    with pytest.raises(RuntimeError,match='NUMA'):
        evidence.check()
    runtime.failed=0
    runtime.dropped=1
    with pytest.raises(RuntimeError,match='overflowed'):
        evidence.finish()
