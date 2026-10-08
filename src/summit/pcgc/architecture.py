"""Gaussian SNP-effect covariance using four independent participant groups.

Two independent variant-probe families turn fourth-order LD traces into
bilinear products. Group separation removes repeated-person LD terms.
"""
import numpy as np
from summit.ldscore.generalized_gxe_variant import generate_global_variant_probes,native_global_variant_probes


def architecture_workspace_bytes(n,q,k,probes,block_size):
    if type(probes) is not int or probes < 2:
        raise ValueError("architecture probes must be an integer >=2")
    c = k*q*(q+1)//2
    d = k*q*probes
    return 8*(4*n*k*probes+6*block_size*q*d+7*c*d*d+3*d*d+4*n)


class ArchitectureSketch:
    def __init__(self,features,annotations,cases,*,probes,seed,nn,tn,native,threads):
        from .gxe import context_pairs
        n,q = features.shape
        k = annotations.shape[1]
        architecture_workspace_bytes(n,q,k,probes,1)
        self.features,self.annotations = features,annotations
        self.mass = annotations.sum(0)
        self.pairs = context_pairs(q)
        self.probes,self.seed = probes,seed
        self.nn,self.tn,self.native,self.threads = nn,tn,native,threads
        self.group = np.empty(n,dtype=np.int64)
        self.weights = np.empty(n)
        rng = np.random.default_rng(np.random.SeedSequence([seed,0x41524348]))
        for case in (False,True):
            rows = rng.permutation(np.flatnonzero(cases == case))
            if len(rows)<4:
                raise ValueError("architecture inference needs at least four people in each case stratum")
            self.group[rows] = np.arange(len(rows))%4
            for group in range(4):
                selected = rows[self.group[rows] == group]
                self.weights[selected] = len(rows)/n/len(selected)
        self.source = np.zeros((n,k,probes))
        d = k*q*probes
        self.left = np.zeros((k*len(self.pairs),d,d))
        self.right = np.zeros_like(self.left)

    def probe_block(self,start,stop,family):
        options = dict(root_seed=self.seed,namespace='summit.pcgc.architecture.'+family+'.v1')
        if self.native:
            return native_global_variant_probes(np.arange(start,stop),np.arange(self.probes),threads=self.threads,**options)
        return generate_global_variant_probes(np.arange(start,stop),np.arange(self.probes),**options)

    def read_block(self,pass_number,start,stop,x):
        n,q = self.features.shape
        k = self.annotations.shape[1]
        a = self.annotations[start:stop]/self.mass
        if pass_number == 1:
            for family,groups in (('a',(0,3)),('b',(1,2))):
                probe = self.probe_block(start,stop,family)
                selected = np.isin(self.group,groups)
                for annotation in range(k):
                    values = self.nn.matmul(x,np.asfortranarray(np.sqrt(a[:,annotation,None])*probe))
                    self.source[selected,annotation,:] += values[selected]
        elif pass_number == 2:
            d = k*q*self.probes
            cross = np.empty((4,stop-start,q,d))
            source = self.source.reshape(n,-1)
            for group in range(4):
                base_weight = self.weights*(self.group == group)
                for u in range(q):
                    for v in range(q):
                        weights = base_weight*self.features[:,u]*self.features[:,v]
                        value = self.tn.matmul_tn(x,np.asfortranarray(source*weights[:,None])).reshape(stop-start,k,self.probes)
                        cross[group,:,u,:].reshape(stop-start,k,q,self.probes)[:,:,v,:] = value
            for annotation in range(k):
                for p,(u,v) in enumerate(self.pairs):
                    orientations = ((u,v),) if u == v else ((u,v),(v,u))
                    c = annotation*len(self.pairs)+p
                    for left,right in orientations:
                        self.left[c] += self.tn.matmul_tn(np.asfortranarray(cross[0,:,left]),
                            np.asfortranarray(a[:,annotation,None]*cross[1,:,right]))
                        self.right[c] += self.tn.matmul_tn(np.asfortranarray(cross[2,:,left]),
                            np.asfortranarray(a[:,annotation,None]*cross[3,:,right]))


def gaussian_architecture_covariance(theta,q,metric,left,right,probes):
    """Working Gaussian architecture covariance; point estimates stay signed.

    PSD projection is confined to the covariance's working model and uses the
    population context metric, preserving invertible changes of context basis.
    """
    from .gxe import _omega
    omega = _omega(theta,q)
    eig,vec = np.linalg.eigh(metric)
    if eig[0] <= 0 or eig[-1]/eig[0] >= 1e10:
        raise ValueError("architecture covariance requires an identifiable population context metric")
    root = (vec*np.sqrt(eig))@vec.T
    inverse = (vec/np.sqrt(eig))@vec.T
    working = []
    for item in omega:
        e,u = np.linalg.eigh(root@item@root)
        working.append(inverse@((u*np.maximum(e,0))@u.T)@inverse)
    working = np.asarray(working)
    covariance = architecture_trace_covariance(working,left,right,probes)
    diagnostics = dict(model="independent_gaussian_snp_effects_by_annotation",
        working_omega=working.tolist(),working_omega_adjustment_norm=float(np.linalg.norm(working-omega)),
        projection_metric="population_context_second_moment",variant_probes_per_family=probes,
        participant_groups=4,minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()))
    return covariance,diagnostics


def architecture_trace_covariance(omega,left,right,probes):
    """Quadratic trace polynomial, including candidate-parameter evaluation.

    The caller supplies the working covariance. During score inversion it is
    varied along an unrestricted coefficient direction about the fitted PSD
    model. This preserves the quadratic variance polynomial.
    """
    from .gxe import context_pairs
    k,q,_ = omega.shape
    d = k*q*probes
    if left.shape != (k*len(context_pairs(q)),d,d) or right.shape != left.shape:
        raise ValueError("architecture sketch axes disagree")
    transform = np.zeros((d,d))
    for annotation,item in enumerate(omega):
        index = slice(annotation*q*probes,(annotation+1)*q*probes)
        transform[index,index] = np.kron(item,np.eye(probes))
    transformed = transform@left@transform
    covariance = 2*np.einsum('cij,dji->cd',transformed,right)/probes**2
    return (covariance+covariance.T)/2
