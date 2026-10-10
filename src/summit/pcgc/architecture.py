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
    # Context products share a larger protected TN call. Reserve its packed
    # operands/copies and output, as well as the later covariance workspace.
    return 8*(4*n*k*probes+10*block_size*q*d+max(7*c+3,2*c+3*q*q)*d*d+4*n)


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
        self.group_rows = tuple(np.flatnonzero(self.group == g) for g in range(4))
        self.family_rows = (np.flatnonzero(np.isin(self.group,(0,3))),
                            np.flatnonzero(np.isin(self.group,(1,2))))
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
            for family,selected in zip(('a','b'),self.family_rows):
                probe = self.probe_block(start,stop,family)
                for annotation in range(k):
                    active = np.flatnonzero(a[:,annotation])
                    if not len(active):
                        continue
                    # Gather in the decoded block's column order. The transpose
                    # is an F-order view, without a second genotype-sized copy.
                    values = self.nn.matmul(x.T[np.ix_(active,selected)].T,
                        np.asfortranarray(np.sqrt(a[active,annotation,None])*probe[active]))
                    self.source[selected,annotation,:] += values
        elif pass_number == 2:
            d = k*q*self.probes
            cross = np.empty((4,stop-start,q,d))
            source = self.source.reshape(n,-1)
            width = k*self.probes
            # Use the spare N*block gather/protected-input allowance for the
            # grouped RHS and its protected copy. Account for unequal groups;
            # the existing source allowance covers the single-product case.
            largest = max(map(len,self.group_rows))
            batch = min(len(self.pairs),max(1,(n-2*largest)*(stop-start)//(4*largest*width)))
            for group,selected in enumerate(self.group_rows):
                genotype = x.T[np.ix_(np.arange(stop-start),selected)].T
                group_source = np.asfortranarray(source[selected])
                rhs = np.empty((len(selected),batch*width),order='F')
                for first in range(0,len(self.pairs),batch):
                    pairs = self.pairs[first:first+batch]
                    for offset,(u,v) in enumerate(pairs):
                        weights = self.weights[selected]*self.features[selected,u]*self.features[selected,v]
                        np.multiply(group_source,weights[:,None],out=rhs[:,offset*width:(offset+1)*width])
                    values = self.tn.matmul_tn(genotype,rhs[:,:len(pairs)*width])
                    for offset,(u,v) in enumerate(pairs):
                        value = values[:,offset*width:(offset+1)*width].reshape(stop-start,k,self.probes)
                        cross[group,:,u,:].reshape(stop-start,k,q,self.probes)[:,:,v,:] = value
                        if u != v:
                            cross[group,:,v,:].reshape(stop-start,k,q,self.probes)[:,:,u,:] = value
                del genotype,group_source,rhs,values
            for annotation in range(k):
                active = np.flatnonzero(a[:,annotation])
                if not len(active):
                    continue
                for first,second,target in ((0,1,self.left),(2,3,self.right)):
                    # All ordered target-context products have the same SNP
                    # weights. One TN product contains every required block.
                    left = np.asfortranarray(cross[first,active].reshape(len(active),q*d))
                    right = np.asfortranarray(a[active,annotation,None]*cross[second,active].reshape(len(active),q*d))
                    moment = self.tn.matmul_tn(left,right)
                    for p,(u,v) in enumerate(self.pairs):
                        c = annotation*len(self.pairs)+p
                        target[c] += moment[u*d:(u+1)*d,v*d:(v+1)*d]
                        if u != v:
                            target[c] += moment[v*d:(v+1)*d,u*d:(u+1)*d]
                    del left,right,moment


def gaussian_architecture_covariance(theta,q,metric,left,right,probes,*,evaluator=None):
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
    covariance = (architecture_trace_covariance(working,left,right,probes) if evaluator is None
                  else evaluator.covariance(working))
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


class ArchitectureCovariance:
    """Apply the context covariance within annotation/probe blocks.

    The native feature transform costs O(C D² Q), with D=K Q B, and uses
    SUMMIT's protected TN reducer for the trace products. Scalar confidence
    polynomials contract equation weights before touching the sketch axes.
    """
    def __init__(self,left,right,q,probes,products):
        self.left,self.right = np.asarray(left),np.asarray(right)
        self.q,self.probes,self.products = q,probes,products
        c,d,_ = self.left.shape
        if self.right.shape != (c,d,d) or d%(q*probes):
            raise ValueError('architecture sketch axes disagree')
        self.k = d//(q*probes)
        self.flat_left = np.ascontiguousarray(self.left.reshape(c,-1))
        self.flat_right = np.ascontiguousarray(self.right.reshape(c,-1))
        if products.native:
            for name in ('pcgc_architecture_features','pcgc_architecture_polynomials'):
                if not callable(getattr(products.module,name,None)):
                    raise RuntimeError('native extension lacks PCGC architecture kernels; rebuild SUMMIT')

    def covariance(self,omega):
        if not self.products.native:
            return architecture_trace_covariance(omega,self.left,self.right,self.probes)
        function = self.products.module.pcgc_architecture_features
        weight = np.ascontiguousarray(omega).reshape(self.k,-1)
        left = np.asarray(function(weight,self.flat_left,self.q,self.probes,False,self.products.threads))
        right = np.asarray(function(weight,self.flat_right,self.q,self.probes,True,self.products.threads))
        value = 2*self.products.tn(left,right)/self.probes**2
        self.products.drain()
        return (value+value.T)/2

    def scalar_polynomials(self,omega,rows,directions):
        rows = np.asarray(rows)
        directions = np.asarray(directions)
        if directions.shape != (len(rows),self.k,self.q,self.q) or rows.shape[1] != len(self.left):
            raise ValueError('architecture confidence directions disagree')
        if not self.products.native:
            result = []
            for row,direction in zip(rows,directions):
                v0 = float(row@self.covariance(omega)@row)
                vp = float(row@self.covariance(omega+direction)@row)
                vm = float(row@self.covariance(omega-direction)@row)
                result.append([v0,(vp-vm)/2,(vp+vm)/2-v0])
            return np.asarray(result)
        # TN output is F-order D² by T; its transpose is the C-order T by D²
        # input used by the native trace reducer, without another large copy.
        left = self.products.tn(self.flat_left,rows.T).T
        right = self.products.tn(self.flat_right,rows.T).T
        result = np.asarray(self.products.module.pcgc_architecture_polynomials(
            np.ascontiguousarray(omega).reshape(self.k,-1),
            np.ascontiguousarray(directions).reshape(len(rows),-1),left,right,
            self.q,self.probes,self.products.threads))
        self.products.drain()
        return result
