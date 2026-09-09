"""Admissible joint SNP priors and baseline-orthogonal response geometry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._validation import array, psd


def _scale(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def common_scale(omega, kappa):
    with np.errstate(over="raise", invalid="raise"):
        value = psd(omega) * _scale(kappa, "kappa")
    value.setflags(write=False)
    return value


def schur(omega):
    omega = psd(omega)
    if omega[0, 0] <= 0:
        raise ValueError("response geometry requires a positive baseline variance")
    gamma = omega[1:, 0] / omega[0, 0]
    s = omega[1:, 1:] - np.outer(omega[1:, 0], gamma)
    if s.size:
        # Cancellation is relative to the parent covariance, not a tiny S.
        w, v = np.linalg.eigh((s + s.T) / 2)
        if w[0] < -1e-11 * np.max(np.abs(omega)):
            raise ValueError("invalid Schur complement")
        s = (v * np.maximum(w, 0)) @ v.T
    return float(omega[0, 0]), gamma, s


def separate_scales(omega, kappa_a, kappa_h):
    a, gamma, s = schur(omega)
    ka, kh = _scale(kappa_a, "kappa_a"), _scale(kappa_h, "kappa_h")
    anchor = np.r_[1.0, gamma]
    result = ka * a * np.outer(anchor, anchor)
    result[1:, 1:] += kh * s
    return psd(result)


@dataclass(frozen=True)
class ResponseGeometry:
    omega: np.ndarray
    metric: np.ndarray
    reference: str
    anchor: str = "context_zero"

    def __post_init__(self):
        omega = psd(self.omega)
        metric = np.empty((0, 0)) if len(omega) == 1 and np.asarray(self.metric).size == 0 else self.metric
        h = array(metric, name="response metric", ndim=2)
        a, gamma, s = schur(omega)
        if h.shape != s.shape or not self.reference or not self.anchor:
            raise ValueError("metric dimension, reference population or baseline anchor is invalid")
        if h.size:
            h = psd(h, name="response metric")
            w, v = np.linalg.eigh(h)
            if w[0] <= 1e-12 * w[-1]:
                raise ValueError("response metric requires a recorded full-rank support/pruning transform")
            root = (v * np.sqrt(w)) @ v.T
            invroot = (v * (1 / np.sqrt(w))) @ v.T
            spectrum, directions = np.linalg.eigh(root @ s @ root)
            order = np.argsort(-spectrum, kind="stable")
            spectrum, directions = np.maximum(spectrum[order], 0), directions[:, order]
            for j in range(directions.shape[1]):
                if directions[np.argmax(np.abs(directions[:, j])), j] < 0:
                    directions[:, j] *= -1
        else:
            root = invroot = directions = h.copy()
            spectrum = np.empty(0)
        for key, value in dict(omega=omega, metric=h, gamma=gamma, schur=s,
                               root=root, invroot=invroot, eigenvalues=spectrum,
                               directions=directions).items():
            object.__setattr__(self, key, array(value, name=key))

    def decompose(self, components, contexts):
        z = np.asarray(components, dtype=np.float64)
        e = np.asarray(contexts, dtype=np.float64)
        if z.ndim != 2 or e.shape != (len(z), len(self.gamma)) or z.shape[1] != len(self.gamma) + 1:
            raise ValueError("component/context dimensions do not match response geometry")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(e)):
            raise ValueError("nonfinite scores or contexts")
        amplification = (1 + e @ self.gamma) * z[:, 0]
        delta = z[:, 1:] - z[:, :1] * self.gamma
        response = (e @ self.invroot @ self.directions) * (delta @ self.root @ self.directions)
        return amplification, response

    def weights(self, tau):
        tau = _scale(tau, "tau")
        w = self.eigenvalues
        if tau == 0:
            return np.ones_like(w)  # Exact full-score reconstruction, including null modes.
        if not w.size or w[0] == 0:
            return np.zeros_like(w)
        return w / (w + tau * w[0])

    def prior(self, *, tau=0.0, rank=None):
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= rank <= len(self.gamma):
                raise ValueError("rank outside response dimension")
            if tau != 0:
                raise ValueError("choose spectral rank or spectral shrinkage, not both")
            values = self.eigenvalues.copy()
            values[rank:] = 0
            if 0 < rank < len(values) and self.eigenvalues[rank] > 1e-12*self.eigenvalues[0] and np.isclose(self.eigenvalues[rank-1], self.eigenvalues[rank], rtol=1e-10, atol=0):
                raise ValueError("rank cuts an unidentified tied eigenspace")
        else:
            values = self.eigenvalues * self.weights(tau)
        s = self.invroot @ (self.directions * values) @ self.directions.T @ self.invroot
        out = separate_scales(self.omega, 1, 0)
        out = out.copy()
        out[1:, 1:] += s
        return psd(out)


def recode_response(omega, metric, transform):
    omega = psd(omega)
    t = array(transform, name="response transform", ndim=2)
    if t.shape != (len(omega)-1, len(omega)-1):
        raise ValueError("response transform dimension mismatch")
    inverse = np.linalg.solve(t, np.eye(len(t)))
    d = np.eye(len(omega))
    d[1:, 1:] = inverse
    return d.T @ omega @ d, t @ metric @ t.T
