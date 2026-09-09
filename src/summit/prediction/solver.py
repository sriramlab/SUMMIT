"""Independent projected PCGs sharing native matrix work and genotype passes."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import linalg

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from .spec import SolverSpec


class ConvergenceError(RuntimeError):
    def __init__(self, reports):
        self.reports = reports
        failed = ["/".join(k) for k, v in reports.items() if not v["converged"]]
        super().__init__("Prediction solve did not converge: " + ", ".join(failed))


@dataclass
class SolveResult:
    solutions: dict
    fixed_coefficients: dict
    reports: dict
    elapsed_seconds: float


def solve(operator, spec=SolverSpec()):
    start = time.monotonic()
    bases, rhs, diagonal, x, residual, directions, rho, reports = {}, {}, {}, {}, {}, {}, {}, {}
    traits = {t.id: t for t in operator.traits}
    for t in operator.traits:
        u = thin_rank_revealing_fixed_effect_basis(t.fixed, rtol=spec.qr_rtol)
        if u.shape[1] >= len(t.rows):
            raise ValueError(f"{t.id}: fixed effects exhaust the sample space")
        bases[t.id] = u

    def project(key, value):
        u = bases[key[0]]
        return value - u @ (u.T @ value)

    for t in operator.traits:
        py = project((t.id, ""), t.y)
        for c in t.candidates:
            key = (t.id, c.id)
            rhs[key] = py
            d = c.residual + operator.row_diagonal[operator.trait_group[t.id]] * np.einsum("nq,qr,nr->n", t.phi, c.covariance, t.phi)
            if not np.all(np.isfinite(d)) or np.any(d <= 0):
                raise ValueError("invalid projected Jacobi diagonal")
            diagonal[key] = d
            x[key] = np.zeros(len(py))
            residual[key] = py.copy()
            z = project(key, py / d)
            directions[key] = z
            rho[key] = float(py @ z)
            norm = float(np.linalg.norm(py))
            reports[key] = dict(converged=False, iterations=0, restarts=0,
                rhs_norm=norm, threshold=max(spec.atol, spec.rtol*norm),
                recursive_residual_norm=norm, true_residual_norm=None,
                relative_true_residual=None, fixed_projection=None,
                fixed_rank=bases[t.id].shape[1], reason="pending")
    active = set(x)
    fixed_coefficients = {}

    def verify(keys):
        # Every success, including a zero RHS, uses an actual covariance check.
        values = operator.apply({key: x[key] for key in keys}, phase="verification")
        for key in keys:
            vu = values[key]
            true = rhs[key] - project(key, vu)
            norm = float(np.linalg.norm(true))
            rep = reports[key]
            rep["true_residual_norm"] = norm
            rep["relative_true_residual"] = norm / rep["rhs_norm"] if rep["rhs_norm"] else norm
            rep["fixed_projection"] = float(np.linalg.norm(bases[key[0]].T @ x[key]) / max(np.linalg.norm(x[key]), np.finfo(float).tiny))
            if norm <= rep["threshold"] and rep["fixed_projection"] <= max(1e-12, 10*spec.qr_rtol):
                rep.update(converged=True, reason="true_residual")
                active.discard(key)
                t = traits[key[0]]
                # Identified minimum-norm coefficients; preserve exact recipe.
                basis = bases[key[0]]
                # Recover within exactly the retained QR span. Applying a
                # second rank threshold to Z could choose a different mean.
                fixed_coefficients[key] = linalg.lstsq(basis.T @ t.fixed,
                    basis.T @ (t.y-vu), cond=0.0, lapack_driver="gelsy")[0] if basis.shape[1] else np.zeros(t.fixed.shape[1])
            elif rep["restarts"] < spec.max_restarts:
                rep["restarts"] += 1
                residual[key] = true
                z = project(key, true/diagonal[key])
                directions[key] = z
                rho[key] = float(true @ z)
            else:
                rep["reason"] = "true_residual_failed"
                active.discard(key)

    zero = [k for k in active if reports[k]["rhs_norm"] <= reports[k]["threshold"]]
    if zero:
        verify(zero)
    for iteration in range(1, spec.max_iterations+1):
        if not active:
            break
        keys = sorted(active)
        for key in keys:
            directions[key] = project(key, directions[key])
        products = operator.apply({key: directions[key] for key in keys})
        check = []
        for key in keys:
            ap = project(key, products[key])
            curvature = float(directions[key] @ ap)
            if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rho[key]) or rho[key] <= 0:
                reports[key]["reason"] = "invalid_curvature_or_preconditioned_norm"
                active.discard(key)
                continue
            step = rho[key] / curvature
            x[key] += step * directions[key]
            x[key] = project(key, x[key])
            residual[key] -= step * ap
            residual[key] = project(key, residual[key])
            rep = reports[key]
            rep["iterations"] = iteration
            rep["recursive_residual_norm"] = float(np.linalg.norm(residual[key]))
            if rep["recursive_residual_norm"] <= rep["threshold"]:
                check.append(key)
                continue
            z = project(key, residual[key] / diagonal[key])
            new_rho = float(residual[key] @ z)
            directions[key] = z + (new_rho/rho[key])*directions[key]
            rho[key] = new_rho
        if check:
            verify(check)
    if active:
        verify(sorted(active))
        for key in active:
            reports[key]["reason"] = "max_iterations"
    if any(not rep["converged"] for rep in reports.values()):
        raise ConvergenceError(reports)
    return SolveResult(x, fixed_coefficients, reports, time.monotonic()-start)
