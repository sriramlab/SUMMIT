"""Cohort-side Gaussian null fitting; no clipping of method-of-moments fits.

The optimized bounded path fits one PSD additive kernel plus iid residual by
profile REML in an orthogonal residual basis. Its covariance parameters are
estimated from the tested phenotype. Parametric calibration must refit them.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.spec import array_sha256, canonical_sha256, owned_readonly_array


class GaussianNullReference:
    def __init__(self, additive_kernel, fixed_effects, *, identity):
        g = np.asarray(additive_kernel, dtype=float)
        if (
            g.ndim != 2
            or g.shape[0] != g.shape[1]
            or len(g) > 1024
            or not np.all(np.isfinite(g))
        ):
            raise ValueError("spectral null preparation is limited to finite N <= 1024")
        if not np.allclose(g, g.T, atol=1e-10):
            raise ValueError("additive kernel must be symmetric")
        fixed = np.asarray(fixed_effects, dtype=float)
        if fixed.ndim != 2 or len(fixed) != len(g):
            raise ValueError("fixed effects must be sample aligned")
        u = thin_rank_revealing_fixed_effect_basis(fixed)
        residual = null_space(u.T) if u.shape[1] else np.eye(len(g))
        if residual.shape[1] < 3:
            raise ValueError("null model has fewer than three residual dimensions")
        eig, vectors = np.linalg.eigh(residual.T @ g @ residual)
        tolerance = 1e-10 * max(abs(eig).max(), 1)
        if eig[0] < -tolerance:
            raise ValueError("additive kernel is not positive semidefinite")
        self.eigenvalues = owned_readonly_array(np.maximum(eig, 0))
        self.rotation = owned_readonly_array(residual @ vectors)
        self.rank = residual.shape[1]
        self.identity = canonical_sha256(
            dict(declared=identity, kernel=array_sha256(g), fixed=array_sha256(fixed))
        )
        self.nuisance_rank = np.linalg.matrix_rank(
            np.column_stack([np.ones(self.rank), self.eigenvalues]), tol=1e-9
        )
        self.grid = np.linspace(0.0, 1.0, 33)
        d = 1 - self.grid[:, None] + self.grid[:, None] * self.eigenvalues
        valid = np.min(d, axis=1) > 1e-12 * np.maximum(np.max(d, axis=1), 1)
        self.grid_inverse = np.zeros_like(d)
        self.grid_inverse[valid] = 1 / d[valid]
        self.grid_logdet = np.full(len(d), np.inf)
        self.grid_logdet[valid] = np.log(d[valid]).sum(axis=1)

    def transform(self, phenotypes):
        y = np.asarray(phenotypes, dtype=float)
        if (
            y.ndim not in (1, 2)
            or len(y) != len(self.rotation)
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("phenotypes must be finite and sample aligned")
        return self.rotation.T @ y

    def simulate(self, fit, draws, rng):
        return np.sqrt(fit["variance"])[:, None] * rng.normal(size=(self.rank, draws))

    def kernel(self, kernel):
        k = np.asarray(kernel, dtype=float)
        if k.shape != (len(self.rotation), len(self.rotation)):
            raise ValueError("interaction kernel must be sample aligned")
        return self.rotation.T @ k @ self.rotation

    def fit(self, residual_phenotype):
        """Profile REML over nonnegative genetic/residual variance coefficients.

        V=sigma2[(1-h)I+hG], h in [0,1]. Endpoints are admitted only when
        covariance is positive definite. Grid-bracketed local minima and valid
        endpoints are compared; returned coefficients minimize this objective,
        not clipped moment estimates. Fixed effects are removed jointly.
        """
        z = np.asarray(residual_phenotype, dtype=float)
        if z.shape != (self.rank,) or not np.all(np.isfinite(z)) or z @ z <= 0:
            raise ValueError("invalid residual phenotype")
        z2, lam, n = z * z, self.eigenvalues, self.rank

        def objective(h):
            d = 1 - h + h * lam
            if np.min(d) <= 1e-12 * max(np.max(d), 1):
                return np.inf
            sigma2 = np.mean(z2 / d)
            return n * np.log(sigma2) + np.log(d).sum()

        grid = self.grid
        with np.errstate(divide="ignore", invalid="ignore"):
            values = n * np.log(self.grid_inverse @ z2 / n) + self.grid_logdet
        values[~np.isfinite(values)] = np.inf
        candidates = [(values[0], 0.0), (values[-1], 1.0)]
        if self.nuisance_rank > 1:
            for i in range(1, len(grid) - 1):
                if values[i] <= min(values[i - 1], values[i + 1]):
                    fit = minimize_scalar(
                        objective,
                        bounds=(grid[i - 1], grid[i + 1]),
                        method="bounded",
                        options={"xatol": 1e-9},
                    )
                    if not fit.success:
                        raise ArithmeticError("profile REML minimization failed")
                    candidates.append((fit.fun, fit.x))
            # Minima close to an endpoint may fall inside its first grid cell.
            intervals = []
            if values[0] <= values[1]:
                intervals.append((grid[0], grid[1]))
            if values[-1] <= values[-2]:
                intervals.append((grid[-2], grid[-1]))
            for lo, hi in intervals:
                fit = minimize_scalar(
                    objective,
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-9},
                )
                if fit.success:
                    candidates.append((fit.fun, fit.x))
        value, h = min(candidates)
        if not np.isfinite(value):
            raise ArithmeticError("no positive definite null covariance fit")
        d = 1 - h + h * lam
        sigma2 = float(np.mean(z2 / d))
        variance = sigma2 * d
        return dict(
            genetic_variance=float(sigma2 * h),
            residual_variance=float(sigma2 * (1 - h)),
            h=float(h),
            scale=sigma2,
            variance=variance,
            objective=float(value),
            objective_kind="profile_reml",
            genetic_boundary=bool(h < 1e-7),
            residual_boundary=bool(h > 1 - 1e-7),
            covariance_condition=float(d.max() / d.min()),
            nuisance_rank=int(self.nuisance_rank),
            null_identity=self.identity,
        )

    def efficient_score(
        self, residual_phenotype, rotated_kernel, fit, *, retain_contrast=False
    ):
        """Nuisance-orthogonal Gaussian score, standardized by efficient information.

        B=V^-1/2 K V^-1/2. Remove its Frobenius projection on the two nuisance
        derivatives. The resulting B may be indefinite. If parameters were
        known its centered quadratic form has the exported signed spectrum;
        substituting an estimated fit is explicitly a different procedure.
        """
        v = fit["variance"]
        k = np.asarray(rotated_kernel, dtype=float)
        derivatives = np.column_stack([1 / v, self.eigenvalues / v])
        diagonal = np.diag(k) / v
        coef = np.linalg.lstsq(derivatives, diagonal, rcond=1e-10)[0]
        projected = derivatives @ coef
        total = float((1 / v) @ (k * k) @ (1 / v))
        norm2 = total - float(projected @ projected)
        fraction = norm2 / max(total, np.finfo(float).tiny)
        if fraction < 1e-8:
            raise ValueError(
                "interaction is not identifiable against the declared nuisance covariance"
            )
        w = residual_phenotype / np.sqrt(v)
        value = (
            (residual_phenotype / v) @ k @ (residual_phenotype / v)
            - (w * w) @ projected
            - diagonal.sum()
            + projected.sum()
        ) / np.sqrt(2 * norm2)
        result = dict(
            statistic=float(value),
            efficient_information=0.5 * norm2,
            information_fraction=fraction,
        )
        if retain_contrast:
            result["contrast"] = (
                k / np.sqrt(np.outer(v, v)) - np.diag(projected)
            ) / np.sqrt(2 * norm2)
        return result


class GeneralGaussianNullReference:
    """Bounded multi-kernel REML for annotations, dominance and residual surfaces.

    Minimize the Gaussian restricted negative log likelihood over nonnegative
    covariance coefficients. This is distinct from unconstrained MoM. Multiple
    starts and a projected-gradient check are used; unresolved fits fail.
    """

    def __init__(self, kernels, fixed_effects, *, names, identity):
        from scipy.linalg import null_space

        kernels = np.asarray(kernels, dtype=float)
        if (
            kernels.ndim != 3
            or kernels.shape[1] != kernels.shape[2]
            or kernels.shape[1] > 384
        ):
            raise ValueError("multi-kernel REML is currently bounded at N <= 384")
        if (
            len(names) != len(kernels)
            or len(set(names)) != len(names)
            or names[-1] != "residual"
        ):
            raise ValueError(
                "nuisance kernels must be uniquely named and end with residual"
            )
        fixed = np.asarray(fixed_effects, dtype=float)
        u = thin_rank_revealing_fixed_effect_basis(fixed)
        self.rotation = null_space(u.T) if u.shape[1] else np.eye(len(fixed))
        self.rank = self.rotation.shape[1]
        if self.rank < 3:
            raise ValueError("insufficient residual rank")
        reduced = np.stack([self.rotation.T @ k @ self.rotation for k in kernels])
        if not np.all(np.isfinite(reduced)) or not np.allclose(
            reduced, reduced.transpose(0, 2, 1), atol=1e-9
        ):
            raise ValueError("invalid nuisance covariance kernels")
        if not np.allclose(reduced[-1], np.eye(self.rank), atol=1e-9):
            raise ValueError("last nuisance must be the residual identity")
        for k in reduced:
            eig = np.linalg.eigvalsh(k)
            if eig[0] < -1e-9 * max(eig[-1], 1):
                raise ValueError("nuisance kernel is not PSD")
        self.scales = np.trace(reduced, axis1=1, axis2=2) / self.rank
        if np.any(self.scales <= 1e-12):
            raise ValueError("nuisance kernel is annihilated by fixed effects")
        self.kernels = reduced / self.scales[:, None, None]
        gram = np.einsum("aij,bij->ab", self.kernels, self.kernels)
        if np.linalg.cond(gram) > 1e10:
            raise ValueError("nuisance covariance components are not identifiable")
        self.names = tuple(names)
        self.identity = canonical_sha256(
            dict(
                declared=identity,
                kernels=array_sha256(kernels),
                fixed=array_sha256(fixed),
            )
        )

    def transform(self, y):
        y = np.asarray(y, dtype=float)
        if (
            y.ndim not in (1, 2)
            or len(y) != len(self.rotation)
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("invalid phenotype alignment")
        return self.rotation.T @ y

    def kernel(self, k):
        return self.rotation.T @ np.asarray(k) @ self.rotation

    def fit(self, z):
        from scipy.linalg import cho_factor, cho_solve
        from scipy.optimize import minimize

        z = np.asarray(z, dtype=float)
        if z.shape != (self.rank,) or not np.all(np.isfinite(z)) or z @ z <= 0:
            raise ValueError("invalid residual phenotype")
        scale = float(z @ z / self.rank)
        zz = z / np.sqrt(scale)

        def objective(theta):
            v = np.einsum("c,cij->ij", theta, self.kernels)
            try:
                cf = cho_factor(v, lower=True, check_finite=False)
                vi = cho_solve(cf, np.eye(self.rank), check_finite=False)
                alpha = vi @ zz
                fun = 0.5 * (2 * np.log(np.diag(cf[0])).sum() + zz @ alpha)
                gradient = 0.5 * (
                    np.einsum("ij,cji->c", vi, self.kernels)
                    - np.einsum("i,cij,j->c", alpha, self.kernels, alpha)
                )
                return float(fun), gradient
            except np.linalg.LinAlgError:
                return 1e100, np.zeros(len(theta))

        c = len(self.kernels)
        starts = [
            np.r_[np.full(c - 1, 0.3 / (c - 1)), 0.7],
            np.r_[np.zeros(c - 1), 1.0],
        ]
        fits = [
            minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, None)] * (c - 1) + [(1e-10, None)],
                options={"ftol": 1e-12, "gtol": 1e-6, "maxiter": 150},
            )
            for start in starts
        ]
        fit = min(fits, key=lambda r: r.fun)

        def kkt(candidate):
            return float(
                np.max(
                    abs(
                        np.where(
                            (candidate.x < 1e-8) & (candidate.jac > 0), 0, candidate.jac
                        )
                    )
                )
            )

        optimizer = "L-BFGS-B"
        if not fit.success or kkt(fit) > 1e-4 * self.rank:
            # Numerical fallback only, with the SAME objective and constraints.
            # Do not change the nuisance model or use the interaction P value.
            additional = [
                minimize(
                    objective,
                    start,
                    jac=True,
                    method="SLSQP",
                    bounds=[(0, None)] * (c - 1) + [(1e-10, None)],
                    options={"ftol": 1e-11, "maxiter": 300},
                )
                for start in (fit.x, starts[0])
            ]
            candidates = [
                r
                for r in [*fits, *additional]
                if r.success
                and kkt(r) <= 1e-4 * self.rank
                and r.fun <= fit.fun + 1e-7 * max(1, abs(fit.fun))
            ]
            if not candidates:
                details = [
                    dict(
                        success=bool(r.success),
                        objective=float(r.fun),
                        kkt=kkt(r),
                        message=str(r.message),
                    )
                    for r in [*fits, *additional]
                ]
                raise ArithmeticError(
                    f"multi-kernel REML convergence failure: {details}"
                )
            fit = min(candidates, key=lambda r: r.fun)
            optimizer = "objective_checked_SLSQP_fallback"
        projected = np.where((fit.x < 1e-8) & (fit.jac > 0), 0, fit.jac)
        v = scale * np.einsum("c,cij->ij", fit.x, self.kernels)
        return dict(
            coefficients=scale * fit.x / self.scales,
            component_names=self.names,
            covariance=v,
            objective=float(fit.fun + 0.5 * self.rank * np.log(scale)),
            objective_kind="constrained_gaussian_reml",
            projected_gradient=float(np.max(abs(projected))),
            optimizer=optimizer,
            genetic_boundary=bool(np.any(fit.x[:-1] < 1e-7)),
            boundary_components=np.flatnonzero(fit.x < 1e-7),
            covariance_condition=float(np.linalg.cond(v)),
            null_identity=self.identity,
        )

    def simulate(self, fit, draws, rng):
        return np.linalg.cholesky(fit["covariance"]) @ rng.normal(
            size=(self.rank, draws)
        )

    def efficient_score(self, z, k, fit, *, retain_contrast=False):
        from scipy.linalg import solve_triangular

        l = np.linalg.cholesky(fit["covariance"])

        def whiten(k):
            left = solve_triangular(l, k, lower=True, check_finite=False)
            return solve_triangular(l, left.T, lower=True, check_finite=False).T

        b = whiten(k)
        nuisance = np.stack([whiten(x) for x in self.kernels])
        gram = np.einsum("aij,bij->ab", nuisance, nuisance)
        alpha = np.linalg.solve(gram, np.einsum("aij,ij->a", nuisance, b))
        efficient = b - np.einsum("c,cij->ij", alpha, nuisance)
        norm2 = float(np.sum(efficient**2))
        fraction = norm2 / max(float(np.sum(b * b)), np.finfo(float).tiny)
        if fraction < 1e-8:
            raise ValueError("interaction is not identifiable against nuisance kernels")
        w = solve_triangular(l, z, lower=True, check_finite=False)
        result = dict(
            statistic=float(
                (w @ efficient @ w - np.trace(efficient)) / np.sqrt(2 * norm2)
            ),
            efficient_information=0.5 * norm2,
            information_fraction=fraction,
        )
        if retain_contrast:
            result["contrast"] = efficient / np.sqrt(2 * norm2)
        return result
