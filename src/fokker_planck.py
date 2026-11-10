"""
Steady-state Fokker-Planck solver for an electron beam injected at the top of a flare loop,
following Siversky & Zharkova (2009). The equation, in its Eq. (28) form, is

    df/dtau = phi_s   df/ds
            + phi_z   df/dz
            + phi_mu  df/dmu
            + phi_2mu d2f/dmu2

with

    phi_s   = - n sqrt(z) mu                                  (streaming along the loop)
    phi_z   = 2 eps mu sqrt(z) + n / sqrt(z)                  (electric field + collisional energy loss)
    phi_mu  = eps (1 - mu^2) / sqrt(z)                        (electric field)
              - n mu / z^(3/2)                                (collisional drag in angle)
              + n (1 - mu^2) sqrt(z) alpha_B / 2              (magnetic mirroring)
    phi_2mu = n (1 - mu^2) / (2 z^(3/2))                      (collisional pitch-angle diffusion)

Electric field, Eq. (12):   eps(s) = 1/sigma(s) * int_z int_mu z mu f dz dmu
Injection, Eq. (15):        f(z, mu)|top = fn z^(delta-1) / (z^(delta+gamma) + 1) * exp[-(1-mu)^2 / Delta_mu^2],  mu > 0
Boundary conditions:        df/dz = 0 at zmin, zmax;   df/dmu = 0 at mu = -1, 1

The equation is advanced in pseudo-time to steady state by operator splitting:
one implicit upwind step in s, one in z and one in mu per iteration.

Model labels: "C" collisions only, "CE" + electric field, "CB" + magnetic convergence, "CEB" all three.

Important:
- No parameter sweep is done here.
- The collisional angular term is not artificially rescaled.
- The only tuning knob is config.SIGMA_FACTOR, because the atmosphere is approximate.
"""

import numpy as np
from numba import njit, prange

import config as C
from grid import (
    s, z, mu, xi, ds_log, dz_log, dmu,
    n_dimless, sigma_dimless, alpha_B, A_cross, IS_DIAG,
)


# Injection
def make_injection_bc(gamma):
    """
    Injected distribution f(z, mu) at the top boundary, Eq. (15).

    Energy part:  z^(DELTA_LOW - 1) / (z^(DELTA_LOW + gamma) + 1)
                  (a power law of index gamma at high energy, smoothly turned over at low energy)
    Angle part:   exp[-(1 - mu)^2 / DELTA_MU^2] for mu > 0 (downward beam), 0 for mu <= 0.

    Normalized with Eq. (16), F_top / F0 = 1:  int z^2 mu f dz dmu = 1.

    Returns an array of shape (NZ, NMU).
    """
    zz = z[:, None]
    mm = mu[None, :]

    shape_E = zz**(C.DELTA_LOW - 1.0) / (zz**(C.DELTA_LOW + gamma) + 1.0)

    shape_mu = np.zeros_like(mm)
    pos = mm > 0.0
    shape_mu[pos] = np.exp(-((1.0 - mm[pos])**2) / C.DELTA_MU**2)

    raw = shape_E * shape_mu

    # Energy-flux normalization, Eq. (16)
    integrand = raw * zz**2 * np.maximum(mm, 0.0)
    norm = np.trapezoid(np.trapezoid(integrand, x=mu, axis=1), x=z)

    return raw / max(norm, 1e-300)


# Numba
@njit(cache=True, fastmath=True)
def impose_boundaries(f, fbc, mu):
    """
    Apply the spatial boundary conditions in place.

    Top (s index 0):      downward electrons (mu > 0) are the injected beam fbc;
                          upward electrons (mu <= 0) leave freely (zero gradient).
    Bottom (s index -1):  nothing comes back up from below (f = 0 for mu < 0);
                          downward electrons leave freely (zero gradient).
    """
    ns, nz, nmu = f.shape

    # Top boundary: injection for mu > 0
    for j in range(nz):
        for k in range(nmu):
            if mu[k] > 0.0:
                f[0, j, k] = fbc[j, k]
            else:
                f[0, j, k] = f[1, j, k]

    # Bottom boundary
    for j in range(nz):
        for k in range(nmu):
            if mu[k] < 0.0:
                f[ns - 1, j, k] = 0.0
            else:
                f[ns - 1, j, k] = f[ns - 2, j, k]


@njit(cache=True, fastmath=True, parallel=True)
def compute_epsilon_eq12(f, z, mu, sigma):
    """
    Self-consistent return-current electric field, Eq. (12):

        eps(s) = 1/sigma(s) * int_z int_mu z mu f dz dmu

    The double integral (proportional to the beam current) is done with the trapezoidal rule.
    Returns (eps, integral), both of shape (NS,).
    """
    ns, nz, nmu = f.shape
    eps = np.zeros(ns)
    integ = np.zeros(ns)

    for i in prange(ns):
        total = 0.0

        for j in range(nz - 1):
            dz = z[j + 1] - z[j]

            mu_int_j = 0.0
            mu_int_jp = 0.0

            for k in range(nmu - 1):
                dm = mu[k + 1] - mu[k]

                mu_int_j += 0.5 * (
                    mu[k] * f[i, j, k]
                    + mu[k + 1] * f[i, j, k + 1]
                ) * dm

                mu_int_jp += 0.5 * (
                    mu[k] * f[i, j + 1, k]
                    + mu[k + 1] * f[i, j + 1, k + 1]
                ) * dm

            total += 0.5 * (z[j] * mu_int_j + z[j + 1] * mu_int_jp) * dz

        integ[i] = total
        eps[i] = total / max(sigma[i], 1e-300)

    return eps, integ


@njit(cache=True, fastmath=True)
def solve_tridiagonal(lo, di, up, rhs):
    """
    Thomas algorithm for a tridiagonal system.

    lo: sub-diagonal, di: diagonal, up: super-diagonal, rhs: right-hand side.
    Tiny pivots are replaced by 1e-100 to avoid division by zero.
    """
    n = di.size
    c = np.empty(n)
    d = np.empty(n)
    x = np.empty(n)

    denom = di[0]
    if abs(denom) < 1e-100:
        denom = 1e-100

    c[0] = up[0] / denom
    d[0] = rhs[0] / denom

    for i in range(1, n):
        denom = di[i] - lo[i] * c[i - 1]
        if abs(denom) < 1e-100:
            denom = 1e-100

        c[i] = up[i] / denom
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / denom

    x[n - 1] = d[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


@njit(cache=True, fastmath=True, parallel=True)
def step_s_implicit(f, s, ds_log, n_dimless, z, mu, dtau):
    """
    One implicit upwind step of the streaming term phi_s df/ds, phi_s = -n sqrt(z) mu.

    Solved line by line in s for every (z, mu). Negative values are set to zero.
    """
    ns, nz, nmu = f.shape
    out = np.empty_like(f)

    total = nz * nmu

    for idx in prange(total):
        j = idx // nmu
        k = idx - j * nmu

        lo = np.zeros(ns)
        di = np.ones(ns)
        up = np.zeros(ns)
        rhs = np.zeros(ns)

        for i in range(ns):
            rhs[i] = f[i, j, k]

        for i in range(1, ns - 1):
            ds_i = s[i] * ds_log[i - 1]
            phi = -n_dimless[i] * np.sqrt(z[j]) * mu[k]
            r = dtau * phi / ds_i

            # upwind direction depends on the sign of the coefficient
            if phi > 0.0:
                di[i] = 1.0 + r
                up[i] = -r
            else:
                lo[i] = r
                di[i] = 1.0 - r

        # end rows are fixed by impose_boundaries
        di[0] = 1.0
        di[ns - 1] = 1.0

        sol = solve_tridiagonal(lo, di, up, rhs)

        for i in range(ns):
            out[i, j, k] = sol[i] if sol[i] > 0.0 else 0.0

    return out


@njit(cache=True, fastmath=True, parallel=True)
def step_z_implicit(f, z, dz_log, n_dimless, mu, eps, dtau):
    """
    One implicit upwind step of the energy term phi_z df/dz,
    phi_z = 2 eps mu sqrt(z) + n / sqrt(z)  (electric field + collisional loss).

    Boundary condition df/dz = 0 at zmin and zmax.
    """
    ns, nz, nmu = f.shape
    out = np.empty_like(f)

    total = ns * nmu

    for idx in prange(total):
        i = idx // nmu
        k = idx - i * nmu

        lo = np.zeros(nz)
        di = np.ones(nz)
        up = np.zeros(nz)
        rhs = np.zeros(nz)

        for j in range(nz):
            rhs[j] = f[i, j, k]

        for j in range(1, nz - 1):
            dz_j = z[j] * dz_log[j - 1]
            phi = 2.0 * eps[i] * mu[k] * np.sqrt(z[j]) + n_dimless[i] / np.sqrt(z[j])
            r = dtau * phi / dz_j

            if phi > 0.0:
                di[j] = 1.0 + r
                up[j] = -r
            else:
                lo[j] = r
                di[j] = 1.0 - r

        # df/dz = 0 at both ends
        di[0] = 1.0
        up[0] = -1.0
        rhs[0] = 0.0

        lo[nz - 1] = -1.0
        di[nz - 1] = 1.0
        rhs[nz - 1] = 0.0

        sol = solve_tridiagonal(lo, di, up, rhs)

        for j in range(nz):
            out[i, j, k] = sol[j] if sol[j] > 0.0 else 0.0

    return out


@njit(cache=True, fastmath=True, parallel=True)
def step_mu_implicit(f, z, mu, dmu, n_dimless, eps, alpha_B, dtau):
    """
    One implicit step of the pitch-angle terms:
        advection  phi_mu  df/dmu   (electric field, collisional drag, magnetic mirroring), upwinded
        diffusion  phi_2mu d2f/dmu2 (collisional scattering), centred

    Boundary condition df/dmu = 0 at mu = -1 and mu = 1.
    """
    ns, nz, nmu = f.shape
    out = np.empty_like(f)

    total = ns * nz

    for idx in prange(total):
        i = idx // nz
        j = idx - i * nz

        lo = np.zeros(nmu)
        di = np.ones(nmu)
        up = np.zeros(nmu)
        rhs = np.zeros(nmu)

        sqrtz = np.sqrt(z[j])
        z32 = z[j] ** 1.5

        for k in range(nmu):
            rhs[k] = f[i, j, k]

        for k in range(1, nmu - 1):
            one_minus = 1.0 - mu[k] * mu[k]

            phi_mu = (
                eps[i] * one_minus / sqrtz                          # electric field
                - n_dimless[i] * mu[k] / z32                        # collisional drag
                + n_dimless[i] * one_minus * sqrtz * 0.5 * alpha_B[i]  # magnetic mirroring
            )

            phi2 = n_dimless[i] * one_minus / (2.0 * z32)           # collisional diffusion

            adv = dtau * phi_mu / dmu
            dif = dtau * phi2 / (dmu * dmu)

            if phi_mu > 0.0:
                di[k] += adv + 2.0 * dif
                up[k] += -adv - dif
                lo[k] += -dif
            else:
                di[k] += -adv + 2.0 * dif
                lo[k] += adv - dif
                up[k] += -dif

        # df/dmu = 0 at both ends
        di[0] = 1.0
        up[0] = -1.0
        rhs[0] = 0.0

        lo[nmu - 1] = -1.0
        di[nmu - 1] = 1.0
        rhs[nmu - 1] = 0.0

        sol = solve_tridiagonal(lo, di, up, rhs)

        for k in range(nmu):
            out[i, j, k] = sol[k] if sol[k] > 0.0 else 0.0

    return out


# Diagnostics
@njit(cache=True, fastmath=True)
def rel_cut(a, b, isel):
    """Relative L2 change between two distributions at depth index isel (convergence measure)."""
    nz = a.shape[1]
    nmu = a.shape[2]
    s1 = 0.0
    s2 = 0.0

    for j in range(nz):
        for k in range(nmu):
            d = a[isel, j, k] - b[isel, j, k]
            s1 += d * d
            s2 += a[isel, j, k] * a[isel, j, k]

    return np.sqrt(s1) / (np.sqrt(s2) + 1e-100)


@njit(cache=True, fastmath=True, parallel=True)
def density_mu_sign_eq22(f, z, mu, A_cross):
    """
    Beam electron density along the loop, Eq. (22):

        n_b(s) ~ int sqrt(z) A(s) f dz dmu

    split into downward (mu >= 0) and upward (mu < 0) electrons.
    Returns (down, up), each of shape (NS,), in code units.
    """
    ns, nz, nmu = f.shape
    down = np.zeros(ns)
    up = np.zeros(ns)

    for i in prange(ns):
        d = 0.0
        u = 0.0

        for j in range(nz - 1):
            dz = z[j + 1] - z[j]

            for k in range(nmu - 1):
                dm = mu[k + 1] - mu[k]
                muc = 0.5 * (mu[k] + mu[k + 1])
                zc = 0.5 * (z[j] + z[j + 1])

                val = 0.25 * (
                    f[i, j, k]
                    + f[i, j + 1, k]
                    + f[i, j, k + 1]
                    + f[i, j + 1, k + 1]
                ) * np.sqrt(zc) * A_cross[i] * dz * dm

                if muc >= 0.0:
                    d += val
                else:
                    u += val

        down[i] = d
        up[i] = u

    return down, up


@njit(cache=True, fastmath=True)
def local_ratio(f, z, mu, A_cross, isel):
    """Ratio of upward to downward beam density (Eq. 22) at depth index isel."""
    down = 0.0
    up = 0.0

    nz = f.shape[1]
    nmu = f.shape[2]

    for j in range(nz - 1):
        dz = z[j + 1] - z[j]

        for k in range(nmu - 1):
            dm = mu[k + 1] - mu[k]
            muc = 0.5 * (mu[k] + mu[k + 1])
            zc = 0.5 * (z[j] + z[j + 1])

            val = 0.25 * (
                f[isel, j, k]
                + f[isel, j + 1, k]
                + f[isel, j, k + 1]
                + f[isel, j + 1, k + 1]
            ) * np.sqrt(zc) * A_cross[isel] * dz * dm

            if muc >= 0.0:
                down += val
            else:
                up += val

    return up / (down + 1e-300)


@njit(cache=True, fastmath=True)
def diagnostics_terms(n_dimless, sigma, z, mu, eps, alpha_B, isel):
    """
    Relative size of the physical terms at depth index isel.

    Returns the tuple
        (n, sigma, eps, eps/n, alpha_B,
         max|z electric| / max|z collisional|,
         max|mu electric| / max|mu collisional|,
         max|mu magnetic| / max|mu collisional|)
    which tells whether E and B matter compared with collisions at that depth.
    """
    nloc = n_dimless[isel]
    sigloc = sigma[isel]
    epsloc = eps[isel]
    aloc = alpha_B[isel]

    max_z_E = 0.0
    max_z_C = 0.0
    max_mu_E = 0.0
    max_mu_C = 0.0
    max_mu_B = 0.0

    for j in range(z.size):
        sqrtz = np.sqrt(z[j])
        z32 = z[j] ** 1.5

        for k in range(mu.size):
            one_minus = 1.0 - mu[k] * mu[k]

            z_E = abs(2.0 * epsloc * mu[k] * sqrtz)
            z_C = abs(nloc / sqrtz)

            mu_E = abs(epsloc * one_minus / sqrtz)
            mu_C = abs(nloc * mu[k] / z32)
            mu_B = abs(nloc * one_minus * sqrtz * 0.5 * aloc)

            if z_E > max_z_E:
                max_z_E = z_E
            if z_C > max_z_C:
                max_z_C = z_C
            if mu_E > max_mu_E:
                max_mu_E = mu_E
            if mu_C > max_mu_C:
                max_mu_C = mu_C
            if mu_B > max_mu_B:
                max_mu_B = mu_B

    return (
        nloc, sigloc, epsloc, epsloc / (nloc + 1e-300), aloc,
        max_z_E / (max_z_C + 1e-300),
        max_mu_E / (max_mu_C + 1e-300),
        max_mu_B / (max_mu_C + 1e-300)
    )


# Solver
def run_model(gamma, model, verbose=True):
    """
    Solve the Fokker-Planck equation to steady state for one injected beam and one model.

    Parameters
    ----------
    gamma : float
        Spectral index of the injected beam (Eq. 15).
    model : str
        "C", "CE", "CB" or "CEB". "E" switches on the electric field, "B" the magnetic convergence.
    verbose : bool
        Print convergence and diagnostics every config.CHECK_EVERY iterations.

    Returns
    -------
    f : ndarray, shape (NS, NZ, NMU)
        Steady-state distribution f(s, z, mu) in code units.
    eps : ndarray, shape (NS,)
        Final electric field.
    integ : ndarray, shape (NS,)
        Final current integral of Eq. (12).
    diag : tuple
        Output of diagnostics_terms at the diagnostic depth.
    """
    use_E = "E" in model
    use_B = "B" in model

    fbc = make_injection_bc(gamma)
    f = np.zeros((C.NS, C.NZ, C.NMU), dtype=np.float64)

    eps = np.zeros(C.NS)
    integ = np.zeros(C.NS)

    alpha = alpha_B.copy() if use_B else np.zeros_like(alpha_B)

    impose_boundaries(f, fbc, mu)

    last_diag = None

    for it in range(C.N_ITER):
        f_old = f.copy()

        # Electric field update, under-relaxed for stability
        if use_E and (it % C.EPS_EVERY == 0):
            eps_new, integ_new = compute_epsilon_eq12(f, z, mu, sigma_dimless)
            eps = C.OMEGA_EPS * eps_new + (1.0 - C.OMEGA_EPS) * eps
            integ = C.OMEGA_EPS * integ_new + (1.0 - C.OMEGA_EPS) * integ
        elif not use_E:
            eps[:] = 0.0
            integ[:] = 0.0

        # Operator splitting: s, then z, then mu
        f = step_s_implicit(f, s, ds_log, n_dimless, z, mu, C.DTAU)
        impose_boundaries(f, fbc, mu)

        f = step_z_implicit(f, z, dz_log, n_dimless, mu, eps, C.DTAU)
        impose_boundaries(f, fbc, mu)

        f = step_mu_implicit(f, z, mu, dmu, n_dimless, eps, alpha, C.DTAU)
        impose_boundaries(f, fbc, mu)

        # Convergence check at the diagnostic depth
        if it % C.CHECK_EVERY == 0:
            r = rel_cut(f, f_old, IS_DIAG)
            last_diag = diagnostics_terms(n_dimless, sigma_dimless, z, mu, eps, alpha, IS_DIAG)

            if verbose:
                ratio = local_ratio(f, z, mu, A_cross, IS_DIAG)
                print(
                    f"it={it:5d} gamma={gamma} model={model} "
                    f"rel={r:.2e} up/down={ratio:.3e} "
                    f"eps_diag={last_diag[2]:.3e} "
                    f"zE/zC={last_diag[5]:.3e} "
                    f"muE/muC={last_diag[6]:.3e} "
                    f"muB/muC={last_diag[7]:.3e}"
                )

            if it > C.MIN_ITER and r < C.TOL:
                if verbose:
                    print(f"Converged gamma={gamma}, model={model}, it={it}")
                break

    if last_diag is None:
        last_diag = diagnostics_terms(n_dimless, sigma_dimless, z, mu, eps, alpha, IS_DIAG)

    return f, eps, integ, last_diag
