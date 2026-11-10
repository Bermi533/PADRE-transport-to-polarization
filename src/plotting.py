"""
Every figure of the project. Functions here only draw: they receive results that were
already computed (the distribution f from results/fokker_planck/, or Stokes arrays/CSVs) and
save a PNG. 

Figures
-------
Distribution function and beam density (solver output):
    make_fig4                     beam density n_b(xi), downward/upward/total (Eq. 22), gamma = 3, 7, all models
    make_fig1_v6                  3D surfaces of f(E, mu), gamma = 3, 4 models x 3 column depths
    plot_E_mu_heatmaps_subplots   heatmaps of log10 f(E, mu) at several depths

Polarization and spectra (bremsstrahlung):
    plot_komarov_fig1             single 8 keV electron, P vs eps/E (check against Komarov et al. 2016, Fig. 1)
    plot_polarization_by_depth    Q/I vs photon energy for each depth CSV and the integrated CSV
    plot_spectrum_by_depth        normalized spectrum for each depth CSV and the integrated CSV
    plot_spectrum_rescaled        spectra by depth, with their real amplitude or times eps^2
    plot_polarization_vs_angle    Q/I vs viewing angle
    plot_intensity_panels         integrated intensity by case, by angle, and local photon index
    make_fig5                     HXR spectra (theta = 0, 90, 180) and polarization (theta = 90) across depth
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from grid import z, mu, xi, s, E_keV, A_cross, IS_DIAG
from fokker_planck import density_mu_sign_eq22
from bremsstrahlung import single_electron_polarization
from data_io import read_csv


def _save(fig, outname, dpi=200, **kw):
    fig.savefig(outname, dpi=dpi, bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(f"Saved: {outname}")


# Distribution function and beam density
def make_fig4(results, outname="fig4_normalized_2009_single.png"):
    """
    Beam electron density along the loop, Eq. (22), like Fig. 4 of Siversky & Zharkova (2009).

    Rows: gamma = 3, 7. Columns: C, C+E, C+B, C+E+B.
    Dotted: downward (mu > 0), dashed: upward (mu < 0), solid: total.
    Each panel is scaled so that its maximum total density is 1e8.

    results : dict {(gamma, model): f}
    """
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.8), sharex=True, sharey=True)

    gammas = [3, 7]
    models = ["C", "CE", "CB", "CEB"]
    labels = ["C", "C+E", "C+B", "C+E+B"]

    for row, gamma in enumerate(gammas):
        for col, model in enumerate(models):
            ax = axes[row, col]

            f = results[(gamma, model)]
            down, up = density_mu_sign_eq22(f, z, mu, A_cross)
            total = down + up

            scale = 1.0e8 / max(np.max(total), 1e-300)

            ax.loglog(xi, np.maximum(total * scale, 1e-300), "k-", lw=1.2)
            ax.loglog(xi, np.maximum(down * scale, 1e-300), "k:", lw=1.2)
            ax.loglog(xi, np.maximum(up * scale, 1e-300), "k--", lw=1.2, dashes=(7, 5))

            ax.set_xlim(1e17, 1e23)
            ax.set_ylim(1e3, 1e8)

            ax.text(0.11, 0.14, rf"$\gamma={gamma}$" + "\n" + labels[col],
                    transform=ax.transAxes, fontsize=10)

            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

            if row == 1:
                ax.set_xlabel(r"$\xi,\ \mathrm{cm}^{-2}$")
            if col == 0:
                ax.set_ylabel(r"$n_b,\ \mathrm{cm}^{-3}$")

    fig.suptitle("Fig. 4 normalized variables, Eq. 22 density")
    fig.text(0.5, 0.015,
             "dotted: downward ($\\mu>0$), dashed: upward ($\\mu<0$), solid: total",
             ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    _save(fig, outname, dpi=200)


def make_fig1_v6(results, outname="fig1_v6_depths_en.png"):
    """
    3D surfaces of f(E, mu) for gamma = 3: 4 models (columns) x 3 column depths (rows:
    xi ~ 1e18, 4e19, 1e20 cm^-2). English labels, large fonts.

    The floor of 1e-300 only avoids log10(0); the visible cut is the clip of Z to [2, 15].
    """
    gamma_fixed = 3
    models = ["C", "CE", "CB", "CEB"]
    labels = ["C", "C+E", "C+B", "C+E+B"]

    xi_targets = [1.0e18, 4.0e19, 1.0e20]
    depth_labels = [
        r"$\xi\approx10^{18}\,\mathrm{cm}^{-2}$",
        r"$\xi\approx4\times10^{19}\,\mathrm{cm}^{-2}$",
        r"$\xi\approx10^{20}\,\mathrm{cm}^{-2}$",
    ]
    i_depths = [int(np.argmin(np.abs(xi - target))) for target in xi_targets]

    letters = list("abcdefghijkl")

    ip0 = int(np.argmin(np.abs(E_keV - 12.0)))
    E_plot = E_keV[ip0:]
    E_g, mu_g = np.meshgrid(E_plot, mu, indexing="ij")

    fig = plt.figure(figsize=(22, 15))

    panel_idx = 0
    for row, (i_depth, depth_label) in enumerate(zip(i_depths, depth_labels)):
        for col, (model, label) in enumerate(zip(models, labels)):
            ax = fig.add_subplot(3, 4, panel_idx + 1, projection="3d")

            f_sl = results[(gamma_fixed, model)][i_depth, ip0:, :].copy()
            f_sl = uniform_filter(f_sl, size=1)

            floor = 1e-300 * np.max(f_sl)
            f_sl = np.maximum(f_sl, floor)

            z_offset = 15.0
            zmin_plot, zmax_plot = 2.0, 15.0
            zticks = [0, 5, 10, 15]
            zticklabels = [r"$10^0$", r"$10^5$", r"$10^{10}$", r"$10^{15}$"]

            Z = np.log10(f_sl / np.max(f_sl)) + z_offset
            Z = np.clip(Z, zmin_plot, zmax_plot)

            ax.plot_surface(
                np.log10(E_g), mu_g, Z,
                rstride=1, cstride=1, linewidth=0.3,
                edgecolor="k", facecolor="white",
                shade=False, antialiased=True
            )

            title = label if row == 0 else ""
            ax.set_title(title, fontsize=16)

            if col == 0:
                ax.text2D(-0.18, 0.5, depth_label, transform=ax.transAxes,
                          fontsize=12, rotation=90, va="center", ha="center")

            ax.set_xlabel("Energy, keV", fontsize=10, labelpad=6)
            ax.set_ylabel(r"$\mu=\cos\alpha$", fontsize=10, labelpad=6)

            xticks_keV = [10, 100, 1000]
            ax.set_xticks([np.log10(x) for x in xticks_keV])
            ax.set_xticklabels([str(x) for x in xticks_keV], fontsize=9)

            ax.set_zlim(zmin_plot, zmax_plot)
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticklabels, fontsize=9)
            ax.tick_params(axis="y", labelsize=9)

            ax.text2D(-0.05, 0.94, f"({letters[panel_idx]})",
                      transform=ax.transAxes, fontsize=13, fontweight="bold")

            ax.view_init(elev=22, azim=-55)
            ax.set_box_aspect([1.3, 1.0, 0.8])

            panel_idx += 1

    fig.suptitle(r"Electron distribution function vs. column depth, $\gamma=3$", fontsize=17)
    fig.text(0.5, 0.005,
             r"$z$-axis (all panels): $f(E,\mu)$ [arbitrary units], log scale ($10^0$ to $10^{15}$)",
             ha="center", fontsize=13)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.05, wspace=0.35, hspace=0.22)
    _save(fig, outname, dpi=170)


def plot_E_mu_heatmaps_subplots(f, s, z, mu, s_indices, outname="heatmaps_E_mu.png",
                                title=None, decades=12):
    """
    Heatmaps of log10 f(E, mu) at several column depths, one panel per depth.

    Parameters
    ----------
    f : ndarray (NS, NZ, NMU)   distribution from the solver
    s, z, mu : solver grids (dimensionless); converted here to xi [cm^-2] and E [keV]
    s_indices : list of depth indices, e.g. [int(0.2*NS), int(0.5*NS), int(0.8*NS)]
    decades : how many decades below the maximum of each panel are shown
    """
    from grid import xi0
    import config as C

    E = z * C.E0_keV
    n = len(s_indices)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, i_s in zip(axes, s_indices):
        f_sl = f[i_s]
        fmax = max(np.max(f_sl), 1e-300)
        logf = np.log10(np.maximum(f_sl / fmax, 10.0**(-decades)))
        pcm = ax.pcolormesh(E, mu, logf.T, shading="auto", cmap="viridis",
                            vmin=-decades, vmax=0.0)
        ax.set_xscale("log")
        ax.set_xlabel("Energy (keV)")
        ax.set_title(rf"$\xi = {s[i_s] * xi0:.2e}$ cm$^{{-2}}$", fontsize=10)
    axes[0].set_ylabel(r"$\mu = \cos\alpha$")
    fig.colorbar(pcm, ax=axes, label=r"$\log_{10}(f / f_{\max})$")
    if title:
        fig.suptitle(title)
    _save(fig, outname, dpi=200)


# Polarization and spectra
def plot_komarov_fig1(outname="komarov_fig1_check.png", E_t=8.0):
    """
    Single electron of 8 keV moving along the axis: P = (sigma_perp - sigma_par)/(sigma_perp + sigma_par)
    versus eps/E, for psi = pi/2, pi/3, pi/6. Reproduces Komarov et al. (2016), Fig. 1
    (the paper's theta is psi here). The sign change is expected near eps ~ E/8.
    """
    ratio = np.logspace(np.log10(6.0e-3), np.log10(0.999), 300)   # sigma = 0 at eps = E
    curves = [
        (np.pi / 2, "-", r"$\theta=\pi/2$"),
        (np.pi / 3, "-.", r"$\theta=\pi/3$"),
        (np.pi / 6, ":", r"$\theta=\pi/6$"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for theta, style, label in curves:
        P = single_electron_polarization(E_t, ratio, np.cos(theta))
        ax.semilogx(ratio, P, style, color="k", lw=1.3, label=label)

    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.axvline(0.125, color="k", ls="--", lw=0.8)

    ax.set_xlim(6.0e-3, 1.1)
    ax.set_ylim(-1.0, 0.5)
    ax.set_xlabel(r"$\epsilon/\varepsilon$")
    ax.set_ylabel("degree of polarization $P$")
    ax.set_title(r"Komarov Fig. 1: $\varepsilon = 8$ keV")
    ax.legend(loc="lower left", frameon=False)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    plt.tight_layout()
    _save(fig, outname)


def plot_polarization_by_depth(layer_files, integrated_file, outname="polarization_by_depth.png", theta=90.0):
    """
    Q/I vs photon energy, one curve per depth CSV and the integrated CSV (black dashed).

    layer_files : dict {legend label: csv path};  integrated_file : csv path
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, fname in layer_files.items():
        E, flux, P = read_csv(fname)
        ax.plot(E, P, lw=1.8, label=label)

    E, flux, P = read_csv(integrated_file)
    ax.plot(E, P, "k--", lw=2.5, label="Depth-integrated")

    ax.axhline(0.0, color="gray", lw=0.8)   # where Q/I changes sign
    ax.set_xscale("log")
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Linear polarization  Q/I")
    ax.set_title(rf"HXR polarization vs photon energy ($\theta = {theta:.0f}^\circ$)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, outname)


def plot_spectrum_by_depth(layer_files, integrated_file, outname="spectrum_by_depth.png"):
    """Normalized spectrum (I / I at the lowest energy) for each depth CSV and the integrated CSV."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, fname in layer_files.items():
        E, flux, P = read_csv(fname)
        ax.plot(E, flux, lw=1.8, label=label)

    E, flux, P = read_csv(integrated_file)
    ax.plot(E, flux, "k--", lw=2.5, label="Depth-integrated")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Normalized flux  (I / I at 15 keV)")
    ax.set_title("HXR spectrum by depth")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, outname)


def plot_spectrum_rescaled(layer_files, I0_factor, mode="brightness", outname="spectrum_by_depth_rescaled.png"):
    """
    Spectra by depth, rescaled.

    mode = "brightness": undo the normalization with the factors I0_factor[label] printed by the
                         pipeline, so each curve carries its real amplitude (code units).
    mode = "E2":         multiply by eps^2 to flatten the steep fall and compare shapes.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, fname in layer_files.items():
        E, flux, P = read_csv(fname)
        if mode == "brightness":
            y = flux * I0_factor[label]
            ylabel = "Emission per unit column depth (code units)"
        else:
            y = flux * E**2
            ylabel = r"Normalized flux $\times$ $\epsilon^2$ (arb. units)"
        ax.plot(E, y, lw=1.8, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel(ylabel)
    ax.set_title("HXR spectrum by depth, rescaled")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, outname)


def plot_polarization_vs_angle(thetas, photon_keV, QI_int, QI_single, outname="polarization_vs_angle.png"):
    """Q/I vs viewing angle: depth-integrated (solid) and the layer near xi = 1e18 (dashed)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for k, e in enumerate(photon_keV):
        ax.plot(thetas, QI_int[:, k], "-o", color=colors[k], lw=1.8, ms=4, label=f"Integrated, {e:.0f} keV")
        ax.plot(thetas, QI_single[:, k], "--", color=colors[k], lw=1.5,
                label=rf"$\xi \approx 10^{{18}}$, {e:.0f} keV")

    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel(r"Viewing angle $\theta$ (deg)")
    ax.set_ylabel("Linear polarization  Q/I")
    ax.set_title("HXR polarization vs viewing angle")
    ax.set_xticks(np.arange(0, 181, 30))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, outname)


def plot_intensity_panels(photon_keV, I_a, I_b, thetas_b, gamma_b, model_b, theta_a=90.0,
                          outname="intensity_panels.png"):
    """
    (a) depth-integrated intensity for several (gamma, model) at theta_a;
    (b) the base case at several viewing angles;
    (c) local photon index -d ln I / d ln eps for the cases in (a).

    I_a : dict {(gamma, model): I(eps)};  I_b : array (n_theta, n_photon)
    """
    COLORES = {3.0: "tab:red", 7.0: "tab:blue"}
    ESTILOS = {"CE": "-", "CEB": "-."}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for (gam, modelo), I in I_a.items():
        axes[0].plot(photon_keV, I, color=COLORES.get(float(gam)), ls=ESTILOS.get(modelo, "-"), lw=1.8,
                     label=rf"$\gamma={gam:g}$, {modelo}")
    axes[0].set_title(rf"Integrated intensity, $\theta={theta_a:.0f}^\circ$")

    for it, th in enumerate(thetas_b):
        axes[1].plot(photon_keV, I_b[it], lw=1.8, label=rf"$\theta={th:.0f}^\circ$")
    axes[1].set_title(rf"Viewing angle ($\gamma={gamma_b:g}$, {model_b})")

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Intensity (code units)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, which="both")

    for (gam, modelo), I in I_a.items():
        indice = -np.gradient(np.log(I), np.log(photon_keV))
        axes[2].plot(photon_keV, indice, color=COLORES.get(float(gam)), ls=ESTILOS.get(modelo, "-"), lw=1.8,
                     label=rf"$\gamma={gam:g}$, {modelo}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Photon energy (keV)")
    axes[2].set_ylabel("Local photon index")
    axes[2].set_title("Spectral hardness")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    _save(fig, outname)


def make_fig5(data, rows, photon_keV, thetas=(0.0, 90.0, 180.0), theta_pol=90.0,
              outname="fig5_hxr_spectra_polarization.png"):
    """
    HXR spectra and polarization degree across depth (like Fig. 5 of Zharkova et al. 2010).

    One row per case in `rows` (xi = 0 injected beam, then the models at a given xi).
    Columns 1-3: intensity at theta = 0, 90, 180 deg. Column 4: Q/I at theta = 90 deg.
    Red gamma = 3, blue gamma = 7. Solid: total, dotted: downward (mu > 0), dashed: upward (mu < 0).
    Each intensity curve set is normalized to its own total maximum (times 1e8), to compare shapes.

    `data` comes from polarization.fig5_data.
    """
    colors = {3: "red", 7: "blue"}
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, len(thetas) + 1, figsize=(10.5, 2.5 * nrow), sharex="col")
    axes = np.atleast_2d(axes)

    for r, (row_label, model, xi_target) in enumerate(rows):
        for c, theta_deg in enumerate(thetas):
            ax = axes[r, c]
            for gamma in colors:
                I = data[(r, gamma)]["I"][theta_deg]
                norm = max(np.max(I[0]), 1e-300)
                scale = 1e8
                ax.loglog(photon_keV, I[0] / norm * scale, color=colors[gamma], lw=1.0)
                if model != "boundary":
                    ax.loglog(photon_keV, I[1] / norm * scale, color=colors[gamma], lw=0.9, ls=":")
                    ax.loglog(photon_keV, I[-1] / norm * scale, color=colors[gamma], lw=0.9, ls="--")

            ax.set_xlim(photon_keV[0], photon_keV[-1])
            ax.set_ylim(1e0, 1e8)
            if r == 0:
                ax.set_title(rf"$\theta={int(theta_deg)}^\circ$", fontsize=9)
            if c == 0:
                ax.set_ylabel("Intensity", fontsize=8)
                ax.text(-0.38, 0.5, f"({chr(ord('a') + r)})", transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center")
                ax.text(0.05, 0.12, row_label, transform=ax.transAxes, fontsize=7)
            if r == nrow - 1:
                ax.set_xlabel("Energy, keV", fontsize=8)
            ax.tick_params(axis="both", which="both", labelsize=7, direction="in", top=True, right=True)

        axp = axes[r, len(thetas)]
        for gamma in colors:
            P = data[(r, gamma)]["P"]
            axp.semilogx(photon_keV, P[0], color=colors[gamma], lw=1.0)
            if model != "boundary":
                axp.semilogx(photon_keV, P[1], color=colors[gamma], lw=0.8, ls=":")
                axp.semilogx(photon_keV, P[-1], color=colors[gamma], lw=0.8, ls="--")

        axp.set_xlim(photon_keV[0], photon_keV[-1])
        axp.set_ylim(-0.75, 0.35)
        axp.axhline(0.0, color="k", lw=0.5)
        if r == 0:
            axp.set_title("Polarization\n" + rf"$\theta={int(theta_pol)}^\circ$", fontsize=9)
        axp.set_ylabel("Polarization degree", fontsize=8)
        if r == nrow - 1:
            axp.set_xlabel("Energy, keV", fontsize=8)
        axp.tick_params(axis="both", which="both", labelsize=7, direction="in", top=True, right=True)

    fig.suptitle("Hard X-ray spectra and polarization degree across depth", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0.04, 0.02, 0.98, 0.985])
    _save(fig, outname, dpi=220)
