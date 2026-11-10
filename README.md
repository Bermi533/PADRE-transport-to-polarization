# PADRE: electron transport to hard X-ray polarization

Theoretical model of the transport of flare-accelerated electrons with the Fokker–Planck equation, and calculation of the linear polarization of the hard X-rays** (bremsstrahlung) produced by their angular distribution. 

**Author:** Milena Andrea Barrios-Gallego 

> ⚠️ **Work in progress.** This code is under active development and may still contain errors or change without notice. 

## What the code does

1. **Fokker–Planck solver** for the electron distribution function f(ξ, E, μ) along the loop, with Coulomb collisions (C), the return-current electric field (E) and magnetic convergence (B).
2. **Polarized bremsstrahlung:** from f, the photon spectrum and the Stokes parameters are computed with the relativistic Gluckstern–Hull cross sections, in the corrected form of Komarov et al. (2016, Appendix B). In this model U = 0 by gyrophase symmetry, so the output is Q/I (negative Q/I means polarization parallel to the sky projection of B).
3. **Output pipeline:** CSV files with three columns (photon energy, normalized flux, Q/I), integrated over depth (what SHARP sees) and at individual column depths.

## Structure

```
├── src/                     functions (not run directly)
│   ├── config.py            ALL parameters: grid, numerics, atmosphere, beam, B field
│   ├── grid.py              grid (s, z, μ), atmosphere, conductivity, B profile
│   ├── fokker_planck.py     injection (Eq. 15), solver (Eq. 28), beam density (Eq. 22)
│   ├── bremsstrahlung.py    polarized cross sections and Stokes I, Q, U (Eq. 6)
│   ├── polarization.py      interpolation in ξ, depth integration (Eq. 7), angle scans
│   ├── data_io.py           save/load f and write the CSVs
│   └── plotting.py          functions for every figure
├── scripts/                 what you run
│   ├── run_solver.py        1. solves Fokker–Planck and saves f to results/fokker_planck/
│   ├── run_pipeline.py      2. HXR polarization CSVs (energy, flux, Q/I)
│   └── make_figures.py      3. figures, reading the saved f (never re-solves)
├── run_all.sh               runs 1, 2 and 3 in order
├── results/
│   ├── fokker_planck/       f (.npz, not tracked by git), summary and profiles, Fig. 4, Fig. 1 v6, heatmaps
│   └── polarization/        HXR CSVs, normalization factors, polarization and spectra figures (incl. Fig. 5)
├── docs/
│   ├── mcconnell2002/       from Klein–Nishina to the azimuthal modulation measured by a Compton polarimeter
│   └── padre_meeting/       talk: from Fokker–Planck to HXR polarization
├── requirements.txt
└── .gitignore
```

## References

**Electron transport (Fokker–Planck) and HXR emission**
- Siversky, T. V., & Zharkova, V. V. (2009). Stationary and impulsive injection of electron beams in converging magnetic field. *Astronomy & Astrophysics*, 504, 1057–1070.
- Zharkova, V. V., Kuznetsov, A. A., & Siversky, T. V. (2010). Diagnostics of energetic electrons with anisotropic distributions in solar flares. I. Hard X-rays bremsstrahlung emission. *Astronomy & Astrophysics*, 512, A8.

**Polarized bremsstrahlung cross sections**
- Gluckstern, R. L., & Hull, M. H. (1953). Polarization dependence of the integrated bremsstrahlung cross section. *Physical Review*, 90, 1030.
- Komarov, S. V., Khabibullin, I. I., Churazov, E. M., & Schekochihin, A. A. (2016). Polarization of thermal bremsstrahlung emission due to electron pressure anisotropy. *Monthly Notices of the Royal Astronomical Society*, 461, 2162–2173. (Appendix B: corrected cross sections used here.)

**HXR polarimetry (Klein–Nishina and Compton polarimeters)**
- McConnell, M. L., Ryan, J. M., Smith, D. M., Lin, R. P., & Emslie, A. G. (2002). RHESSI as a hard X-ray polarimeter. *Solar Physics*, 210, 125–142.

**HXR polarization and directivity in solar flares**
- Jeffrey, N. L. S., & Kontar, E. P. (2011). Spatially resolved hard X-ray polarization in solar flares: effects of Compton scattering and bremsstrahlung. *Astronomy & Astrophysics*, 536, A93.
- Jeffrey, N. L. S., et al. (2024). A modeling investigation for solar flare X-ray stereoscopy with Solar Orbiter/STIX and Earth-orbiting missions. *The Astrophysical Journal*, 964, 145.

