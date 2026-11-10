# McConnell et al. (2002): from Klein–Nishina (eq. 1) to the azimuthal modulation (eq. 3)

Paper: McConnell, Ryan, Smith, Lin & Emslie, *RHESSI as a Hard X-Ray Polarimeter*, Solar Physics 210, 125–142 (2002). 

---

## 0. What the paper says

**Eq. (1), Klein–Nishina for a linearly polarized photon:**

$$
d\sigma = \frac{r_0^2}{2}\,d\Omega\left(\frac{\nu}{\nu_0}\right)^2\left[\frac{\nu_0}{\nu}+\frac{\nu}{\nu_0}-2\sin^2\theta\cos^2\eta\right]
$$

**Eq. (2), Compton kinematics:**

$$
\frac{\nu}{\nu_0}=\frac{1}{1+\dfrac{h\nu_0}{mc^2}(1-\cos\theta)}
$$

- $\theta$: scattering (polar) angle, measured from the direction of the incident photon.
- $\eta$: azimuthal angle of the scattered photon, measured **from the plane containing the electric vector $\vec E$ of the incident photon**.
- $r_0 = e^2/mc^2$: classical electron radius.

**Eq. (3), integrated azimuthal distribution:**

$$
C(\eta)=A\cos \left[2\left(\eta-\phi+\frac{\pi}{2}\right)\right]+B
$$

where $\phi$ is the polarization angle and $A, B$ are fit constants.

The paper only says "from Equation (1), we see that the azimuthal modulation follows a $\cos 2\eta$ distribution". The intermediate steps are below.

---

## 1. Sanity check of eq. (1): the Thomson limit

For $h\nu_0 \ll mc^2$ (20–100 keV against 511 keV), eq. (2) gives $\nu\approx\nu_0$ and eq. (1) reduces to

$$
d\sigma \approx r_0^2\,d\Omega\,\left[1-\sin^2\theta\cos^2\eta\right].
$$

Since $\sin\theta\cos\eta = \hat{\varepsilon}\cdot\hat{k}'$ (the projection of the scattered direction $\hat k'$ onto the incident polarization $\hat\varepsilon$), this is $r_0^2(1-(\hat\varepsilon\cdot\hat k')^2)$, the classical dipole pattern. The electron oscillates along $\vec E$ and **does not radiate along $\vec E$** ($\eta=0$, $\theta=90^\circ$). That is the whole physics of Compton polarimetry: photons are preferentially scattered **perpendicular** to $\vec E$.

---

## 2. Separate the $\eta$-dependent part

Use the identity $\cos^2\eta = \tfrac12(1+\cos 2\eta)$:

$$
\frac{\nu_0}{\nu}+\frac{\nu}{\nu_0}-2\sin^2\theta\cos^2\eta
=\underbrace{\left(\frac{\nu_0}{\nu}+\frac{\nu}{\nu_0}-\sin^2\theta\right)}_{\text{independent of }\eta}
\;-\;\sin^2\theta\,\cos 2\eta .
$$

So, for fixed $\theta$:

$$
\frac{d\sigma}{d\Omega}(\theta,\eta)=\underbrace{\frac{r_0^2}{2}\left(\frac{\nu}{\nu_0}\right)^2\left(\frac{\nu_0}{\nu}+\frac{\nu}{\nu_0}-\sin^2\theta\right)}_{b(\theta)}
\;-\;\underbrace{\frac{r_0^2}{2}\left(\frac{\nu}{\nu_0}\right)^2\sin^2\theta}_{a(\theta)}\;\cos 2\eta .
$$

The "constant + amplitude × $\cos 2\eta$" form is already there. The period is $180^\circ$ (not $360^\circ$) because the physics cannot tell $\vec E$ from $-\vec E$.

---

## 3. Go from the angle relative to $\vec E$ to the instrument's angle

The detector **does not know** the direction of $\vec E$; it measures the azimuth of the scattered photon in its own reference frame (for RHESSI, the spacecraft roll angle plus which rear detector segment recorded the event). Call that measured azimuth $\psi$, and call $\phi$ the (unknown) angle of the polarization plane in the same frame. Then

$$
\eta=\psi-\phi .
$$

Using $\cos(x+\pi) = -\cos x$:

$$
-\cos 2(\psi-\phi) = \cos \left[2(\psi-\phi)+\pi\right]=\cos \left[2\left(\psi-\phi+\frac{\pi}{2}\right)\right].
$$

So

$$
\frac{d\sigma}{d\Omega}= b(\theta)+a(\theta)\cos \left[2\left(\psi-\phi+\frac{\pi}{2}\right)\right].
$$

**This is where the $+\pi/2$ in eq. (3) comes from:** it absorbs the minus sign so that $A$ is positive. The paper reuses the letter $\eta$ in eq. (3), but there it means the azimuth measured in the instrument frame ($\psi$ here). Consequence: the **minimum** of the counts occurs at $\psi=\phi$, i.e. the minimum of the curve marks the plane of polarization (as Fig. 1 states).

---

## 4. "Integrated": sum over $\theta$ (and over the geometry)

A real detector does not select a single $\theta$; it accepts a range of scattering angles (for RHESSI, around $90^\circ$, from the Be scatterer to the rear Ge segments). The counts in an azimuth bin are

$$
C(\psi)\propto N_\gamma\int \varepsilon(\theta)\,\frac{d\sigma}{d\Omega}(\theta,\psi)\,\sin\theta\,d\theta ,
$$

with $\varepsilon(\theta)$ the geometric efficiency/acceptance. Since $\psi$ only appears in the $\cos 2(\dots)$ factor, it comes out of the integral:

$$
C(\psi)=A\cos \left[2\left(\psi-\phi+\frac{\pi}{2}\right)\right]+B,
$$

$$
A\propto\int\varepsilon\,a(\theta)\sin\theta\,d\theta,\qquad
B\propto\int\varepsilon\,b(\theta)\sin\theta\,d\theta .
$$

**This is eq. (3).** In practice $A$ and $B$ are fitted to the data (and the distribution is corrected with the response to an unpolarized beam, as the paper says) rather than computed by hand.

---

## 5. Partially polarized beam (links to eqs. 4–6)

A beam with degree of linear polarization $P$ is a mixture: a fraction $P$ fully polarized at angle $\phi$ plus a fraction $1-P$ unpolarized. For the unpolarized part we average over all orientations of $\vec E$: $\langle\cos 2\eta\rangle=0$, so it only contributes to $B$. Result:

$$
\frac{d\sigma}{d\Omega}= b(\theta)+P\,a(\theta)\cos \left[2\left(\psi-\phi+\frac{\pi}{2}\right)\right].
$$

So $B$ is unchanged and $A_p = P\,A_{100}$. With the paper's definitions,

$$
\mu_p=\frac{A_p}{B_p}=P\,\frac{A_{100}}{B_{100}}=P\,\mu_{100}\quad\Longrightarrow\quad P=\frac{\mu_p}{\mu_{100}}\quad\text{(eq. 6).}
$$

For a single $\theta$, the ideal modulation factor is

$$
\mu_{100}(\theta)=\frac{a}{b}=\frac{\sin^2\theta}{\dfrac{\nu_0}{\nu}+\dfrac{\nu}{\nu_0}-\sin^2\theta}.
$$

At $\theta=90^\circ$: $\mu_{100}\approx 0.999$ (20 keV), $0.991$ (50 keV), $0.969$ (100 keV). RHESSI's actual values (Fig. 7) are much lower because of the finite range of $\theta$, the broad beam, and unmodulated scattered photons.

---

## 6. Relation to Stokes parameters ($Q/I$, $U/I$), for the pipeline

**Step 1: write eq. (3) with the minus sign back.** Since $\cos(x+\pi)=-\cos x$,

$$
C(\psi)=B+A\cos \left[2\left(\psi-\phi+\tfrac{\pi}{2}\right)\right]=B-A\cos 2(\psi-\phi).
$$

**Step 2: put $P$ in.** From eqs. (4)–(6), $A/B=\mu_p=P\,\mu_{100}$, so $A=B\,\mu_{100}P$:

$$
C(\psi)=B\left[1-\mu_{100}\,P\cos 2(\psi-\phi)\right].
$$

**Step 3: expand the cosine.** With $\cos(x-y)=\cos x\cos y+\sin x\sin y$:

$$
P\cos 2(\psi-\phi)=(P\cos2\phi)\cos2\psi+(P\sin2\phi)\sin2\psi .
$$

**Step 4: recognize the Stokes parameters.** For linear polarization, $I$ is the total intensity and

$$
Q=I_{0^\circ}-I_{90^\circ},\qquad U=I_{45^\circ}-I_{135^\circ},
$$

where $I_\alpha$ is the intensity transmitted by an ideal polarizer at angle $\alpha$. For a beam with degree $P$ and angle $\phi$ (Malus' law on the polarized part, half of the unpolarized part passes):

$$
I_\alpha=\frac{I}{2}\left[1+P\cos 2(\alpha-\phi)\right].
$$

Plugging in the four angles: $I_{0}=\tfrac I2(1+P\cos2\phi)$, $I_{90}=\tfrac I2(1-P\cos2\phi)$, $I_{45}=\tfrac I2(1+P\sin2\phi)$, $I_{135}=\tfrac I2(1-P\sin2\phi)$. Subtracting the pairs:

$$
\frac{Q}{I}=P\cos2\phi,\qquad \frac{U}{I}=P\sin2\phi,\qquad P=\frac{\sqrt{Q^2+U^2}}{I},\qquad \phi=\tfrac12\,\mathrm{atan2}(U,Q).
$$

**Step 5: substitute.** Steps 2–4 give

$$
C(\psi)=B\left[1-\mu_{100}\left(\frac{Q}{I}\cos 2\psi+\frac{U}{I}\sin 2\psi\right)\right].
$$

**Step 6: what you fit.** Write the measured histogram as a linear model $C(\psi)=c_0+c_1\cos2\psi+c_2\sin2\psi$ (ordinary linear least squares, no angle to guess). Comparing term by term:

$$
c_0=B,\qquad c_1=-B\,\mu_{100}\frac{Q}{I},\qquad c_2=-B\,\mu_{100}\frac{U}{I}
\quad\Longrightarrow\quad
\frac{Q}{I}=-\frac{c_1}{\mu_{100}\,c_0},\qquad \frac{U}{I}=-\frac{c_2}{\mu_{100}\,c_0}.
$$

So the $\cos2\psi$ amplitude (divided by the mean and by $\mu_{100}$) is $Q/I$, and the $\sin2\psi$ amplitude is $U/I$. The minus sign is there because counts are *lowest* along $\vec E$.

**Sign check.** If $\vec E$ lies along $\psi=0$ ($\phi=0$): $Q/I=P>0$, $U=0$, and $C(0)=B(1-\mu_{100}P)$ is the minimum, as it should be. If $\vec E$ is along $\psi=90^\circ$: $Q/I=-P$ and the minimum moves to $90^\circ$. The sign of $Q$ therefore depends on which axis you call $\psi=0$; you must use the same reference axis as your model's Stokes convention.

**Note for the Fokker–Planck pipeline (U = 0).** In the beam model, Q is referenced to the direction perpendicular to the plane containing the magnetic field and the line of sight (Komarov et al. 2016 convention), and $U=0$ to numerical precision because electrons at opposite gyrophases cancel the $\sin 2\chi$ term. So the model only needs $Q/I$, and negative $Q/I$ means $\vec E$ parallel to the field's projection on the sky. An instrument, however, measures $\psi$ from its own fixed axis (e.g. solar north), which in general is rotated by some angle $\zeta$ from the model's Q axis. Rotating the reference axis mixes the two:

$$
Q'=Q\cos2\zeta+U\sin2\zeta,\qquad U'=-Q\sin2\zeta+U\cos2\zeta ,
$$

so with $U=0$ the instrument sees $Q'=Q\cos2\zeta$ and $U'=-Q\sin2\zeta\neq0$. To compare model and data, either rotate the measured $(Q',U')$ by $\zeta$ into the model frame (then the measured $U$ should come out $\approx0$, a useful consistency check), or choose $\psi=0$ along the model's Q axis, so that $C(\psi)=B\,[1-\mu_{100}(Q/I)\cos2\psi]$.
