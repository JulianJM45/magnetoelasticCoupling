import numpy as np

mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
mu_B = 9.2740100783e-24  # Bohr magneton
hbar = 1.054571817e-34  # Reduced Planck constant


#  SAW 
k = 6.73e6
f = 2.45e9
f = 3.53e9
# Structure Properties
Hk = 0
Deff = 0

g=2.026899502405165
mu_0Ms=0.18381376682738004

# g=2.013
mu_0Ms=0.17236


def allHres(Angles, phiu=-0.99, mu_0Hani=0.002, t=89.19e-9, A=4.022e-12, mu_0Ms=mu_0Ms, g=g):
    Ms = mu_0Ms / mu_0
    Hani = mu_0Hani / mu_0
    
    GSW = (1 - np.exp(-np.abs(k) * t)) / (np.abs(k) * t)

    allmue0Hres = np.zeros_like(Angles)

    for i, phiH in enumerate(Angles):
        allmue0Hres[i] = Hres(phiH, k, A, Hani, Ms, phiu, Deff, GSW, Hk)
        allmue0Hres[i] = Hres(phiH=phiH, k=k, A=A, g=g, Hani=Hani, Ms=Ms, phiu=phiu, Deff=Deff, GSW=GSW, Hk=Hk, f=f)

    return allmue0Hres


def Hres(phiH=0, k=k, A=4.02e-12, g=2.013, Hani=0.00219/mu_0, Ms=0.17236/mu_0, phiu=-0.99, Deff=0, GSW=None, t=89e-9, Hk=0, f=2.45e9):
    if GSW is None:
        GSW = (1 - np.exp(-np.abs(k) * t)) / (np.abs(k) * t)
    gamma = g * (mu_B / hbar)
    omega0 = 2 * np.pi * f
    phiH = np.radians(phiH)
    phiu = np.radians(phiu)
    mu0HDMI = (2 * Deff / Ms) * k * np.sin(phiH)
    mu0Hip = mu_0 * (1 - GSW) * Ms * np.sin(phiH)**2 + (2 * A / Ms) * k**2 + mu_0 * Hani * np.cos(2 * (phiH - phiu))
    mu0Hoop = mu_0 * GSW * Ms - mu_0 * Hk + (2 * A / Ms) * k**2 + mu_0 * Hani * np.cos(phiH - phiu)**2

    mue0Hres = 0.5 * (-mu0Hip - mu0Hoop + np.sqrt((mu0Hip - mu0Hoop)**2 + (4 * (gamma * mu0HDMI + omega0)**2) / gamma**2))

    return mue0Hres
