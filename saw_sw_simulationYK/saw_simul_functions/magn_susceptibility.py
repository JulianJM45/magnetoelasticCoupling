import numpy as np

mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
mu_B = 9.2740100783e-24  # Bohr magneton
hbar = 1.054571817e-34  # Reduced Planck constant


def MagnSusceptibility(params):
    field, phi0, phiH, phiani, Hani, A, mue0Ms, alpha, g, k, f, t, AniType = params

    H = np.abs(field) / mu_0
    # A = A * 1e-12
    Ms = mue0Ms / mu_0
    gamma = g * mu_B / hbar
    omega = 2 * np.pi * f

    Hk = 0

    if field < 0:
        phiH += np.pi
    if phiH < 0:
        phiH += 2 * np.pi
    if field == 0:
        ang_phi0phi = phiani
        ang_phi0phiani = 0
    else:
        ang_phi0phi = phi0 - phiH
        ang_phi0phiani = 0
        if AniType == 0:
            ang = [phi0 - phiani + (i-1) * np.pi for i in range(1, 3)]
            ang_phi0phiani = 0
        elif AniType in (1, 2, 3):
            ang_phi0phiani = 0

    G0 = (1 - np.exp(-k * t)) / (k * t) if k != 0 else 1

    cos_ang_phi0phi = np.cos(ang_phi0phi)
    cos_ang_phi0phiani = np.cos(ang_phi0phiani)
    sin_phi0 = np.sin(phi0)
    cos_2_ang_phi0phiani = np.cos(2 * ang_phi0phiani)

    chi_inv_11 = H * cos_ang_phi0phi + 2 * A * k**2 / mue0Ms + Ms * G0 - Hk + Hani * cos_ang_phi0phiani**2 + 1j * alpha * omega / (mu_0 * gamma)
    chi_inv_12 = -1j * omega / (mu_0 * gamma)
    chi_inv_21 = 1j * omega / (mu_0 * gamma)
    chi_inv_22 = H * cos_ang_phi0phi + 2 * A * k**2 / mue0Ms + Ms * (1 - G0) * sin_phi0**2 + Hani * cos_2_ang_phi0phiani + 1j * alpha * omega / (mu_0 * gamma)

    chi_inv = np.array([[chi_inv_11, chi_inv_12], [chi_inv_21, chi_inv_22]]) / Ms

    return chi_inv   



def MagnSusceptibilityNEU(params):
    # field, phi0, phiH, phiani, Hani, A, mue0Ms, alpha, g, k, f, t, AniType = params
    field, phi0, angle, phiu, mu_0Hani, A, mu_0Ms, alpha, g, k, f, t, AniType = params

    mu_0H = np.abs(field) 
    Ds = 2 * A * g * mu_B / mu_0Ms
    Bd = mu_0Ms / 2
    Bu = mu_0Hani

    gamma = g * mu_B / hbar
    omega = 2 * np.pi * f

    u1 = 0
    u2 = np.sin(phi0-phiu)
    u3 = np.cos(phi0-phiu)

    G3 = -mu_0H + 2*Bu*u3**2 + Ds*k**2 
    G12 = 2*Bu*u1*u2 
    G11 = 2*Bd + 2*Bu*u1**2 
    G22 = 2*Bu*u2**2

    D = (G11 - G3 - 1j*omega*alpha/gamma) * (G22 - G3 - 1j*omega*alpha/gamma) - G12**2 - (omega/gamma)**2

    chi_11 = G22 - G3 - 1j*omega*alpha/gamma
    chi_12 = 1j*omega/gamma
    chi_21 = - 1j*omega/gamma
    chi_22 = G11 - G3 - 1j*omega*alpha/gamma
    
    chi = np.array([[chi_11, chi_12], [chi_21, chi_22]]) * mu_0Ms / D

    return chi