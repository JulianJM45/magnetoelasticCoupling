import numpy as np
'''
def MagnSusceptibility(Input):
    mue0 = 4 * np.pi * 1e-7
    mueB = 9.27400968 * 1e-24
    hbar = 1.05457173 * 1e-34

    H = abs(Input[0]) / mue0         # Ext field, Input in T, thus div by mue0
    phi0 = np.deg2rad(Input[1])      # Equilibrium direction in deg
    phiH = Input[2]                  # Angle of ext field in rad
    AniType = Input[12]              # AniType for the switch
    phiani = np.deg2rad(Input[3])    # Anisotropy Axis
    Hani = Input[4]                  # Magnitude of Anisotropy field
    A = Input[5] * 1e-12             # Exchange constant Converting to SI
    mue0Ms = Input[6]                # Sat Magn in T
    Ms = Input[6] / mue0              # conv to A/m
    alpha = Input[7]                 # Gilbert Damping
    g = Input[8]                     # g-factor

    k = Input[9]                      # k-vector
    omega = 2 * np.pi * Input[10]     # frequency

    t = Input[11]                     # Film thickness
    Hk = 0                            # Out-of-plane shape anisotropy -> set to zero for now
    gamma = g * mueB / hbar

    if Input[0] < 0:                  # Converting annoying coordinate system into polar coordinates
        phiH = phiH + np.pi
    if phiH < 0:
        phiH = phiH + 2 * np.pi
    if Input[0] == 0:                 # Calculating angles and handling zero-field case
        ang_phi0phi = phiani
        ang_phi0phiani = 0
    else:
        ang_phi0phi = phi0 - phiH      # Unfinished part but influence seems low
        ang_phi0phiani = 0
        if AniType == 0:
            ang = [phi0 - phiani + (i-1) * np.pi for i in range(1, 3)]
            ang_phi0phiani = 0
        elif AniType == 1:
            ang_phi0phiani = 0
        elif AniType == 2:
            ang_phi0phiani = 0
        elif AniType == 3:
            ang_phi0phiani = 0

    if k != 0:                        # in case of FMR pumping, avoid div by zero
        G0 = (1 - np.exp(-k * t)) / (k * t)
    else:
        G0 = 1

    chi_inv_11 = H * np.cos(ang_phi0phi) + 2 * A * k**2 / mue0Ms + Ms * G0 - Hk + Hani * np.cos(ang_phi0phiani)**2 + 1j * alpha * omega / (mue0 * gamma)
    chi_inv_12 = -1j * omega / (mue0 * gamma)
    chi_inv_21 = 1j * omega / (mue0 * gamma)
    chi_inv_22 = H * np.cos(ang_phi0phi) + 2 * A * k**2 / mue0Ms + Ms * (1 - G0) * np.sin(phi0)**2 + Hani * np.cos(2 * ang_phi0phiani) + 1j * alpha * omega / (mue0 * gamma)

    chi_inv = np.array([[chi_inv_11, chi_inv_12], [chi_inv_21, chi_inv_22]]) / Ms

    return chi_inv

'''

def MagnSusceptibility(Input):
    mue0 = 4 * np.pi * 1e-7
    mueB = 9.27400968e-24
    hbar = 1.05457173e-34

    H = np.abs(Input[0]) / mue0
    # phi0 = np.deg2rad(Input[1])
    phi0 = Input[1]
    phiH = Input[2]
    AniType = Input[12]
    phiani = np.deg2rad(Input[3])
    Hani = Input[4]
    A = Input[5] * 1e-12
    mue0Ms = Input[6]
    Ms = Input[6] / mue0
    alpha = Input[7]
    g = Input[8]
    k = Input[9]
    omega = 2 * np.pi * Input[10]
    t = Input[11]
    Hk = 0
    gamma = g * mueB / hbar

    if Input[0] < 0:
        phiH += np.pi
    if phiH < 0:
        phiH += 2 * np.pi
    if Input[0] == 0:
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

    chi_inv_11 = H * cos_ang_phi0phi + 2 * A * k**2 / mue0Ms + Ms * G0 - Hk + Hani * cos_ang_phi0phiani**2 + 1j * alpha * omega / (mue0 * gamma)
    chi_inv_12 = -1j * omega / (mue0 * gamma)
    chi_inv_21 = 1j * omega / (mue0 * gamma)
    chi_inv_22 = H * cos_ang_phi0phi + 2 * A * k**2 / mue0Ms + Ms * (1 - G0) * sin_phi0**2 + Hani * cos_2_ang_phi0phiani + 1j * alpha * omega / (mue0 * gamma)

    chi_inv = np.array([[chi_inv_11, chi_inv_12], [chi_inv_21, chi_inv_22]]) / Ms

    return chi_inv

