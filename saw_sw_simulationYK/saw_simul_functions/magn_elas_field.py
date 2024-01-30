import numpy as np

mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability

def MagnElasField(params):
    phi0, b1, b2, eps = params

    # phi0 = np.deg2rad(phi0)

    if isinstance(eps, dict):
        xx, yy, zz, xy, xz, yz = eps.values()
    else: xx, yy, zz, xy, xz, yz = eps
    
    cos_phi0 = np.cos(phi0)
    sin_phi0 = np.sin(phi0)
    cos_2_phi0 = np.cos(2 * phi0)

    h_dr_1 = 2 * b2 * (xz * cos_phi0 + yz * sin_phi0)   # oop component

    h_dr_2 = 2 * b1 * (xx * cos_phi0 * sin_phi0 - yy * cos_phi0 * sin_phi0) - 2 * b2 * xy * cos_2_phi0  # ip component



    h_dr = np.array([h_dr_1, h_dr_2])

    return h_dr


