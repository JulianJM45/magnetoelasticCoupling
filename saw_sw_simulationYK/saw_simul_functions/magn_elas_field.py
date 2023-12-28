import numpy as np
'''
def MagnElasField(Input):
    phi0 = np.deg2rad(Input[0])

    eps_xx = Input[1]
    eps_yy = Input[2]
    eps_zz = Input[3]
    eps_xy = Input[4]
    eps_xz = Input[5]
    eps_yz = Input[6]

    b1 = Input[7]
    b2 = Input[8]

    h_dr_1 = 2 * b2 * (eps_xz * np.cos(phi0) + eps_yz * np.sin(phi0))   # oop component

    h_dr_2 = 2 * b1 * (eps_xx * np.cos(phi0) * np.sin(phi0) - eps_yy * np.cos(phi0) * np.sin(phi0)) - 2 * b2 * eps_xy * np.cos(2*phi0)  # ip component

    h_dr = np.array([h_dr_1, h_dr_2])

    return h_dr
'''

def MagnElasField(Input):
    # phi0 = np.deg2rad(Input[0])
    phi0 = Input[0]
    b1, b2 = Input[1:3]
    eps = Input[3]
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


