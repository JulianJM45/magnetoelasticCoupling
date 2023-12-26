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

    eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz = Input[1:7]
    b1, b2 = Input[7:9]

    cos_phi0 = np.cos(phi0)
    sin_phi0 = np.sin(phi0)
    cos_2_phi0 = np.cos(2 * phi0)

    h_dr_1 = 2 * b2 * (eps_xz * cos_phi0 + eps_yz * sin_phi0)   # oop component

    h_dr_2 = 2 * b1 * (eps_xx * cos_phi0 * sin_phi0 - eps_yy * cos_phi0 * sin_phi0) - 2 * b2 * eps_xy * cos_2_phi0  # ip component

    h_dr = np.array([h_dr_1, h_dr_2])

    return h_dr


