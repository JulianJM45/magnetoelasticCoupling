import numpy as np

def AbsPower(h, chi):
    # p0 = 1
    # c = inputs[0]
    # h = inputs[1]
    # chi = inputs[2]

    PAbs = np.imag(np.dot(h, np.dot(chi, np.conj(h).T))) / ((4 * np.pi * 1e-7) ** 2)

    return PAbs



