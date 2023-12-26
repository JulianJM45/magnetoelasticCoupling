import numpy as np
from scipy.optimize import minimize

Angles = np.linspace(-90, 90, num=91)






def total_energy(phi, mue0H, phiH, AniType, mue0Hani, phiu):
    G_Zeeman = -mue0H * np.cos(phiH - phi)
    if AniType == 0:
        G_Ani = -mue0Hani/2 * (np.cos(phiu - phi))**2
    elif AniType == 1:
        G_Ani = -mue0Hani/2 * (np.cos(phiu - phi))**2 * (np.sin(phiu - phi))**2
    elif AniType == 2:
        G_Ani = -mue0Hani/2 * (np.cos(3*(phiu - phi)/2))**2
    elif AniType == 3:
        G_Ani = -mue0Hani/2 * (np.cos(3*(phiu - phi)))**2
    G = G_Zeeman + G_Ani
    return G