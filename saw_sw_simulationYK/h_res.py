import numpy as np

# Constants
hbar = 1.05457173e-34
mue0 = 4 * np.pi * 1e-7
mueB = 9.27400968e-24


#  SAW 
k = 6.73e6
f = 2.45e9
# f = 3.53e9
omega0 = 2 * np.pi * f
# Structure Properties
Hk = 0
phiu = 0
Deff = 0

A = 3.7e-12

def allHres(Angles, t, g, mue0Hani, mue0Ms):

    """
    Calculate the magnetic field for different angles.

    Parameters:
    - Angles (array): Array of angles.
    - SAWparams (array): Array of parameters related to the SAW.
    - fitparams (array): Array of fitting parameters.

    Returns:
    - array: Array of magnetic field values.
    """

    # fitparams
    # A, g, mue0Hani, mue0Ms, Hk, phiu, Deff = fitparams
    # t = 1.0 * 100e-9
    # g = 2.0269
 

    Angles = np.radians(Angles)
    
    GSW = (1 - np.exp(-np.abs(k) * t)) / (np.abs(k) * t)
    Ms = mue0Ms / mue0
    Hani = mue0Hani / mue0
    gamma = g * (mueB / hbar)

    

    # Preallocate mue0Hres array
    allmue0Hres = np.zeros_like(Angles)

    # Calculation
    for i, phiH in enumerate(Angles):
        # Calculation
        allmue0Hres[i] = Hres(phiH, k, omega0, A, gamma, Hani, Ms, phiu, Deff, GSW, Hk)

    

    return allmue0Hres


def Hres(phiH, k, omega0, A, gamma, Hani, Ms, phiu, Deff, GSW, Hk):

    """
    Calculate the magnetic field for a specific angle.

    Parameters:
    - phiH (float): Angle.
    - k, omega0, A, gamma, Hani, Ms, phiu, Deff, GSW, Hk: Various parameters.

    Returns:
    - float: Magnetic field value.
    """

    mu0HDMI = (2 * Deff / Ms) * k * np.sin(phiH)
    mu0Hip = mue0 * (1 - GSW) * Ms * np.sin(phiH)**2 + (2 * A / Ms) * k**2 + mue0 * Hani * np.cos(2 * (phiH - phiu))
    mu0Hoop = mue0 * GSW * Ms - mue0 * Hk + (2 * A / Ms) * k**2 + mue0 * Hani * np.cos(phiH - phiu)**2

    # Calculation
    mue0Hres = 0.5 * (-mu0Hip - mu0Hoop + np.sqrt((mu0Hip - mu0Hoop)**2 + (4 * (gamma * mu0HDMI + omega0)**2) / gamma**2))

    return mue0Hres
