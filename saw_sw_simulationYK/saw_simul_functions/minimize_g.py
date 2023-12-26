import numpy as np
from scipy.optimize import minimize

def MinimizeG(Input):
   mue0H = abs(Input[0])        # Magnitude of ext field
   phiH = Input[1]              # Direction of ext field, input already in rad
   AniType = Input[2]           # For the switch below
   phiu = np.deg2rad(Input[3])  # Anisotropy Axis
   mue0Hani = Input[4]          # Magnitude of Anisotropy field

   # Convert H and phiH into proper polar coordinates

   if Input[0] < 0:
       phiH = phiH + np.pi      # Converting annoying coordinate system into polar coordinates
   if phiH < 0:
       phiH = phiH + 2*np.pi

   if Input[0] == 0:            # Handling the zero-field case
       phi0 = phiu
   else:
       phi = np.linspace(0, 2*np.pi, 361)       # YK is lazy, so simply iterate over angles and pick the one with minimum energy
       G_Zeeman = -mue0H * np.cos(phiH - phi)
       if AniType == 0:
           G_Ani = - mue0Hani/2 * (np.cos(phiu - phi))**2
       elif AniType == 1:
           G_Ani = - mue0Hani/2 * (np.cos(phiu - phi))**2 * (np.sin(phiu - phi))**2
       elif AniType == 2:
           G_Ani = - mue0Hani/2 * (np.cos(3*(phiu - phi)/2))**2
       elif AniType == 3:
           G_Ani = - mue0Hani/2 * (np.cos(3*(phiu - phi)))**2
       G = G_Zeeman + G_Ani
       min_idx = np.argmin(G)

    #    phi0 = np.rad2deg(phi[min_idx])  # Equilibrium angle in deg
       phi0 = phi[min_idx]

   return phi0


def MinimizeGJu(Input):
    mue0H = Input[0]       # Magnitude of ext field
    phiH = Input[1]              # Direction of ext field, input already in rad
    AniType = Input[2]           # For the switch below
    phiu = np.deg2rad(Input[3])  # Anisotropy Axis
    mue0Hani = Input[4]          # Magnitude of Anisotropy field


    # Initial guess for the minimum
    if mue0H > 0:
            initial_guess = phiH
    else:
            initial_guess = phiH + np.pi
    
    # Use np.minimize_scalar to find the minimum
    result = minimize(total_energy, initial_guess, args=(mue0H, phiH, AniType, mue0Hani, phiu), tol=1e-1) #method='Nelder-Mead',

    # Extract the result
    phi0_minimized = result.x[0]
    
    return phi0_minimized

   

# Define the function to minimize
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


def MinimizeGFast(field, angle):

    if field > 0:
        phi0 = angle
    else:
        phi0 = angle + np.pi
    
    return phi0












'''
def free_enthalpy_density(m, H_amplitude, H_angle, Bd, Bu, Hex, u):
    H_external = H_amplitude * np.array([np.cos(H_angle), np.sin(H_angle), 0.0])
    return -np.dot(H_external, m) + Bd * m[2]**2 + Bu * np.dot(m, u)**2 - np.dot(Hex, m)

def equilibrium_direction(H_amplitude, H_angle, Bd, Bu, Hex, u):
    # Define the objective function to minimize
    objective_function = lambda m: free_enthalpy_density(m, H_amplitude, H_angle, Bd, Bu, Hex, u)

    # Initial guess for the magnetization direction
    initial_guess = np.array([0.0, 0.0, 1.0])

    # Minimize the objective function to find equilibrium direction
    result = minimize(objective_function, initial_guess, method='BFGS')

    # Extract the equilibrium direction
    equilibrium_direction = result.x / np.linalg.norm(result.x)

    return equilibrium_direction

# Example parameters
H_amplitude = 1.0
H_angle = np.pi/4  # 45 degrees
Bd_shape_anisotropy = 1.0
Bu_uniaxial_anisotropy = 1.0
Hex_exchange_field = np.array([0.0, 0.0, 0.0])  # Exchange field
unit_vector_u = np.array([1.0, 0.0, 0.0])  # Unit vector defining anisotropy direction

# Find equilibrium direction
equilibrium_dir = equilibrium_direction(H_amplitude, H_angle, Bd_shape_anisotropy, Bu_uniaxial_anisotropy, Hex_exchange_field, unit_vector_u)

print("Equilibrium Direction:", equilibrium_dir)
'''


'''
import numpy as np
from scipy.optimize import minimize

def G(m_xyz, H_123, Ms, Hk, Hani, u_123):
    m_123 = m_xyz[2]
    u123_dot_m123 = np.dot(u_123, m_123)
    
    G_value = -np.dot(H_123, m_123) + ((np.dot(Ms, Ms) / 2) - (np.dot(Hk, Hk) / 2)) * m_123**2 - (Hani**2 * u123_dot_m123**2)
    
    return G_value

def G_test(phi0, phiH1, params):
    m_xyz = np.array([0, 0, 1])  # Assuming m1 = m2 = 0, m3 = 1
    params["\[Phi]H"], params["\[Mu]0Hmag"] = phiH1, params["\[Mu]0Hmag1"]
    G_value = G(m_xyz, **params)
    return G_value

def FindEqM(field, angle, theata, params):
    Gtest = lambda phi0: G_test(phi0, theata, params)
    
    # Finding minimum of Gtest using scipy's minimize
    result = minimize(Gtest, x0=np.pi/2, bounds=[(0, np.pi)])  # Assuming \[Theta]1 = \[Pi]/2
    
    # Extracting \[Phi]1 from the result
    phi0 = result.x[0]
    
    # Returning the equilibrium direction as spherical coordinates
    return phi0

# Example parameters
params = {
    "\[Mu]0Hmag1": 1.0,
    "\[Theta]H1": np.pi/4,
    "\[Phi]H1": np.pi/6,
    "H_123": np.array([Hx, Hy, Hz]),  # External magnetic field
    "Ms": saturation_magnetization,  # Saturation magnetization
    "Hk": anisotropy_field,  # Anisotropy field
    "u_123": np.array([ux, uy, uz]),  # Anisotropy direction
}

# Call the FindEqM function
equilibrium_direction = FindEqM(params["\[Mu]0Hmag1"], params["\[Theta]H1"], params["\[Phi]H1"], params)

print("Equilibrium Direction:", equilibrium_direction)

'''