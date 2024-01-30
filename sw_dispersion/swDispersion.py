import numpy as np
from my_modules import *

mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
mu_B = 9.2740100783e-24  # Bohr magneton
hbar = 1.054571817e-34  # Reduced Planck constant


def calculate_omega_SW(k, H0=0.1, g=2, A=3.7e-12, Ms=0.18, t=100e-9, phi=0):
    H0 = H0 / mu_0
    Ms = Ms / mu_0
    phi = np.deg2rad(phi)

    H_Ex = (2 * A / (mu_0 * Ms)) * k**2
    H_Dip_x = Ms * (1 - np.exp(-k * t)) / (k * t)
    H_Dip_y = Ms * (1 - (1 - np.exp(-k * t))/(k * t) ) * np.sin(phi)**2

    omega_SW = (g * mu_0 * mu_B / hbar) * np.sqrt((H0 + H_Ex + H_Dip_x) * (H0 + H_Ex + H_Dip_y))
    # return omega_SW
    f = omega_SW / (2 * np.pi)
    return f

mygraph = Graph()
k = np.linspace(1e6, 12e6, 1000)

omega_SW = calculate_omega_SW(k, H0=0.02)
mygraph.add_plot(k*1e-6, omega_SW*1e-9, label='$H_0 = 20$\u2009mT', color='red')

omega_SW = calculate_omega_SW(k, H0=0.04)
mygraph.add_plot(k*1e-6, omega_SW*1e-9, label='$H_0 = 40$\u2009mT', color='blue')

omega_SW = calculate_omega_SW(k, H0=0.06)
mygraph.add_plot(k*1e-6, omega_SW*1e-9, label='$H_0 = 60$\u2009mT', color='green')

omega_SW = calculate_omega_SW(k, H0=0.08)
mygraph.add_plot(k*1e-6, omega_SW*1e-9, label='$H_0 = 80$\u2009mT', color='yellow')


mygraph.plot_Graph(xlabel='$k$\u2009$10^6$(1/m)', ylabel='$f_\mathrm{SW}$\u2009(GHz)', legend=True, save=False)