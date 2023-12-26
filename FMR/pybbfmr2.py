from nptdms import TdmsFile
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel
from lmfit import Model

# Load the TDMS data
tdms_file = TdmsFile.read("Angle_0.0.tdms")
group = tdms_file["Read.Group_1"]

# Define a fitting model (you can choose Gaussian or Lorentzian)
def fit_resonance(data, angle):
    x = data["Frequency"].time_track()
    y_real = data["S41_REAL"].time_track()
    y_imag = data["S41_IMAG"].time_track()
    amplitude = np.sqrt(y_real**2 + y_imag**2).max()
    center = x[np.argmax(np.sqrt(y_real**2 + y_imag**2))]
    model = GaussianModel(prefix="g1_") + GaussianModel(prefix="g2_")
    model.set_param_hint("g1_amplitude", value=amplitude, vary=True)
    model.set_param_hint("g2_amplitude", value=amplitude, vary=True)
    model.set_param_hint("g1_center", value=center, min=center-0.1, max=center+0.1)
    model.set_param_hint("g2_center", value=center, min=center-0.1, max=center+0.1)
    model.set_param_hint("g1_sigma", value=0.01, min=0.001, max=0.05)
    model.set_param_hint("g2_sigma", value=0.01, min=0.001, max=0.05)

    params = model.make_params()
    result = model.fit(np.sqrt(y_real**2 + y_imag**2), x=x, params=params)

    resonance_freq = result.best_values["g1_center"]
    bandwidth = result.best_values["g1_sigma"] + result.best_values["g2_sigma"]

    return angle, resonance_freq, bandwidth

# Perform the fitting for each angle
angle_resonance_bandwidth = []
for angle in [0.0, 15.0, 30.0]:  # Replace with your desired angles
    data = group  # Replace with data for the specific angle
    angle, resonance_freq, bandwidth = fit_resonance(data, angle)
    angle_resonance_bandwidth.append((angle, resonance_freq, bandwidth))

# Print or store the results
for angle, resonance_freq, bandwidth in angle_resonance_bandwidth:
    print(f"Angle {angle}: Resonance Frequency = {resonance_freq}, Bandwidth = {bandwidth}")
