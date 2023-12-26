from bbfmr.measurement.vna import VNASeparateFieldsMeasurement
from bbfmr.measurement.loading import  reshape

import bbfmr.processing as bp
import matplotlib.pyplot as plt
from bbfmr.models.susceptibility import VNAFMR_EllipsoidDifferenceQuotientSphereModel
from nptdms import TdmsFile
import numpy as np


input_tdms_file_path = r'C:\Users\Julian\Documents\python\FMRTest.tdms'

x_val = -0.5

# Load the input TDMS file
input_tdms_file = TdmsFile.read(input_tdms_file_path)

fields = np.unique(input_tdms_file['SetSweep.']['Field (T)'].data)
numfields = len(fields)
frequencies = np.unique(input_tdms_file['Read.ZNA']['Frequency'].data)
numfreq = len(frequencies)
field_before_data = input_tdms_file['Read.readbefore']['Field (T)'].data
field_after_data = input_tdms_file['Read.readafter']['Field (T)'].data
frequency_data = input_tdms_file['Read.ZNA']['Frequency'].data
real_data = input_tdms_file['Read.ZNA']['S41_REAL'].data
imag_data = input_tdms_file['Read.ZNA']['S41_IMAG'].data


class CustomVNASeparateFieldsMeasurement(VNASeparateFieldsMeasurement):
    def __init__(self, field_before_data, field_after_data, frequency_data, real_data, imag_data, **kwargs):
        super(CustomVNASeparateFieldsMeasurement, self).__init__(**kwargs)
        self.load_data_from_arrays(field_before_data, field_after_data, frequency_data, real_data, imag_data)

    def load_data_from_arrays(self, field_before_data, field_after_data, frequency_data, real_data, imag_data):
        x = (field_before_data+field_after_data)/2
        y = frequency_data
        real_signal = real_data
        imag_signal = imag_data

        # Combine real and imaginary parts into a complex signal
        complex_signal = real_signal + 1j * imag_signal

        # You may need to adapt this part to match the structure of your data
        x, y, signal = reshape(x, y, complex_signal)

        self.set_XYZ(x, y, signal)


# Create an instance of CustomVNASeparateFieldsMeasurement
m = CustomVNASeparateFieldsMeasurement(
    field_before_data=field_before_data,
    field_after_data=field_after_data,
    frequency_data=frequency_data,
    real_data=real_data,
    imag_data=imag_data
)

m.set_xlabel("$\mu_0 H_0$ (T)")
m.set_ylabel("$\omega/2\pi$ (Hz)")
m.set_zlabel("Signal [a.u]")


# Check for x_val slicing
X=m.X
Y=m.Y
Z=m.Z

# Check for x_val slicing
if x_val is not None:
    x_idx = np.argmin(np.abs(X[:, 0] - x_val))
    if x_idx >= X.shape[0]:
        raise ValueError("x_val is out of bounds for X")
    

    X_slc=X[x_idx, slice(None)]

    # print(X_slc)
    # print(X_slc.shape)


# Plot colormap
ax_color = plt.subplot(211)
m.operations = []
m.add_operation(bp.derivative_divide, modulation_amp=4, delay=True)
m.add_operation(bp.real, delay=True)
m.add_operation(bp.limit, x_slc=slice(20, -400, 1))
mesh, cbar = m.plot(rasterized=True, cmap="Blues_r", ax=ax_color)



# Plot real and imaginary part for cut at constant field
ax_cut = plt.subplot(212)
for op in [bp.real, bp.imag]:
    m.operations = []
    m.add_operation(bp.derivative_divide, modulation_amp=4, delay=True)
    m.add_operation(bp.conjugate, delay=True)
    m.add_operation(bp.linear_moving_limit, intercept=-2.9610e+10, slope=2.8734e+10, span=500, delay=True)
    m.add_operation(bp.limit, x_slc=slice(20, -400, 1), delay=True)
    m.add_operation(bp.cut, x_val=x_val, delay=True)
    m.add_operation(op)
    m.plot(ax=ax_cut, marker='s', markersize=3)

# Fit susceptibility model to cut and plot best fit
m.operations.pop()  # Remove last operation
m.process()  # Recalculate m.X, m.Y, m.Z
B = np.squeeze(np.unique(m.X))
f = np.squeeze(m.Y)
S = np.squeeze(m.Z)  # complex-valued

model = VNAFMR_EllipsoidDifferenceQuotientSphereModel()
params = model.guess(x=f, data=S, mod_f=2.4e-3 * 28e9, B=B, phi=-1.6)
params["phi"].vary = True
params["Z"].value = params["Z"].value / 100
roi = model.recommended_roi(f, S, 3)
# fit = model.fit(x=f[roi], data=S[roi], params=params)

# ax_cut.plot(f[roi], np.real(fit.best_fit), '-')
# ax_cut.plot(f[roi], np.imag(fit.best_fit), '-')
# ax_cut.set_xlim([fit.params["f_r"].value - 0.6e9, fit.params["f_r"].value + 0.6e9])

# Show the plots
plt.show()
