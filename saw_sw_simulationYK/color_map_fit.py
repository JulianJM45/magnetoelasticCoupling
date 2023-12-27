
import os
from my_modules import *
from saw_sw_simulationYK.saw_simul_functions import *
from scipy.optimize import curve_fit
from multiprocessing import Pool

name = 'Rayleigh'
if name == 'Rayleigh': input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\PostTG_Rayleigh.txt'


Angles, Fields, params = GetParams(name)
eps = {}
alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz'] = params

def InitialGuess():
    alpha = 0.008
    eps = {
            'xx': 0.39-0.69j,
            'xy': 0,
            'xz': 1.00+0.56j,
            'yz': -0.02-0.02j
        }
    eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = SplitEps(eps)
    return alpha, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi

def main():
    Fields, Angles, P_abs = loadData(input_filepath)

    initial_guess = InitialGuess()

    # X, Y, Z = CreateMatrix(Fields, Angles, P_abs)
    # Z = FillMatrix(Z)
    # cmPlot(Z, Fields, Angles)

    xdata = np.column_stack((Fields, Angles))

    fitparams, covariance = curve_fit(CalculationSweep, xdata, P_abs, p0=initial_guess)

    alpha, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)

    print("Alpha:", alpha)
    print("Eps_xx:", eps['xx'])
    print("Eps_xy:", eps['xy'])
    print("Eps_xz:", eps['xz'])
    print("Eps_yz:", eps['yz'])

# def CalculationSweep(x, *fitparams):
#     alpha, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
#     eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
#     eps['yy'] = 0
#     eps['zz'] = 0
#     params = alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'] 
#     P_abs =  np.array([singleCalculate(field, angle, params) for (field, angle) in x])

#     scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

#     return scaled_P_abs

def CalculationSweep(x, *fitparams):
   alpha, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
   eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
   eps['yy'] = 0
   eps['zz'] = 0
   params = alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'] 

   with Pool() as p:
       P_abs = p.map(singleCalculate, [(field, angle, params) for (field, angle) in x])

   P_abs = np.array(P_abs)

   scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

   return scaled_P_abs


def SplitEps(eps):
    eps_xxr = eps['xx'].real
    eps_xxi = eps['xx'].imag

    eps_xyr = eps['xy'].real
    eps_xyi = eps['xy'].imag

    eps_xzr = eps['xz'].real
    eps_xzi = eps['xz'].imag

    eps_yzr = eps['yz'].real
    eps_yzi = eps['yz'].imag

    return eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi

def MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi):
    eps = {
        'xx': complex(eps_xxr, eps_xxi),
        'xy': complex(eps_xyr, eps_xyi),
        'xz': complex(eps_xzr, eps_xzi),
        'yz': complex(eps_yzr, eps_yzi)
    }

    return eps



def loadData(input_filepath):
    # Read the data from the .txt file
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)

    #Extract columns
    Fields = data[:, 0]
    Angles = data[:, 1]
    P_abs = data[:, 2]

    Fields, Angles, P_abs, = SliceData(Fields, Angles, P_abs)

    # Apply MinMaxScaling
    scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

    return Fields, Angles, scaled_P_abs

def SliceData(Fields, Angles, P_abs):
    if name == 'Rayleigh':
        a, b = 6, 42
    else: print('bounds have to be set in SliceData')
    # Define the condition for slicing
    condition = np.logical_and(np.logical_or(Fields >= -b, Fields <= -a), np.logical_or(Fields >= a, Fields <= b))

    # Apply the condition to slice the data
    Fields = Fields[condition]
    Angles = Angles[condition]
    P_abs = P_abs[condition]

    return(Fields, Angles, P_abs)


def CreateMatrix(Fields, Angles, DeltaS12):
    unique_fields = np.unique(Fields)
    unique_angles = np.unique(Angles)
    # Create X and Y grids using numpy.meshgrid
    X, Y = np.meshgrid(unique_fields, unique_angles)

    # Initialize Z with None values to indicate empty fields
    Z = np.empty_like(X)

    # Create index maps for angles and fields
    angle_index_map = {angle: index for index, angle in enumerate(unique_angles)}
    field_index_map = {field: index for index, field in enumerate(unique_fields)}

    # Fill the Z-values with corresponding deltaS12 values using advanced indexing
    for angle, field, deltaS12 in zip(Angles, Fields, DeltaS12):
        angle_index = angle_index_map.get(angle)
        field_index = field_index_map.get(field)
        if angle_index is not None and field_index is not None:
            Z[angle_index, field_index] = deltaS12
    X = np.matrix(X)
    Y = np.matrix(Y)
    return X, Y, Z

def FillMatrix(Z):
    # Fill empty cells of Z
    for row_index in range(Z.shape[0]):
        row = Z[row_index, :]
        non_empty_indices = np.nonzero(row)[0]
        if non_empty_indices.size > 0:
            for i in range(Z.shape[1]):
                if row[i] == 0:
                    nearest_non_empty_index = non_empty_indices[np.argmin(np.abs(non_empty_indices - i))]
                    Z[row_index, i] = row[nearest_non_empty_index]

    return Z


if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()