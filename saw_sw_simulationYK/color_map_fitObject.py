
import os
from my_modules import *
from saw_simul_functions import *
from scipy.optimize import curve_fit
# from multiprocessing import Pool
from sklearn.metrics import r2_score
# import platform
import numpy as np
from collections import defaultdict


name = 'Rayleigh'
name = 'Sezawa'


input_folder = '../dataForPython/Field_Angle_Sweep#3'

mySWdirectory = {}
P_absDict = {}

def InitialGuess():
    # alpha = 0.005
    alpha = 5.13e-3
    # alpha = 0.01
    b1 = 3.5
    b2 = 7
    if name == 'Rayleigh':
        eps = {
                'xx': 0.39-0.69j,
                'xy': 0,
                'xz': 1.00+0.56j,
                'yz': -0.02-0.02j
            }
    elif name =='Sezawa':
        eps = {
            'xx': 0.25+0.02j,
            'xy': 0.01+0j,
            'xz': 0.09-1.00j,
            'yz': 0.02+0.02j
        }
    eps['xx'] =  (68.5599918997257+17.122278103400706j)
    eps['xy'] =  (7.518160375585753-30.322126695128436j)
    eps['xz'] =  (17.754438781091295-84.71784906405115j)
    eps['yz'] =  (12.629232043550473+4.6202909648719j)


    eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = SplitEps(eps)
    return alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi

def main():
    Angles, Fields, params = GetParams(name)
    params_ = params[:10]

    Fields, Angles, P_abs = loadData()

    initial_guess = InitialGuess()

    # X, Y, Z = CreateMatrix(Fields, Angles, P_abs)
    # Z = FillMatrix(Z)
    # cmPlot(Z, Fields, Angles)
    
    prepareObjects(Fields, Angles, P_abs, params_, initial_guess)
    
    
    
    fitparams, covariance = curve_fit(CalculationSweep, mySWdirectory.keys(), P_abs, p0=initial_guess)

    alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)

    errors = np.sqrt(np.diag(covariance))
    alpha_err, b1_err, b2_err, eps_xxr_err, eps_xxi_err, eps_xyr_err, eps_xyi_err, eps_xzr_err, eps_xzi_err, eps_yzr_err, eps_yzi_err = errors
    eps_err = MergeEps(eps_xxr_err, eps_xxi_err, eps_xyr_err, eps_xyi_err, eps_xzr_err, eps_xzi_err, eps_yzr_err, eps_yzi_err)

    print("alpha = ", alpha)
    print("b1 = ", b1)    
    print("b2 = ", b2)
    print("eps['xx'] = ", eps['xx'])
    print("eps['xy'] = ", eps['xy'])
    print("eps['xz'] = ", eps['xz'])
    print("eps['yz'] = ", eps['yz'])

    print("alpha_err = ", alpha_err)
    print("b1_err = ", b1_err)
    print("b2_err = ", b2_err)
    print("eps_err['xx'] = ", eps_err['xx'])
    print("eps_err['xy'] = ", eps_err['xy'])
    print("eps_err['xz'] = ", eps_err['xz'])
    print("eps_err['yz'] = ", eps_err['yz'])


    compare(Fields, Angles, P_abs, fitparams, name=name, sign='pos', save=False, show=True)
    compare(Fields, Angles, P_abs, fitparams, name=name, sign='neg', save=False, show=True)
    # CalculateR2(Fields, Angles, P_abs, fitparams)
    
    
def CalculateR2(Fields, Angles, P_abs, fitparams):
    alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
    eps['yy'] = 0
    eps['zz'] = 0
    eps = eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz']

    P_absFittet = {}
    scaled_P_abs = {}

    for i, field in enumerate(Fields):
        angle = Angles[i]
        P_absDict[field, angle] = P_abs[i]
        mySWdirectory[field, angle].calcH_dr(b1, b2, eps)
        mySWdirectory[field, angle].calcP_abs()
        P_absFittet[field, angle] = mySWdirectory[field, angle].P_abs

    max_value = max(P_absFittet.values())
    min_value = min(P_absFittet.values())
    for (field, angle) in P_absFittet.keys():
        scaled_P_abs[field, angle] = (P_absFittet[field, angle] - min_value) / (max_value - min_value)
    fitted_P_absArray = np.array(list(scaled_P_abs.values()))

    r2 = r2_score(P_abs, fitted_P_absArray)
    print('r2 = ', r2)

def compare(Fields, Angles, P_abs, fitparams, name='Rayleigh', sign='pos', save=False, show=True):
    alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
    eps['yy'] = 0
    eps['zz'] = 0
    eps = eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz']

    if sign=='pos': mask = np.array(Fields) > 0
    elif sign=='neg': mask = np.array(Fields) < 0
    Fields = np.array(Fields)[mask]
    Angles = np.array(Angles)[mask]
    P_abs = np.array(P_abs)[mask]
    
    angle_means = defaultdict(float)
    angle_counts = defaultdict(int)
    for angle, p_abs in zip(Angles, P_abs):
        angle_means[angle] += p_abs
        angle_counts[angle] += 1
    
    for angle in angle_means:
        angle_means[angle] /= angle_counts[angle]
    
    mygraph = Graph(width_cm=8.2)
    mygraph.add_scatter(angle_means.keys(), angle_means.values())

    P_absFittet = {}
    scaled_P_abs = {}

    for i, field in enumerate(Fields):
        angle = Angles[i]
        P_absDict[field, angle] = P_abs[i]
        mySWdirectory[field, angle].calcH_dr(b1, b2, eps)
        mySWdirectory[field, angle].calcP_abs()
        P_absFittet[field, angle] = mySWdirectory[field, angle].P_abs

    max_value = max(P_absFittet.values())
    min_value = min(P_absFittet.values())
    for (field, angle) in P_absFittet.keys():
        scaled_P_abs[field, angle] = (P_absFittet[field, angle] - min_value) / (max_value - min_value)
    fitted_P_absArray = np.array(list(scaled_P_abs.values()))

    angle_means = defaultdict(float)
    angle_counts = defaultdict(int)
    for angle, p_abs in sorted(zip(Angles, fitted_P_absArray)):
        angle_means[angle] += p_abs
        angle_counts[angle] += 1

    for angle in angle_means:
        angle_means[angle] /= angle_counts[angle]

    mygraph.add_plot(sorted(angle_means.keys()), angle_means.values())

    mygraph.plot_Graph()


def CalculationSweep(keys, *fitparams):
    alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = fitparams
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
    eps['yy'] = 0
    eps['zz'] = 0
    eps = eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz']  


    for field, angle in mySWdirectory.keys():
        mySWdirectory[field, angle].calcChi(alpha)           # only for fitting alpha
        # mySWdirectory[field, angle].calcH_dr(b1=3.5, b2=7, eps=eps)
        mySWdirectory[field, angle].calcP_abs()
    
    P_absFittet = {}
    scaled_P_abs = {}

    for (field, angle) in mySWdirectory.keys():
        P_absFittet[field, angle] = mySWdirectory[field, angle].P_abs

    max_value = max(P_absFittet.values())
    min_value = min(P_absFittet.values())

    for (field, angle) in P_absFittet.keys():
        scaled_P_abs[field, angle] = (P_absFittet[field, angle] - min_value) / (max_value - min_value)

    scaled_P_absArray = np.array(list(scaled_P_abs.values()))
    # P_abs = np.array(P_abs)

    # scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

    return scaled_P_absArray




def prepareObjects(Fields, Angles, P_abs, params_, initial_guess):
    alpha, b1, b2, eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi = initial_guess
    eps = MergeEps(eps_xxr, eps_xxi, eps_xyr, eps_xyi, eps_xzr, eps_xzi, eps_yzr, eps_yzi)
    eps['yy'] = 0
    eps['zz'] = 0
    eps = eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz'] 

    for i, field in enumerate(Fields):
        angle = Angles[i]
        P_absDict[field, angle] = P_abs[i]
        mySWdirectory[field, angle] = SWcalculatorSingle(field, angle, params_)
        mySWdirectory[field, angle].calcPhi0()
        mySWdirectory[field, angle].calcChi()
        mySWdirectory[field, angle].calcH_dr(b1, b2, eps)


def calculateP_abs(args):
    field, angle, eps = args
    mySWdirectory[field, angle].calcH_dr(eps)
    mySWdirectory[field, angle].calcP_abs()


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



def loadData():
    # Define file paths
    if name == 'Rayleigh': filename = 'ResS21_Rayleigh.txt'
    elif name == 'Sezawa': filename = 'ResS21_Sezawa.txt'
    else: print('no file for this name')
    filepath = os.path.join(input_folder, filename)
    # Read the data from the .txt file
    data = np.loadtxt(filepath, dtype=float, skiprows=1)

    #Extract columns
    Fields = data[:, 0]
    Angles = data[:, 1]
    P_abs = data[:, 2]

    # Fields, Angles, P_abs, = SliceData(Fields, Angles, P_abs)

    # Apply MinMaxScaling
    # scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

    return Fields, Angles, P_abs

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