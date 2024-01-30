import os
from my_modules import *
from saw_simul_functions import *
import platform
import numpy as np
from collections import defaultdict



if platform.system() == 'Linux':
    input_folder = '/home/julian/Seafile/BAJulian/dataForPython/Field_Angle_Sweep#3'
    
elif platform.system() == 'Windows':
    input_folder = r'C:\Users\Julian\Seafile\BAJulian\dataForPython\Field_Angle_Sweep#3'

name = 'Rayleigh'
# if name == 'Rayleigh': input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\PostTG_Rayleigh.txt'


mySWdirectory = {}
P_absDict = {}

b1 =  3.088859456681142
b2 =  9.521573671152646
eps = {
            'xx': 0.3-0.76j,
            'yy': 0,
            'zz': 0,
            'xy': -0.06-0.06j,
            'xz': 0.69+0.51j,
            'yz': -0.16+0.3j
        }
b1 =  5.673661434063979
b2 =  13.275894852492831
eps['xx'] =  (0.2343399384821517-0.6667814477374165j)
eps['xy'] =  (-0.026910792178006807-0.008234472083130103j)
eps['xz'] =  (0.7629413525425355+0.17825797926794548j)
eps['yz'] =  (-0.12396855843912173+0.4577164447108008j)



def main():
    Angles, Fields, params = GetParams()
    

   
    Fields, Angles, P_abs = loadData()

    prepareObjects(Fields, Angles, params[:10])

    ResS12Pos = ResS12(Fields, Angles, P_abs, sign='pos')

    mygraph = Graph()
    mygraph.add_scatter(ResS12Pos.keys(), ResS12Pos.values())

    epsparams = b1, b2, eps
    ResS12PosFit = ResS21Fit(Fields, Angles, epsparams, sign='pos')
    mygraph.add_plot(ResS12PosFit.keys(), ResS12PosFit.values())

    mygraph.plot_Graph()
    


    

    
def ResS12(Fields, Angles, P_abs, sign='pos'):
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

    return angle_means

def ResS21Fit(Fields, Angles, params, sign='pos'):
    b1, b2, eps = params
    P_absFittet = {}
    scaled_P_abs = {}

    for i, field in enumerate(Fields):
        angle = Angles[i]
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

    return angle_means


def prepareObjects(Fields, Angles, params_):
    for i, field in enumerate(Fields):
        angle = Angles[i]
        mySWdirectory[field, angle] = SWcalculatorSingle(field, angle, params_)
        mySWdirectory[field, angle].calcPhi0()
        mySWdirectory[field, angle].calcChi()


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


    # Apply MinMaxScaling
    # scaled_P_abs = (P_abs - np.min(P_abs)) / (np.max(P_abs) - np.min(P_abs))

    return Fields, Angles, P_abs



if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()