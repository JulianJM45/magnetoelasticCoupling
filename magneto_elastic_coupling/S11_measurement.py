import os
from my_modules import *
from nptdms import TdmsFile
import numpy as np
from scipy.optimize import curve_fit


# Define the path to your input TDMS file
# input_folder = r'C:\Users\Julian\Documents\BA\S11#3'
input_folder = '/home/julian/BA/dataForPython/S11#3'

name = 'Rayleigh'


def main():
    mygraph = Graph()
    # Fields, S11 = GetData(name)

    meanS11 = GetData()

    mygraph.add_scatter(meanS11.keys(), meanS11.values(), label='Messdaten')

    calcPeaks(meanS11)

    # fields, subS11_fit = FitLorenz(meanS11)

    # mygraph.add_plot(fields, subS11_fit, label='fit')

    mygraph.plot_Graph(save=True, legend=False, xlabel='$\mu_0$H in mT', ylabel='S11 in dB', name=f'S11_{name}')



    



def GetData():
    if name == 'Rayleigh': filename = '2023-DEC-20-SequenceS11CW2.43.tdms'
    elif name == 'Sezawa': filename = '2023-DEC-20-SequenceS11CW3.53.tdms'
    else: print('no file to this name')   
    file_path = os.path.join(input_folder, filename)
    tdms_file = TdmsFile.read(file_path)

    Fieldsbefore = tdms_file['Read.LSbefore']['Field (T)'].data
    Fieldsafter = tdms_file['Read.LSafter']['Field (T)'].data
    Fields = (Fieldsbefore + Fieldsafter) / 2 *1e3
    Real = tdms_file['Read.ZNA']['S11_REAL'].data
    Imag = tdms_file['Read.ZNA']['S11_IMAG'].data
    S11 = Real ** 2 + Imag ** 2 

    # repFields = np.repeat(Fields, (len(S11)/len(Fields)))
    LEN = int(len(S11)/len(Fields))

    meanS11 = {}

    for i, field in enumerate(Fields):
        meanS11[field] = np.mean(S11[i*LEN:(i+1)*LEN-1])

    # return repFields, S11
    return meanS11



def FitLorenz(meanS11):
    if name == 'Rayleigh': x1 = -35
    elif name == 'Sezawa': x1 = -65
    else: print('no file to this name') 

    x0 = closest_key(meanS11, x1)

    subFields, subS11 = getSubarray(meanS11, x0)

    params, covariance = curve_fit(CauchyPDF, subFields, subS11, maxfev=50000)

    x0_fit, gamma_fit, y_fit, m_fit = params

    fields = np.linspace(np.min(subFields), np.max(subFields), 100)

    subS11_fit = CauchyPDF(fields, x0_fit, gamma_fit, y_fit, m_fit)

    return fields, subS11_fit


def getSubarray(dictionary, key):
    keys = list(dictionary.keys())
    key_index = keys.index(key) 

    neighbors = 5

    neighboring_keys = keys[key_index-neighbors:key_index+neighbors]
    neighboring_values = [dictionary[k] for k in neighboring_keys]

    return neighboring_keys, neighboring_values



def closest_key(dictionary, target_value):
    closest_key = min(dictionary, key=lambda key: abs(key - target_value))
    return closest_key

def CauchyPDF(x, x0, gamma, y, m):
    return -1 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2)) + y + m*x



def calcPeaks(meanS11):
    if name == 'Rayleigh': x1 = -35
    elif name == 'Sezawa': x1 = -65
    else: print('no file to this name') 

    x0 = closest_key(meanS11, x1)

    subFields, subS11 = getSubarray(meanS11, x0)

    deltaS11 = np.mean(subS11) - np.min(subS11)

    print(deltaS11)




if __name__ == '__main__':
    main()


