
import os
from my_modules import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile



# Define the path to your input TDMS file
input_folder = r'C:\Users\Julian\Documents\BA\S11#3'
# input_folder = '/home/julian/BA/dataForPython/S11#3'


# name = 'Rayleigh'
name = 'Sezawa'

def main():

    # Fields, Frequencies, S21 = loadData()

    S11 = loadDataDD()
    # S11 = loadDataTDMS()

    # S11_sub = extractFreq(S11, name=name)

    # x = list(S11_sub.keys())
    # y = list(S11_sub.values())

    # I_S11_dict = Integrate(S11_sub)

    # x = list(I_S11_dict.keys())
    # y = list(I_S11_dict.values())

    # GraphPlot(x, y, name='', ylabel='', save=False)


    X, Y, Z = CreateMatrix(S11)

    cmPlot(Z, X*1e3, Y*1e-9, name='S11', save=False, ylabel='$f$\u2009(GHz)', cbarlabel='d$S_{11}/$d$H$\u2009(dB)', cmap='seismic', equalBounds=True)



def loadDataTDMS():
    filename = '2023-DEC-19-SequenceS11.tdms' 
    file_path = os.path.join(input_folder, filename)
    tdms_file = TdmsFile.read(file_path)

    Fieldsbefore = tdms_file['Read.LSbefore']['Field (T)'].data
    Fieldsafter = tdms_file['Read.LSafter']['Field (T)'].data
    Fields = (Fieldsbefore + Fieldsafter) / 2 #*1e3
    Frequencies = tdms_file['Read.ZNA']['Frequency'].data
    Real = tdms_file['Read.ZNA']['S11_REAL'].data
    Imag = tdms_file['Read.ZNA']['S11_IMAG'].data
    S11 = Real ** 2 + Imag ** 2 

    repFields = np.repeat(Fields, (len(S11)/len(Fields)))

    S11 = {(field, freq): s21 for field, freq, s21 in zip(repFields, Frequencies, S11)}

    return S11
    
def Integrate(S11_sub):
    I_S11_dict = {}
    fields = [list(S11_sub.keys())[0]]
    S11 = [list(S11_sub.values())[0]]
    counter = 0
    for field, s11 in S11_sub.items():
        if counter == 0:
            counter += 1
            continue
        fields.append(field)
        S11.append(s11)
        I_S11 = np.trapz(fields, S11)
        I_S11_dict[field] = I_S11
    return I_S11_dict



def extractFreq(S11, name='Rayleigh'):
    if name == 'Rayleigh':
        frequency = 2.45e9
    elif name == 'Sezawa':
        frequency = 3.53e9

    # Find the nearest frequency to the given value
    nearest_frequency = min(S11.keys(), key=lambda f: abs(f[1] - frequency))

    # Extract fields and S11 for the nearest frequency
    fields = [field for field, freq in S11.keys() if freq == nearest_frequency[1]]
    s11_values = [s11 for (field, freq), s11 in S11.items() if freq == nearest_frequency[1]]

    # Create a dictionary with fields as keys and S11 values as values
    S11_sub = {field: s11 for field, s11 in zip(fields, s11_values)}

    # Return the dictionary S11_sub
    return S11_sub
    

def loadDataDD():
    filename = 'S11FreqSweep.dat'

    importfilepath = os.path.join(input_folder, filename)

    column_names = ['Field', 'Frequency', 'S21']

    df = pd.read_csv(importfilepath, delimiter='\t', skiprows=4, names=column_names)

    S11 = {(field, freq): s21 for field, freq, s21 in zip(df['Field'], df['Frequency'], df['S21'])}

    return S11


def CreateMatrix(S11):
    Fields, Frequencies = zip(*S11.keys())
    S21 = list(S11.values())

    unique_fields = np.unique(Fields)
    unique_frequencies = np.unique(Frequencies)

    # Create X and Y grids using numpy.meshgrid
    X, Y = np.meshgrid(unique_fields, unique_frequencies)

    # Initialize Z with None values to indicate empty fields
    Z = np.empty_like(X)

    # Create index maps for fields and frequencies
    field_index_map = {field: index for index, field in enumerate(unique_fields)}
    frequency_index_map = {freq: index for index, freq in enumerate(unique_frequencies)}

    # Fill the Z-values with corresponding S21 values using advanced indexing
    for (field, freq), s21 in S11.items():
        field_index = field_index_map.get(field)
        frequency_index = frequency_index_map.get(freq)
        if field_index is not None and frequency_index is not None:
            Z[frequency_index, field_index] = s21

    X = np.matrix(X)
    Y = np.matrix(Y)
    return X, Y, Z









if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()