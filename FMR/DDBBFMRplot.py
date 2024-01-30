import os
from my_modules import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import platform

# if platform.system() == 'Linux': input_folder = '/home/julian/BA/dataForPython/FMR#3'
# elif platform.system() == 'Windows': input_folder = r'C:\Users\Julian\Seafile\BAJulian\dataForPython\FMR#3'

input_folder = '../dataForPython/FMR#3'


def main():

    Fields, Frequencies, S21 = loadData()

    Fields, Frequencies, S21 = cutOneAngle(Fields, Frequencies, S21)

    X, Y, Z = CreateMatrix(Fields, Frequencies, S21)

    Z = ScaleMatrix(Z)



    mygraph = Graph(width_cm=12.2)
    mygraph.add_cmPlot(Z, X, Y, cbarlabel='Re(d$S_{21}/$d$H$)\u2009(arb. u.)', cmap='seismic', equalBounds=True)
    mygraph.plot_Graph(save=True, name='FMR', xlabel='$\mu_0H_0$\u2009(mT)', ylabel='$f$\u2009(GHz)', title='b)')

    # cmPlot(Z, Fields, Frequencies, name='FMR', save=False, ylabel='$f$\u2009(GHz)', cbarlabel='d$S_{21}/$d$H$\u2009(dB)', cmap='seismic', equalBounds=True)




def loadData():
    filename = 'DerivateDevideM05.dat'

    importfilepath = os.path.join(input_folder, filename)

    column_names = ['Field', 'Frequency', 'S21']

    df = pd.read_csv(importfilepath, delimiter='\t', skiprows=4, names=column_names)

    # Extract columns as separate arrays
    Fields = df['Field'].to_numpy()
    Frequencies = df['Frequency'].to_numpy()
    S21 = df['S21'].to_numpy()

    return Fields, Frequencies, S21


def cutOneAngle(Fields, Frequencies, S21):
    LengthOneAngle = int(len(Fields)/9)
    # #update for first angle (160°)
    # Fields = Fields[:LengthOneAngle]*-1000
    # Frequencies = Frequencies[:LengthOneAngle]*1e-9
    #update for last angle (180°)
    Fields = Fields[-LengthOneAngle:]*-1000
    Frequencies = Frequencies[-LengthOneAngle:]*1e-9
    S21 = S21[:LengthOneAngle]

    return Fields, Frequencies, S21


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

def ScaleMatrix(Z):
    Z = np.matrix(Z)
    Z = Z/np.max(abs(Z))
    return Z





if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()