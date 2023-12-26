import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
from nptdms import TdmsFile
import numpy as np


# Define the path to your input TDMS file
input_folder = r'C:\Users\Julian\Documents\BA\S11#3'




def main():
    name = 'Sezawa'
    Fields, S11 = GetData(name)


    GraphPlot(Fields*1e3, S11, xlabel='$\mu_0$H in mT', ylabel='S11 in dB', s=0.1, title=f'S11 in CW {name} mode')


    



def GetData(name):
    if name == 'Rayleigh': filename = '2023-DEC-20-SequenceS11CW2.43.tdms'
    elif name == 'Sezawa': filename = '2023-DEC-20-SequenceS11CW3.53.tdms'
    else: print('no file to this name')   
    file_path = os.path.join(input_folder, filename)
    tdms_file = TdmsFile.read(file_path)

    Fieldsbefore = tdms_file['Read.LSbefore']['Field (T)'].data
    Fieldsafter = tdms_file['Read.LSafter']['Field (T)'].data
    Fields = (Fieldsbefore + Fieldsafter) / 2
    Real = tdms_file['Read.ZNA']['S11_REAL'].data
    Imag = tdms_file['Read.ZNA']['S11_IMAG'].data
    S11 = Real ** 2 + Imag ** 2 

    repFields = np.repeat(Fields, (len(S11)/len(Fields)))

    return repFields, S11







if __name__ == '__main__':
    main()


