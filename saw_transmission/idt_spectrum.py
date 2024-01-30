import os
import glob
from my_modules import *
import numpy as np
import platform

if platform.system() == 'Linux': input_folder = '/home/julian/Seafile/BAJulian/dataForPython/Freq_Sweep#3'
elif platform.system() == 'Windows': input_folder=r'C:\Users\Julian\Seafile\BAJulian\dataForPython\Freq_Sweep#3'


def main():
    file_pattern = 'FreqSpace - *.txt'

    data_dict = GetDataforfiles(file_pattern)

    colors = get_plot_colors(len(data_dict))

    graph = Graph()

    # graph.add_vline(5, label='5')

    simulFreqArray = [1.08, 5.5, 2.90, 3.68, 1.87, 4.72]

    for i, (number, (frequency, magnitude)) in enumerate(data_dict.items()):
        frequencies, magnitudes = GenerateArrayOne(number, frequency, magnitude)
        resFreq = round(GetResFreq(frequencies, magnitudes)*1e-9, 2)
        color = colors[i]
        yheight = -150
        if resFreq == 1.23 or resFreq==4.49: yheight = -155
        graph.add_scatter(frequencies*1e-9, magnitudes, label=f'{resFreq} GHz peak', color=color)
        graph.add_vline(resFreq, y=yheight, color=color, label=f'{resFreq}\u2009GHz', linestyle='--')
        

    graph.plot_Graph(legend=False, save=False, name='IDT-spectrum', xlabel='$f$\u2009(GHz)', ylabel='$S_{21}$\u2009(dB)')
 






def GetResFreq(frequencies, magnitudes):
    max_magnitude_index = np.argmax(magnitudes)
    res_frequency = frequencies[max_magnitude_index]

    return res_frequency
    



def GenerateArray(data_dict):
    frequencies =[]
    magnitudes =[]
    for number, (frequency, magnitude) in data_dict.items():
        num_points = len(frequency)
        cut_points = int(num_points * 0.01)
        frequency_cut = frequency[cut_points:-cut_points]
        magnitude_cut = magnitude[cut_points:-cut_points]
        
        # Append the cut data
        frequencies.append(frequency_cut)
        magnitudes.append(magnitude_cut)

    frequencies = np.concatenate(frequencies)
    magnitudes = np.concatenate(magnitudes)

    return frequencies, magnitudes

def GenerateArrayOne(number, frequency, magnitude):
    cut_factor = 0.1  
    num_points = len(frequency)
    cut_points = int(num_points * cut_factor)
    frequency_cut = frequency[cut_points:-cut_points]
    magnitude_cut = magnitude[cut_points:-cut_points]

    return frequency_cut, magnitude_cut

    
def GetDataforfiles(file_pattern):
    data_dict = {}
    pattern = os.path.join(input_folder, file_pattern)
    file_list = glob.glob(pattern)
    file_list.sort()
    for file_path in file_list:
        number = extract_number_from_filename(file_path)
        data = np.loadtxt(file_path, dtype=float, skiprows=1)
        frequency = data[:, 0]
        magnitude = data[:, 1]
        data_dict[number] = (frequency, magnitude)

    return data_dict

def extract_number_from_filename(file_path):
    # Extract the numeric part from the filename
    base_name = os.path.basename(file_path)
    number_str = base_name.replace('FreqSpace - ', '').replace('.txt', '')
    return float(number_str)



def GetData(filename):
    input_filepath = os.path.join(input_filepath, filename)
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)
    frequency = data[:, 0]
    magnitude = data[:, 1]
    return frequency, magnitude




if __name__ == "__main__":
    main()