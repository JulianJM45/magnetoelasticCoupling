import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
import numpy as np

# importfolder = r'C:\Users\Julian\Documents\BA\Freq_Sweep#3'
importfolder = '/home/julian/BA/dataForPython/Freq_Sweep#3'


def main():
    file_pattern = 'FreqSpace - *.txt'

    data_dict = GetDataforfiles(file_pattern)

    colors = get_plot_colors(len(data_dict))

    graph = Graph()

    for i, (number, (frequency, magnitude)) in enumerate(data_dict.items()):
        frequencies, magnitudes = GenerateArrayOne(number, frequency, magnitude)
        resFreq = round(GetResFreq(frequencies, magnitudes)*1e-9, 2)
        color = colors[i]
        graph.add_scatter(frequencies*1e-9, magnitudes, label=f'{resFreq} GHz peak', color=color)

    graph.plot_Graph(legend=True, safe=True, name='IDT-spectrum', xlabel='$f$ in GHz', ylabel='$S21$ in dB', outputfolder='/home/julian/BA/pictures')






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
    pattern = os.path.join(importfolder, file_pattern)
    file_list = glob.glob(pattern)
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
    input_filepath = os.path.join(importfolder, filename)
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)
    frequency = data[:, 0]
    magnitude = data[:, 1]
    return frequency, magnitude


def get_plot_colors(num_colors):
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    return colors



if __name__ == "__main__":
    main()