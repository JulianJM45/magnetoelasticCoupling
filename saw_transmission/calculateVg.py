import os
from my_modules import *
import numpy as np
import platform
import glob

if platform.system() == 'Linux': input_folder = '/home/julian/Seafile/BAJulian/dataForPython/Freq_Sweep#3'
elif platform.system() == 'Windows': input_folder=r'C:\Users\Julian\Seafile\BAJulian\dataForPython\Freq_Sweep#3'





def main():
    data_dict = GetDataforfiles()

    # PlotSpectrum(data_dict, save=False)

    CalculateVg(data_dict)
    






def CalculateVg(data_dict):
    x = 500e-6
    x_err = x*0.1
    f_array = []
    vg_array = []
    vg_err_array = []

    for number, (time, amplitude) in data_dict.items():
        # Find the index of the maximum amplitude
        t_max = time[np.argmax(time)]
        t_min = time[np.argmin(time)]
        # print(f'for {number} t_max: {t_max}')
        # print(f'for {number} t_min: {t_min}')

        t_mean = (t_max + t_min) / 2
        t_err = (t_max - t_min) / 2

        vg = x / t_mean
        vg_err = np.sqrt((x_err / t_mean)**2 + (t_err * x / t_mean**2)**2)

        f_array.append(number)
        vg_array.append(vg)
        vg_err_array.append(vg_err)

        print(f'for {number} vg: {int(vg)}')

    print(f'f_array = {f_array}')
    print(f'vg_array = {vg_array}')
    print(f'vg_array_err = {vg_err_array}')


def PlotSpectrum(data_dict, save=False):
    graph = Graph()
    colors = get_plot_colors(len(data_dict), alpha=0.7)
    for i, (number, (time, amplitude)) in enumerate(data_dict.items()):
        color = colors[i]
        graph.add_plot(time, amplitude, color=color, label=f'{number}')   

    graph.plot_Graph(legend=True, save=save, name='IDT-spectrum', xlabel='$t$\u2009(s)', ylabel='$S_{21}$\u2009(dB)')


def GetDataforfiles():
    file_pattern = 'TimeSpace(s) - *.txt'
    # file_pattern = 'FreqSpace - *.txt'
    data_dict = {}
    pattern = os.path.join(input_folder, file_pattern)
    file_list = glob.glob(pattern)
    file_list.sort()
    for file_path in file_list:
        number = extract_number_from_filename(file_path)
        data = np.loadtxt(file_path, dtype=float, skiprows=1)
        time = data[:, 0]
        amplitude = data[:, 1]
        # Filter rows where amplitude is not 0
        non_zero_rows = amplitude != 0
        time = time[non_zero_rows]
        amplitude = amplitude[non_zero_rows]
        data_dict[number] = (time, amplitude)

    return data_dict


def extract_number_from_filename(file_path):
    # Extract the numeric part from the filename
    base_name = os.path.basename(file_path)
    number_str = base_name.replace('TimeSpace(s) - ', '').replace('.txt', '')
    return float(number_str)




if __name__ == '__main__':
    main()