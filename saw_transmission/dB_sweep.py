import os

from my_modules import *


# Define Import folder
input_folder = '/home/julian/BA/dataForPython/dB_Sweep#3'
output_folder = '/home/julian/BA/pictures'



def main():
    name = 'Sezawa'
    Powers, setFields, Fields, S12 = loadData(name)
    unique_powers = np.unique(Powers)

    Fields, S12 = splitData(Powers, Fields, S12)

    means = getMean(unique_powers, S12)
    means_values = [means[power] for power in unique_powers]

    mins = getMins(unique_powers, S12)
    mins_values = [mins[power] for power in unique_powers]

    deltaS12 = getDelta12(unique_powers, means, mins)
    deltaS12_values = [deltaS12[power] for power in unique_powers]

    
    # GraphPlot(unique_powers, deltaS12_values, save=False, xlabel='P in dB', ylabel='$\Delta S12$ in dB', name=f'{name}_DeltaDB', outputfolder=output_folder)





    # unique_powers = np.unique(Powers)
    # unique_fields = np.unique(setFields)
    # unique_powers = unique_powers[:3]

    colors = get_plot_colors(len(unique_powers))
    mygraph = Graph()

    for i, power in enumerate(unique_powers): 
        color = colors[i]
        mygraph.add_plot(Fields[power], S12[power], color=color, label=f'{power} dB Power input')

    mygraph.plot_Graph(safe=True, legend=False, xlabel='$\mu_0 H$ in T', ylabel='$S12$ in dB', name=f'{name}_dBplot', outputfolder=output_folder)


def getDelta12(unique_powers, means, mins):
    deltaS12 = {}
    for power in unique_powers:
        deltaS12[power] = means[power] - mins[power]
    
    return deltaS12


def getMean(unique_powers, S12):
    means = {}
    for power in unique_powers:
        means[power] = np.mean(S12[power])
    
    return means


def getMins(unique_powers, S12):
    mins = {}
    for power in unique_powers:
        mins[power] = np.min(S12[power])
    
    return mins

def splitData(Powers, Fields, S12):
    fields = {}
    s12 = {}
    unique_powers = np.unique(Powers)
    for power in unique_powers:
        mask = Powers == power
        fields[power] = Fields[mask]
        s12[power] = S12[mask]

    return fields, s12
  









def loadData(name):
    if name == 'Rayleigh': filename = 'DBSweepRayleigh.txt'
    elif name == 'Sezawa': filename = 'DBSweepSezawa.txt'
    else: print('false name')
    filepath = os.path.join(input_folder, filename)
    # Read the data from the .txt file
    data = np.loadtxt(filepath, dtype=float, skiprows=1)

    #Extract columns
    setField_column = data[:, 1]
    dB_column = data[:, 0]
    Field_column = 1000 * (data[:, 2] + data[:, 3]) / 2
    S12_column = data[:, 5]
    real_column = data[:, 7]
    imag_column = data[:, 8]

    return dB_column, setField_column, Field_column, S12_column







if __name__ =='__main__':
    main()