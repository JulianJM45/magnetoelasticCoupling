import os
import numpy as np
from my_modules import *
import re



# Define the angle offset
alpha = 48

# # Specify the full path to the input and output folders
# foldername_input = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\FreqFieldsM06'
# foldername_output = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\FreqFieldsM06Colormaps'

input_folder = '../dataForPython/Field_Angle_Sweep#3/FreqFields'
output_folder = '../picturesForBA/FreqFieldsM06Colormaps'


#mode name
mode = 'Rayleigh'


if mode == 'Rayleigh': 
    # Define the frequency range
    min_freq =2.3*1e9
    max_freq = 2.65*1e9
elif mode == 'Sezawa':
    min_freq =3.3*1e9
    max_freq = 3.65*1e9

def main():
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    input_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    # print(input_files)

    # Process each data file
    for input_file in input_files:
        if not input_file.startswith(mode):
            continue
        
        match = re.search(r'\d+', input_file)
        if match is None:
            continue

        new_number = int(match.group()) - alpha
        
        if new_number != 0:
            continue

        Fields, Freqs, S21 =  loadData(input_file, new_number)

        averageS21 =  CalculateAverage(Fields, Freqs, S21)

        deltaS21 = CalculateDeltaS12(Fields, Freqs, S21, averageS21)


        X, Y, Z = CreateMatrix(Fields, Freqs, deltaS21)

        # cmPlot(Z, X, Y, ylabel='$f$\u2009(GHz)', save=False, show=True, outputfolder=output_folder, cmap='hot', name=f'CMfreq{mode}_{new_number}°', vmin=-5, vmax=0)
        mygraph = Graph(width_cm=8.2)   
        mygraph.add_cmPlot(Z, X, Y, cmap='hot', vmin=-5, vmax=0)

        f = np.linspace(min_freq, max_freq, 1000)
        Hres = h_res.Hres(f=f, phiH=new_number)
        mygraph.add_plot(Hres*1e3, f*1e-9, label='Resonance', color='blue', linewidth=1)
        mygraph.add_plot(-Hres*1e3, f*1e-9, label='Resonance', color='blue', linewidth=1)

        mygraph.plot_Graph(show=True, save=True, xlabel='$\mu_0H_0$\u2009(mT)', ylabel='$f$\u2009(GHz)', name=f'CMfreq{mode}_{new_number}°')







def loadData(input_file, new_number):
    filepath_input = os.path.join(input_folder, input_file)
    with open(filepath_input, 'r') as f:
        data = f.read().replace(',', '.')  # Replace commas with periods in the data string
    data_lines = data.split('\n')  # Split the data into lines
    data_rows = [line.split('\t') for line in data_lines if line]  # Split each line into values
    original_matrix = np.array(data_rows, dtype=float)  # Convert the values to a NumPy array
    # output_file_name = f'ColorMapFreq_{new_number}.png'
    # filepath_output = os.path.join(foldername_output, output_file_name)
    # Extract the Field column
    Field_column = (original_matrix[:, 0] + original_matrix[:, 1]) / 2
    non_zero_Field_values = Field_column[Field_column != 0]
    Number_Fields = len(np.unique(non_zero_Field_values))



     # Filter rows based on frequency
    mask = (original_matrix[:, 2].astype(float) >= min_freq) & (original_matrix[:, 2].astype(float) <= max_freq)
    filtered_matrix = original_matrix[mask, :]

    # Extract the relevant columns
    Freq_column = (10**-9) *filtered_matrix[:, 2]
    S12_column = filtered_matrix[:, 3]
    
    # Repeat the field values as needed
    unique_freqs = np.unique(Freq_column)
    num_repeats = len(unique_freqs)
    Field_values_repeated = np.repeat(Field_column[:Number_Fields], num_repeats)
    Field_column = -1000 * Field_values_repeated

    if new_number < -90:
        new_number=new_number+180
        Field_column = -Field_column

    return Field_column, Freq_column, S12_column




def CalculateAverage(Field_column, Freq_column, S12_column):
    # Calculate average S_12 of 3 min/max field elemnts for each freq
    kth = 3
    average_s12 = {}
    unique_freqs = np.unique(Freq_column)
    for freq in unique_freqs: 
        mask = Freq_column == freq
        smallest_fields = np.partition(Field_column[mask], kth-1)[:kth]
        mask2 = np.isin(Field_column[mask], smallest_fields)
        avg_s12_smallest = np.mean(S12_column[mask][mask2])
        largest_fields = np.partition(Field_column[mask], -kth)[-kth:]
        mask2 = np.isin(Field_column[mask], largest_fields)
        avg_s12_largest = np.mean(S12_column[mask][mask2])
        average_s12[freq] = (avg_s12_smallest+avg_s12_largest)/2

    return average_s12

def CalculateDeltaS12(Field_column, Freq_column, S12_column, average_s12):
    # Calculate the Delta S_12 column
    unique_freqs = np.unique(Freq_column)
    deltaS12_column = np.zeros_like(S12_column)
    for freq in unique_freqs:
        mask = Freq_column == freq
        delta_S12 = S12_column[mask] - average_s12[freq] 
        # Handle invalid or missing values (e.g., NaN)
        delta_S12[np.isnan(delta_S12)] = 0  # Replace NaN with 0 or another suitable value
        deltaS12_column[mask] = delta_S12
    
    return deltaS12_column


def CreateMatrix(Field_column, Freq_column, deltaS12_column):
    # Create X and Y grids using numpy.meshgrid
    unique_fields = np.unique(Field_column)
    unique_freqs = np.unique(Freq_column)
    X, Y = np.meshgrid(unique_fields, unique_freqs)
    # Initialize Z with None values to indicate empty fields
    Z = np.empty_like(X)

    # Create index maps for angles and fields
    freq_index_map = {freq: index for index, freq in enumerate(unique_freqs)}
    field_index_map = {field: index for index, field in enumerate(unique_fields)}

    # Fill the Z-values with corresponding deltaS12 values using advanced indexing
    for freq, field, deltaS12 in zip(Freq_column, Field_column, deltaS12_column):
        freq_index = freq_index_map.get(freq)
        field_index = field_index_map.get(field)
        if freq_index is not None and field_index is not None:
            Z[freq_index, field_index] = deltaS12

    return X, Y, Z





# # Define the frequency range
# min_freq =2.19* 10**9
# max_freq = 2.71* 10**9 

# # Create the output folder if it doesn't exist
# os.makedirs(foldername_output, exist_ok=True)

# # Get a list of data files in the input folder
# input_files = [f for f in os.listdir(foldername_input) if f.endswith('.txt')]

# # Process each data file
# for input_file in input_files:
#     # Construct the full file paths for input and output
#     filepath_input = os.path.join(foldername_input, input_file)
#     file_number = int(input_file.split('.')[0])
#     new_number = file_number - alpha
#     output_file_name = f'ColorMapFreq_{new_number}.png'
#     filepath_output = os.path.join(foldername_output, output_file_name)

#     # Read and process the data from the input file as a string
#     with open(filepath_input, 'r') as f:
#         data = f.read().replace(',', '.')  # Replace commas with periods in the data string
#     data_lines = data.split('\n')  # Split the data into lines
#     data_rows = [line.split('\t') for line in data_lines if line]  # Split each line into values
#     original_matrix = np.array(data_rows, dtype=float)  # Convert the values to a NumPy array
#     #original_matrix = np.loadtxt(filepath_input, dtype=float)

#     # Extract the Field column
#     Field_column = (original_matrix[:, 0] + original_matrix[:, 1]) / 2
#     non_zero_Field_values = Field_column[Field_column != 0]
#     Number_Fields = len(np.unique(non_zero_Field_values))

#      # Filter rows based on frequency
#     mask = (original_matrix[:, 2].astype(float) >= min_freq) & (original_matrix[:, 2].astype(float) <= max_freq)
#     filtered_matrix = original_matrix[mask, :]

#     # Extract the relevant columns
#     Freq_column = (10**-9) *filtered_matrix[:, 2]
#     S12_column = filtered_matrix[:, 3]
    
#     # Repeat the field values as needed
#     unique_freqs = np.unique(Freq_column)
#     num_repeats = len(unique_freqs)
#     Field_values_repeated = np.repeat(Field_column[:Number_Fields], num_repeats)
#     Field_column = -1000 * Field_values_repeated

    

    

    # Create final matrix
    #header = np.array(["Field in mT", "Frequency in GHz", "Delta S_12"])
    #final_matrix = np.column_stack((Field_column, Freq_column, deltaS12_column))

    

    

#    # Set the desired figure size (width_cm, height_cm) in centimeters
#     width_cm = 16  # Adjust width as needed in cm
#     height_cm = 9  # Adjust height as needed in cm
#     fontsize = 12

#     # Convert centimeters to inches
#     width_in = width_cm / 2.54
#     height_in = height_cm / 2.54

#     # Create the figure with the specified size
#     fig = plt.figure(figsize=(width_in, height_in))

#     # Define the colormap for the heatmap
#     cmap = 'hot'

#     vmin=np.min(Z)
#     vmax=np.max(Z)
#     print(vmin, vmax)
#     vmax=0
#     vmin=-5
#     im = plt.imshow(
#             Z,
#             cmap=cmap,
#             aspect='auto',
#             interpolation='nearest',
#             origin='lower',
#             extent=[X.min(), X.max(), Y.min(), Y.max()],
#             vmin=vmin,  # Set the minimum value for the color scale
#             vmax=vmax  # Set the maximum value for the color scale
#         )
#     plt.minorticks_on()

#     # Add a vertical colorbar on the right side of the plot
#     cbar = plt.colorbar(im)
#     # Add labels and a title
#     title='$\Delta$$S_{12}$'+f' for {mode} mode and angle {new_number}°'
#     cbar.ax.set_title('$\Delta$$S_{21}$ (dB)', fontsize=fontsize)
#     plt.xlabel('Field $\mu_0H$ (mT)', fontsize=fontsize)
#     plt.ylabel('Frequency (GHz)', fontsize=fontsize)
#     plt.title(title, fontsize=fontsize)
#     plt.xticks(fontsize=fontsize)  
#     plt.yticks(fontsize=fontsize)

#     # Save the plot as an image file (e.g., PNG)
#     plt.savefig(filepath_output, dpi=300, bbox_inches='tight')

#     # Show the final plot with all heatmaps
#     #plt.show()



if __name__ == '__main__':
    main()


