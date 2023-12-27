
import os
import shutil



# Specify the root directory where your .dat files are located
input_folder = r'C:\Users\Julian\Documents\BA\FMR#3\singleAngle'
output_folder = r'C:\Users\Julian\Documents\dataForPython\FMR#3'


# Iterate through directories with names like Angle_X.0(FTF)
for angle_dir in os.listdir(input_folder):
    if angle_dir.startswith("Angle_") and os.path.isdir(os.path.join(input_folder, angle_dir)):
        # Construct the full path to the .dat file in the current directory
        file_path = os.path.join(input_folder, angle_dir, '1. FMR-Susceptibility Fit', 'Resonance Fit.dat')

        if os.path.isfile(file_path):
            # Construct the output path for the copied file
            output_path = os.path.join(output_folder, f'{angle_dir}_Resonance Fit.dat')

            # Copy the file to the output folder
            shutil.copyfile(file_path, output_path)