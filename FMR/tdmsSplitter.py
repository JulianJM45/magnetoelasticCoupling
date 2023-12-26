from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
import numpy as np

# Define the path to your input TDMS file
input_tdms_file_path = r'C:\Users\Julian\Documents\BA\FMR#3\2023-OCT-23-FMR_Field_Angle_sweepPart1.tdms'
# Output directory
output_directory = r'C:\Users\Julian\Documents\BA\FMR#3\singleAngle'

#desired datapoints
desired_datapoints = 51

# Load the input TDMS file
input_tdms_file = TdmsFile.read(input_tdms_file_path)

# List of group names and their respective channel names gac = groups_and_channels
gac = [
    ('SetSweep.Caylar Rot', 'angle (°)'),
    ('SetSweep.Caylar PS', 'Field (T)'),
    ('Read.readbefore', 'Field (T)'),
    ('Read.Caylar PS', 'Frequency'),
    ('Read.Caylar PS', 'S41_REAL'),
    ('Read.Caylar PS', 'S41_IMAG'),
    ('Read.readafter', 'Field (T)')
]

angles = input_tdms_file['SetSweep.Caylar Rot']['angle (°)'].data
numangles = len(angles)
fields = np.unique(input_tdms_file['SetSweep.Caylar PS']['Field (T)'].data)
numfields = len(fields)
frequencies = np.unique(input_tdms_file['Read.Caylar PS']['Frequency'].data)
numfreq = len(frequencies)

jumper=int(numfreq/desired_datapoints)
jumper = 1


for indexangle, angle in enumerate(angles):
    if angle == 40:
        print(angle)
        startindexf = indexangle*numfields
        endindexf = (indexangle+1)*numfields-1
        fieldbefore = input_tdms_file['Read.readbefore']['Field (T)'].data[startindexf:endindexf]
        fieldafter = input_tdms_file['Read.readafter']['Field (T)'].data[startindexf:endindexf]
        fieldmean = (fieldbefore+fieldafter)/2
        
        startindexv = indexangle*numfields*numfreq
        endindexv = (indexangle+1)*numfields*numfreq-1
        #frequency = input_tdms_file['Read.Caylar PS']['Frequency'].data[startindexv:endindexv]
        frequency = frequencies
        freqmulti = np.tile(frequencies, numfields)
        real = input_tdms_file['Read.Caylar PS']['S41_REAL'].data[startindexv:endindexv]
        imag = input_tdms_file['Read.Caylar PS']['S41_IMAG'].data[startindexv:endindexv]

        # print(len(real), (numfields-1)*numfreq+numfreq-1)
        fieldm=[]
        freqs=[]
        ReS41=[]
        ImS41=[]
        
        # for i in enumerate(frequencies[::jumper]):
        #     fieldmulti.append(fieldmean)
        for j, field in enumerate(fieldmean):
            for i, freq in enumerate(frequencies[::jumper]):
                fieldm.append(field)
                freqs.append(freq)
                ReS41.append(real[j*numfreq+i*jumper])
                ImS41.append(imag[j*numfreq+i*jumper]) 


        

        # fieldmulti = []
        # for i in range(desired_datapoints):
        #     for field in fieldmean:
        #         fieldmulti.append(field)       

        '''
        #rearrangement + jumper
        freqs=[]
        ReS41=[]
        ImS41=[]
        # print('lenreal', len(real))
        # # print(len(fieldmean)*numfreq+len(frequencies[::jumper])*jumper)
        for i, freq in enumerate(frequencies[::jumper]):
            for j, field in enumerate(fieldmean):
                freqs.append(freq)
                ReS41.append(real[j*numfreq+i*jumper])
                ImS41.append(imag[j*numfreq+i*jumper])  

        # frequencies =  freqs
        # real = ReS41
        # imag = ImS41
        '''

        # Create a new TDMS writer for the current angle
        output_file_path = f'{output_directory}\\Angle_{angle}forPYBBFMR.tdms'
        with TdmsWriter(output_file_path) as writer:
            # Create RootObject and GroupObject
            root_object = RootObject()
            group_object = GroupObject("Read.Group_1")

            # Create channels and write data
            fieldbefore_channel = ChannelObject("Read.Group_1", "Field Before", fieldbefore)
            fieldafter_channel = ChannelObject("Read.Group_1", "Field After", fieldafter)
            frequency_channel = ChannelObject("Read.Group_1", "Frequency", freqmulti)
            real_channel = ChannelObject("Read.Group_1", "S41_REAL", real)
            imag_channel = ChannelObject("Read.Group_1", "S41_IMAG", imag)

            # Write the objects to the TDMS file
            writer.write_segment([root_object, group_object, fieldbefore_channel, fieldafter_channel, frequency_channel, real_channel, imag_channel])


# Close the input TDMS file
input_tdms_file.close()

