# import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def Plot(Angles, Fields, P_abs, name=''):
    output_path=f'./ColorMap{name}.png'
    Field_column = Fields
    Angle_column = Angles

    im = plt.imshow(P_abs, extent=[min(Field_column),max(Field_column),min(Angle_column),max(Angle_column)], cmap='hot', interpolation='nearest', aspect='auto')

    #labels 
    fontsize = 14
    plt.minorticks_on()
    cbar = plt.colorbar(im)
    cbar.ax.set_title('$\Delta$$S_{21}$ (dB)', fontsize=fontsize)
    plt.xlabel('Field $\mu_0H$ (mT)', fontsize=fontsize)
    plt.ylabel('$θ_H$ (°)', fontsize=fontsize)
    # plt.title('Simulated $\Delta$$S_{12}$ '+f'for {name} mode', fontsize=fontsize)
    plt.title('Rayleigh Test', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)

    # plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    plt.clf()