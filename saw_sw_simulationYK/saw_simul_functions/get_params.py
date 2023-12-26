import numpy as np

def GetParams(name='Rayleigh'):
    # Sweep Definition
    Angles = np.linspace(-90, 90, num=91)

    # Constants
    hbar = 1.05457173e-34
    mue0 = 4 * np.pi * 1e-7
    mueB = 9.27400968e-24

    # Magn Material Parameters
    alpha = 0.008
    AniType = 0     # 0 = Uniaxial; 1 = cubic; 2 = threefold; 3 = sixfold
    mue0Hani = 0.0001204
    phiu = 0
    A = 3.7408e-12
    g = 2.0468
    mue0Ms = 0.179

    b1 = 4.37
    b2 = 8.75

    # Structure Properties
    t = 1.012e-07


    if name == 'Rayleigh':
        Fields = np.linspace(-50, 50, num=101) 
        # Simulated SAW
        k =  4670000
        f = 2.43e9
        # SAW Material Properties
        eps = {
            'xx': 0.39-0.69j,
            'yy': 0,
            'zz': 0,
            'xy': 0,
            'xz': 1.00+0.56j,
            'yz': -0.02-0.02j
        }
    elif name =='Sezawa':
        Fields = np.linspace(-100, 100, num=201) 
        # Simulated SAW
        k =  5096000
        f = 3.53e9
        # SAW Material Properties
        eps = {
            'xx': 0.25+0.02j,
            'yy': 0,
            'zz': -0.07-0.01j,
            'xy': 0.01+0.00j,
            'xz': 0.09-1.00j,
            'yz': 0.02+0.02j
        }
    else: print('false Name !')


    params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz']]

    return Angles, Fields, params