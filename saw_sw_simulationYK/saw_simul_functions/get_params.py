import numpy as np

# Constants
hbar = 1.05457173e-34
mue0 = 4 * np.pi * 1e-7
mueB = 9.27400968e-24


def GetParams(name='Rayleigh', scale=2):
    # Sweep Definition
    Angles = np.linspace(-90, 90, num=91*scale)



    # Magn Material Parameters
    

    AniType = 0     # 0 = Uniaxial; 1 = cubic; 2 = threefold; 3 = sixfold

    g = 2.026899502405165
    mue0Ms = 0.18381376682738004
    alpha = 0.0005134569097331153

    A = 3.7e-12

    # b1 = 4.37
    # b2 = 8.75
    b1 = 3.5
    b2 = 7

    # Structure Properties
    t = 1.012e-07

    k = 6.73e6


    if name == 'Rayleigh':
        Fields = np.linspace(-45, 45, num=165*scale) 
        # Simulated SAW
        f = 2.45e9
        # SAW Material Properties
        eps = {
            'xx': 0.39-0.69j,
            'yy': 0,
            'zz': 0.53-0.92j,
            'xy': 0,
            'xz': 1.00+0.56j,
            'yz': -0.02-0.02j
        }
        phiu = -1.2073697709042888 
        mue0Hani = 0.0017973461959371983 
        t = 8.910486597840024e-08 
        A = 4.099995562207347e-12 
        mue0Ms = 0.16897047988638672 
        g = 2.0365796945417864 

    elif name =='Sezawa':
        Fields = np.linspace(-80, 80, num=230*scale) 
        # Simulated SAW
        f = 3.53e9
        # SAW Material Properties
        eps = {
            'xx': 0.25+0.02j,
            'yy': 0,
            'zz': -0.07-0.01j,
            'xy': 0.01+0j,
            'xz': 0.09-1.00j,
            'yz': 0.02+0.02j
        }
        phiu = -1.8190235993425725 
        mue0Hani = 0.00016433786171173227 
        t = 8.00158852667914e-08 
        A = 3.509754743290906e-12 
        mue0Ms = 0.18082609830144458 
        g = 2.0243306049220244 
    else: print('false Name !')


    # params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps['xx'], eps['yy'], eps['zz'], eps['xy'], eps['xz'], eps['yz']]
    params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, t, k, f, b1, b2, eps]

    return Angles, Fields, params