import numpy as np
from .minimize_g import *
from .magn_susceptibility import MagnSusceptibility, MagnSusceptibilityNEU
from .magn_elas_field import MagnElasField
from .abs_power import AbsPower



# Constants
# hbar = 1.05457173e-34
# mue0 = 4 * np.pi * 1e-7
# mueB = 9.27400968e-24

def calculate(Fields, Angles, params):
    Fields, Angles = Fields/1000, np.deg2rad(Angles)
    eps = {}
    alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, t, k, f, b1, b2, eps = params
    phiu = np.deg2rad(phiu)


    # Calculation
    # phi0 = np.zeros((len(Angles), len(Fields)))
    # phitest = np.zeros_like(phi0)
    # test = np.zeros_like(phi0)
    P_abs = np.zeros((len(Angles), len(Fields)))

    # Assuming Fields and Angles are numpy arrays
    for angle_ind, angle in enumerate(Angles):
        for field_ind, field in enumerate(Fields):
            
            # Step 1: find Magnetic equilibrium
            phi0 = MinimizeGJu([field, angle, AniType, phiu, mue0Hani])
            # phi0 = MinimizeGFast(field, angle)
        
            # print(np.rad2deg(phi0))

            # Step 2: derive magnetic susceptibility
            chi_inv = MagnSusceptibility([field, phi0, angle, phiu, mue0Hani, A, mue0Ms, alpha, g, k, f, t, AniType])
            chi = np.linalg.inv(chi_inv)
            # chi = MagnSusceptibilityNEU([field, phi0, angle, phiu, mue0Hani, A, mue0Ms, alpha, g, k, f, t, AniType])
            # test = np.imag(chi[0,0])

            # Step 3: derive magnetoelastic driving field
            h_dr = MagnElasField([phi0, b1, b2, eps])

            # Step 4: derive absorbed power
            P_abs[angle_ind, field_ind] = AbsPower(h_dr, chi)

    #normalize
    P_abs = MinMaxScaling(P_abs)

    return P_abs


def singleCalculate(args):
    field, angle, params = args
    eps = {}
    alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, b1, b2, t, k, f, eps = params
    phiu = np.deg2rad(phiu)
    # Step 1: find Magnetic equilibrium
    phi0 = MinimizeGJu([field, angle, AniType, phiu, mue0Hani])
    # phi0 = MinimizeGFast(field, angle)

    # Step 2: derive magnetic susceptibility
    chi_inv = MagnSusceptibility([field, phi0, angle, phiu, mue0Hani, A, mue0Ms, alpha, g, k, f, t, AniType])
    chi = np.linalg.inv(chi_inv)
    # test = np.imag(chi[0,0])

    # Step 3: derive magnetoelastic driving field
    h_dr = MagnElasField([phi0, b1, b2, eps])

    # Step 4: derive absorbed power
    P_abs = AbsPower(h_dr, chi)
    return P_abs

def MinMaxScaling(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix






class SWcalculator():
    def __init__(self, Fields, Angles, params_):
        Fields, Angles = Fields*1e-3, np.deg2rad(Angles)
        alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, t, k, f = params_
        self.Fields = Fields
        self.Angles = Angles
        self.alpha = alpha
        self.AniType = AniType
        self.mue0Hani = mue0Hani
        self.phiu = np.deg2rad(phiu)
        self.A = A
        self.g = g
        self.mue0Ms = mue0Ms
        self.t = t
        self.k = k
        self.f = f

        self.P_abs = np.zeros((len(Angles), len(Fields)))
        self.Phi0 = np.zeros_like(self.P_abs)
        self.Chi = np.empty((len(self.Angles), len(self.Fields)), dtype=object)
        self.H_dr = np.empty((len(self.Angles), len(self.Fields)), dtype=object)




    def calcPhi0(self):
        AniType = self.AniType
        mue0Hani = self.mue0Hani
        phiu = self.phiu

        for angle_ind, Angle in enumerate(self.Angles):
            for field_ind, Field in enumerate(self.Fields):
                self.Phi0[angle_ind, field_ind] = MinimizeGJu([Field, Angle, AniType, phiu, mue0Hani])
                # self.Phi0[angle_ind, field_ind] = MinimizeGFast(Field, Angle)
                

    def calcChi(self, alpha=None):
        Field = self.Fields
        Angle = self.Angles
        AniType = self.AniType
        mue0Hani = self.mue0Hani
        phiu = self.phiu
        if alpha is None: alpha = self.alpha
        A = self.A
        g = self.g
        mue0Ms = self.mue0Ms
        t = self.t
        k = self.k
        f = self.f

        for angle_ind, Angle in enumerate(self.Angles):
            for field_ind, Field in enumerate(self.Fields):
                phi0 = self.Phi0[angle_ind, field_ind]
                chi_inv = MagnSusceptibility([Field, phi0, Angle, phiu, mue0Hani, A, mue0Ms, alpha, g, k, f, t, AniType])
                self.Chi[angle_ind, field_ind] = np.linalg.inv(chi_inv)

    def calcH_dr(self, b1=None, b2=None, eps=None):
        for angle_ind, Angle in enumerate(self.Angles):
            for field_ind, Field in enumerate(self.Fields):
                phi0 = self.Phi0[angle_ind, field_ind]
                self.H_dr[angle_ind, field_ind] = MagnElasField([phi0, b1, b2, eps])

    def calcP_abs(self, scale=False):
        for angle_ind, Angle in enumerate(self.Angles):
            for field_ind, Field in enumerate(self.Fields):
                h_dr = self.H_dr[angle_ind, field_ind]
                chi = self.Chi[angle_ind, field_ind]
                self.P_abs[angle_ind, field_ind] = AbsPower(h_dr, chi)

        if scale: self.P_abs = MinMaxScaling(self.P_abs)

    
    



class SWcalculatorSingle:
    def __init__(self, Field, Angle, params_):
        Field, Angle = Field*1e-3, np.deg2rad(Angle)
        alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, t, k, f = params_
        self.Field = Field
        self.Angle = Angle
        self.alpha = alpha
        self.AniType = AniType
        self.mue0Hani = mue0Hani
        self.phiu = np.deg2rad(phiu)
        self.A = A
        self.g = g
        self.mue0Ms = mue0Ms
        self.t = t
        self.k = k
        self.f = f

    def calcPhi0(self):
        Field = self.Field
        Angle = self.Angle
        AniType = self.AniType
        mue0Hani = self.mue0Hani
        phiu = self.phiu

        self.Phi0 = MinimizeGJu([Field, Angle, AniType, phiu, mue0Hani])
    
    def calcChi(self, alpha=None):
        Field = self.Field
        Angle = self.Angle
        Phi0 = self.Phi0
        AniType = self.AniType
        mue0Hani = self.mue0Hani
        phiu = self.phiu
        if alpha is None:
            alpha = self.alpha
        A = self.A
        g = self.g
        mue0Ms = self.mue0Ms
        t = self.t
        k = self.k
        f = self.f

        chi_inv = MagnSusceptibility([Field, Phi0, Angle, phiu, mue0Hani, A, mue0Ms, alpha, g, k, f, t, AniType])
        self.chi = np.linalg.inv(chi_inv)


    
    def calcH_dr(self, b1=None, b2=None, eps=None):
        phi0 = self.Phi0
        self.h_dr = MagnElasField([phi0, b1, b2, eps])


    def calcP_abs(self):
        h_dr = self.h_dr
        chi = self.chi
        self.P_abs = AbsPower(h_dr, chi)
