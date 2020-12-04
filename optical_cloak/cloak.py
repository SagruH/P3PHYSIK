import numpy as np
import PhyPraKit as ppk #wurde von mir verändert
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const

import uncertainties as uc
from uncertainties.umath import sqrt, log

def task2():
    data = np.loadtxt("task2_data.csv", delimiter = ",", unpack = False, skiprows = 1)
    nr = len(data)
    nc = len(data[0])
    nra = np.arange(nr)
    thickness = 0.015 #meter
    g = 0.523
    v = const.c / 1.4
    #create new arrays: z, P(mean and std), concentration
    z = np.array([ (data[i][0]*thickness) for i in nra ])
    fc = np.array([ data[i][1] for i in nra ])
    conc = np.array([ data[i][2] for i in nra])

    #uc version
    #P =  np.array([ uc.ufloat( np.mean( data[i][3:] ), np.std( data[i][3:] ) ) for i in nra ]) /fc
    #P0 = P[0]
    #ls = (-z[1:])/ np.array([ log(P[i]/P0) for i in np.arange(1,nc) ])


    #non uc version
    P = np.array([np.mean( data[i][3:]) for i in nra ]) / fc
    ls = (-z[1:])/np.log(P[1:]/P[0])
    lt = ls/(1-g)
    D = (1/3)*v*lt
    return D, conc[1:], z[1:];


def main():
    D, conc, z = task2()
    for i in np.arange(len(D)):
        print("Concentration: %.3f || z: %.3f || D: %.3f" % ( conc[i], z[i], D[i] ) )
        
    return;


main()
