import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const

import uncertainties as uc
from uncertainties.umath import sqrt, log

def aufgabe1_mag():
    data = np.loadtxt("daten.csv", delimiter = ",", skiprows = 1, unpack = True)
    I = data[0]
    B = data[1]

    slp, b, r_v, p_v, std_err = stats.linregress(I, B)
    xr = np.linspace(0,10, 350)

    print("m :",slp ," b: ", b, " r: " , r_v)
    plt.plot(I,B, "xb")
    plt.plot(xr, slp*xr+b, "-r")

    plt.xlabel("I in A")
    plt.ylabel("B in mT")
    plt.grid(True)
    plt.show()
    return;



def main():
    aufgabe1_mag()

    [A, c, mt] = np.loadtxt("gruppe115_daten/beta_spectrum.txt", unpack = True)
    mt = 90 #s

    #background radiation
    A_back = A[:7]
    A = A[7:]
    c_back = c[:7]
    c = c[7:]
    background_counts = np.sum(c_back)/len(c_back)

    # corecttions
    Z = (968/1000) * (c-background_counts) / A



    if 0:
        print("background counts average:  ", background_counts)
        plt.plot(A,Z, "xr")
        plt.plot(A,Z, "-b")
        plt.xlabel("Stromstärke in A")
        plt.ylabel("Gezählte Ereignisse")
        plt.grid(True)
        plt.show()

    return;

main()
