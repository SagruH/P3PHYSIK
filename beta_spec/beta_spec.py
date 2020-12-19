import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const

import uncertainties as uc
from uncertainties.umath import sqrt, log

def aufgabe1_mag():
    data = np.loadtxt("daten.txt", delimiter = ",", skiprows = 1, unpack = True)

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
    print("background counts average:  ", background_counts)
    # corect for backgroudn radiation
    c -= background_counts




    plt.plot(A,c, "xr")
    plt.plot(A,c, "-b")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Gezählte Ereignisse")
    plt.grid(True)
    #plt.show()

    return;

main()
