import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy import optimize

import uncertainties as uc
from uncertainties.umath import sqrt, log

from beta_helper import *

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


def aufgabe2b(Z, I):
    popt, pcov = optimize.curve_fit(gausfitf, I, Z, [124.4,8.38,0.07,34,8.688,0.08])
    [a1,m1,s1,a2,m2,s2] = popt
    xg = np.linspace(8.1,8.9,250)
    yg = gausfitf(xg,a1,m1,s1,a2,m2,s2)
    yg_est = gausfitf(xg,124.4,8.38,0.07,34,8.688,0.08)

    plt.plot(I,Z, "xb", label = "data")
    plt.plot(xg,yg, "-r", label = "fit")
    plt.plot(xg,yg_est, "-y", label = "guess")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Ereignisse (korrigiert)")
    plt.legend()
    plt.grid(True)
    #plt.show()
    #print("a1,m1,s1,a2,m2,s2: ")
    #print(popt)
    plt.clf()
    return m1,m2;


def main():
    #aufgabe1_mag() #decomment to use
    [A, c, mt] = np.loadtxt("gruppe115_daten/beta_spectrum.txt", unpack = True)
    mt = 90 #s

    #background radiation
    A_back = A[:7]
    A = A[7:]
    c_back = c[:7]
    c = c[7:]
    background_counts = uc.ufloat(np.mean(c_back), np.std(c_back))
    #print("background counts average:  ", background_counts)

    # correcttions
    Z = (968/1000) * (c-background_counts.n) / A
    #plt_corr_data(A,Z) #decomment to use

    #curve fit koversion peaks
    m1, m2 = aufgabe2b(Z[35:],A[35:])

    #calc etas and fit
    EK = 624.6e3 *(1.6e-19) #eV
    EL = 656.4e3 *(1.6e-19) #eV
    eta_K = etaf(EK)
    eta_L = etaf(EL)
    #print(eta_K, eta_L)
    eta_fit=[0,eta_K,eta_L]
    I_fit =[0,m1,m2]
    m, t, r_v, p_v, std_err = stats.linregress(I_fit, eta_fit)
    plt_eichung_x(I_fit, eta_fit)


    return;

main()
