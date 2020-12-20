import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy import optimize

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

def plt_corr_data(I,Z):
    plt.plot(I,Z, "xr")
    plt.plot(I,Z, "-b")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Ereignisse (korrigiert)")
    plt.grid(True)
    plt.show()
    return;

def gausfitf(I, a1, m1, s1, a2, m2, s2):
    #m1 = 8.38
    #m2 = 8.688
    e1 = ((I-m1)**2)/(2*s1**2)
    e2 = ((I-m2)**2)/(2*s2**2)
    f = a1 * np.exp(-e1) + a2 * np.exp(-e2)
    return f;

def aufgabe2b(Z, I):
    popt, pcov = optimize.curve_fit(gausfitf, I, Z, [124.4,8.38,0.07,34,8.688,0.08])
    [a1,m1,s1,a2,m2,s2] = popt
    print(popt)
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
    plt.show()
    return;

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
    #plt_corr_data(I,Z) #decomment to use

    #curve fit koversion peaks
    aufgabe2b(Z[35:],A[35:])

    return;

main()
