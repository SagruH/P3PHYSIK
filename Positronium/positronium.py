import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy.optimize import curve_fit
from scipy import interpolate as ip

def aufgabe1():
    data_oR = np.loadtxt("data/A1_ohne_Rauschen.RPT", unpack = True)
    data_mR = np.loadtxt("data/A1_mit_Rauschen.RPT", unpack = True)

    plt.plot(data_oR[0],data_oR[1],"b.",label="Daten ohne Rauschen")
    plt.plot(data_mR[0],data_mR[1],"r.",label="Daten mit Rauschen")

    plt.xlabel("Channel")
    plt.ylabel("Ereignisse")
    plt.legend()
    plt.grid(True)
    plt.show()
    return;

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        x0 = params[i]
        amp = params[i+1]
        sig = params[i+2]
        y = y + amp/np.sqrt(2*np.pi*sig**2) * np.exp( (-2)*((x - x0)/sig)**2)
    return y;

def aufgabe2():     #zeitliche Auflösung
    data = np.loadtxt("data/A2_zeitliche_Aufloesung.RPT", unpack = True)

    plt.plot(data[0],data[1],"b.")

    plt.xlabel("Channel")
    plt.ylabel("Ereignisse")
    #plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()

    x = data[0]
    y = data[1]
    guess = []
    for i in range(10):
        guess += [40+50*i, 220, 25]

    gbounds = guess
    j = 0
    for i in range(0, len(guess), 3):
        gbounds[i] = [(40+50*j)-20, (40+50*j)+20]
        gbounds[i+1] = [180, 250]
        gbounds[i+2] = [0, 100]
        j +=1
    gbounds = np.transpose(gbounds)

    popt, pcov = curve_fit(func, x, y, p0=guess, maxfev=20000)
    print(popt)
    fit = func(x, *popt)

    plt.plot(x, y, "b.")
    plt.plot(x, fit, 'r-')
    plt.grid(True)
    plt.show()

    return;

def main():
    #aufgabe1()
    aufgabe2()
    return;

main()
