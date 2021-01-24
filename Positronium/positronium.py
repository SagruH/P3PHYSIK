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
        y = y + amp/(sig*np.sqrt(np.pi/2)) * np.exp( (-2)*((x - x0)/sig)**2)
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

    """
    gbounds = guess[:]
    jkl = 0
    for i in range(0, len(gbounds), 3):
        gbounds[i] = [(40+50*jkl)-20, (40+50*jkl)+20]
        gbounds[i+1] = [180, 250]
        gbounds[i+2] = [0, 100]
        jkl +=1
    gbounds = np.transpose(gbounds)
    """

    popt, pcov = curve_fit(func, x, y, p0=guess, maxfev=50000)

    fit = func(x, *popt)
    perr = np.sqrt(np.diag(pcov))

    dt = np.arange(2,42,4)
    x0 = []
    A = []
    sig = []
    x0er = []
    Aer = []
    siger = []

    for i in range(0, len(popt), 3):
        #print("t %2i : x0: %3.5f +- %2.3f , A: %5.5f +- %5.3f , sig: %3.5f +- %2.3f" % (dt[int(i/3)],popt[i],perr[i],popt[i+1],perr[i+1],popt[i+2],perr[i+2]))
        x0 += [popt[i]]
        A += [popt[i+1]]
        sig += [popt[i+2]]
        x0er += [perr[i]]
        Aer += [perr[i+1]]
        siger += [perr[i+2]]


    m, c, r_v, p_v, std_err = stats.linregress(x0,dt)
    xt = np.linspace(0,500,250)
    #print("linfit m:  ", m, " +- ", std_err," *x + ", c)

    fwhm = 2.355 * np.array([sig])
    delt = np.mean(fwhm*m)
    delter = np.std(fwhm*m)
    #print(delt,delter)

    plt.plot(x, y, "b.",label="Data")
    plt.plot(x, fit, 'r-',label="Gaussfit")
    plt.xlabel("Channel")
    plt.ylabel("t in ns")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()

    plt.plot(x0, dt, "b.")
    plt.plot(xt, m*xt+c, "r-",label="linear fit")
    plt.xlabel("Channel")
    plt.ylabel("t in ns")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()
    return delt;

def aufgabe3(delt):
    data = np.loadtxt("data/A3_Lebensdauer.RPT", unpack = True)
    return;

def main():
    #aufgabe1()
    delt = aufgabe2()
    aufgabe3(delt)
    return;

main()
