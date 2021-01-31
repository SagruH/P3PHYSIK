import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy.optimize import curve_fit
from scipy import interpolate as ip

import uncertainties as uc


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
    popt2, pcov2 = curve_fit(linf, x0, dt, p0=[m,c])
    perr2 = np.sqrt(np.diag(pcov2))
    #print("fit  ",popt2,perr2)

    xt = np.linspace(0,500,250)

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
    return delt,m;

def funca3(x, A1,t1,A2,t2,y0):
    y1 = A1*np.exp(-(x/t1))
    y2 = A2*np.exp(-(x/t2))
    y = y0 + y1 + y2
    return y;

def linf(x,m,b):
    return m*x+b;

def aufgabe3(delt,m):
    data = np.loadtxt("data/A3_Lebensdauer.RPT", unpack = True)
    t = data[0]*m
    y = data[1]
    xp = np.linspace(0,42,1000)

    #data plots
    plt.plot(t, data[1], "b.",label="Data")
    #plt.errorbar(t,data[1],xerr=delt)
    plt.yscale("log")
    plt.xlabel("t is ns")
    plt.ylabel("Ereignisse")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()

    #-------------------------------------------------------
    #data shift
    iymax = np.argmax(y)
    t = t[iymax:]
    y = y[iymax:]
    t = t-t[0]

    #fit
    guess = [4000,0.6,500,4,0.2]
    popt, pcov = curve_fit(funca3, t, y, p0=guess)
    perr = np.sqrt(np.diag(pcov))

    opt = []
    for i in range(len(popt)):
        opt += [uc.ufloat(popt[i],perr[i])]

    #Verhältniss
    v = (opt[0]*opt[1])/(opt[2]*opt[3])
    print(v)

    plt.plot(t, y, "b.",label="Data")
    plt.plot(xp, funca3(xp, *popt), "r-",label="fit")
    #plt.errorbar(t,data[1],xerr=delt)
    plt.yscale("log")
    plt.xlabel("t is ns")
    plt.ylabel("Ereignisse")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()
    return;

def aufgabe4(delt,m):
    x0,y0 = np.loadtxt("data/0cm.RPT", unpack = True)
    x10,y10 = np.loadtxt("data/10cm.RPT", unpack = True)
    x20,y20 = np.loadtxt("data/20cm.RPT", unpack = True)
    x30,y30 = np.loadtxt("data/30cm.RPT", unpack = True)
    x40,y40 = np.loadtxt("data/40cm.RPT", unpack = True)
    x50,y50 = np.loadtxt("data/30cm.RPT", unpack = True)
    x60,y60 = np.loadtxt("data/40cm.RPT", unpack = True)

    data = [x0,y0,x10,y10,x20,y20,x30,y30,x40,y40,x50,y50,x60,y60]
    labell = ["Data 0cm","Data 10cm","Data 20cm","Data 30cm","Data 40cm","Data 50cm","Data 60cm"]

    for i in range(0,len(data),1):
        data[i] = data[i][:150]

    for (i,j) in zip(range(0,len(data),2),labell):
        plt.plot(data[i],data[i+1],".-", lw=0.5, label=j)

    plt.xlabel("Channel")
    plt.ylabel("Ereignisse")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()


    #find peaks
    peakpos = []
    tpeak = []
    abs = [0,0.1,0.2,0.3,0.4]
    for i in range(0,len(data),2):
        peak = np.argmax([data[i+1]])
        peakpos += [data[i+1][peak]]
        tpeak += [data[i][peak] * m]

    mf, c, r_v, p_v, std_err = stats.linregress(tpeak,abs)

    xw = np.linspace(2.5,5,300)

    c0 = uc.ufloat(mf,std_err) * 1e9
    print(c0)

    plt.plot(tpeak,abs,"b.", label="peaks für jeden abstand")
    plt.plot(xw,mf*xw+c,"r-", label="fit")
    plt.xlabel("t in ns")
    plt.ylabel("Abstand in m")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()

    return;

def main():
    #aufgabe1()
    delt,m = aufgabe2()
    #aufgabe3(delt,m)
    aufgabe4(delt,m)
    return;

main()
