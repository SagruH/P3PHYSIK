import numpy as np
import PhyPraKit as ppk #wurde von mir verändert
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy.interpolate import make_interp_spline, BSpline

#import kafe
#from kafe.function_tools import FitFunction, LaTeX, ASCII
#from kafe.function_library import quadratic_3par

import uncertainties as uc
from uncertainties.umath import sqrt


def aufgabe3():
    #Berechung der Federkonstante aus gegbene Daten
    E = 1.69 * 10**11 #N/m^2 (vom Blatt)
    l = uc.ufloat(350,5) * 10**(-6) #m
    w = uc.ufloat(35,3) * 10**(-6)  #m
    t = uc.ufloat(1,0.3) * 10**(-6) #m

    c = (E*w*t**3)/(4*l**3)

    print("l: ", l)
    print("w: ", w)
    print("t: ", t)
    print("c: ", c)
    return;

def aufgabe4():
    #delta z höhenunterschied in nm
    Dz = np.array([228, 219, 222, 216, 225])
    #Abstände in x Richtung in mym
    Ax = np.array([5.5, 4.86, 5.9, 5.5, 5.08])
    #Periodizizät in x Richtung in mym
    Px = np.array([10.8, 11.9, 11.1, 10.7, 11.5])
    #Abstände in y Richtung in mym
    Ay = np.array([4.27, 4.68, 4.63, 5.52, 5.47])
    #Periodizizät in y Richtung in mym
    Py = np.array([9.82, 9.39, 9.03, 9.55, 9.54])
    #Unterschied Rand x Richtung in nm
    URx = np.array([76.2, 55.2, 43.8, 58.4, 55.5])
    #Unterschied Rand in y Richtung in nm
    Ury = np.array([2.9, 2.92, 5.84, 8.76, 14.6])

    Dzm = np.mean(Dz)
    Axm = np.mean(Ax)
    Aym = np.mean(Ay)

    xf = 5/Axm
    yf = 5/Aym
    zf = 200/Dzm
    print("x Wert = ", Axm, xf)
    print("y Wert = ", Aym, yf)
    print("z Wert = ", Dzm, zf)
    return;

def aufgabe7(): #CD
    #Tiefe in nm
    t = np.array([172, 175, 134, 161, 152])
    #Breite in nm
    b = np.array([847, 888, 839, 806, 808])
    #Spurabstand (Mittelpunkt zu MP) in mym
    a = np.array([1.76, 1.76, 1.72, 1.79, 1.79])

    tm = np.mean(t)
    bm = np.mean(b)
    am = np.mean(a)


    print(tm, bm, am)
    return;

aufgabe7()
