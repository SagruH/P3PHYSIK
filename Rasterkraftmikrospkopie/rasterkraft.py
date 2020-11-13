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

aufgabe3()
