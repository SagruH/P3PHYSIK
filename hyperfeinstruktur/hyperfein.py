import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy.optimize import curve_fit
from scipy import interpolate as ip

import uncertainties as uc

def uc_lin_reg(x,xerr,y,yerr):
    #Tests and clean up
    if(len(x) != len(y)):
        print("ERROR: unequal length")
        return;
    if np.isscalar(xerr): xerr = np.zeros_like(x);
    if np.isscalar(yerr): yerr = np.zeros_like(y);

    #variable ini
    xmt = uc.ufloat(0,0)
    ymt = uc.ufloat(0,0)
    n = len(x)
    ucx = []
    ucy = []
    sp = 0
    sq = 0
    m = 0
    b = 0

    #reshape vars into uc variables and get means
    for i in range(n):
        ucx += [uc.ufloat(x[i],xerr[i])]
        ucy += [uc.ufloat(y[i],yerr[i])]
        xmt += uc.ufloat(x[i],xerr[i])
        ymt += uc.ufloat(y[i],yerr[i])

    xm = xmt/n
    ym = ymt/n

    #least square fit
    for i in range(n):
        sp += (ucx[i] - xm) * (ucy[i] - ym)
        sq += (ucx[i] - xm)**2
    m = sp/sq
    b = ym - m*xm
    return m,b;

def laser():
    data1 = np.loadtxt("115/laser_green1.txt",unpack=True)
    data2 = np.loadtxt("115/laser_green2.txt",unpack=True)
    data3 = np.loadtxt("115/laser_green3.txt",unpack=True)

    lam = uc.ufloat(650,1)*1e-9
    f = uc.ufloat(150,1)*1e-3
    d1 = np.array([840-475,1090-235,1236-79]) /155
    d2 = np.array([937-497,1141-321,1310-184]) /155
    d3 = 1/np.sqrt(2) * np.array([1156-568,1481-253,1600-155]) / 155
    dm = np.mean([d1,d2,d3],axis = 0)
    derr = np.std([d1,d2,d3],axis = 0)

    dm = dm**2
    derr = derr**2

    m,b = uc_lin_reg([1,2,3],0,dm,derr)
    d = (4*lam*f**2)/(m*1e-6) #convert mm**2 to m**2
    print("m = ", m , "b = ", b)
    print("d = ", d*1e3, "mm")



    #plt.plot(data1[0], data1[1], "b.-",label="Data1")
    #plt.plot(data2[0], data2[1], "r.-",label="Data2")
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")

    xw = np.linspace(0.5,3.5,1000)
    plt.plot([1,2,3],dm,"b.", label = "Data")
    plt.plot(xw,m.n*xw+b.n,"r-", label = ("Fit: ("+ str(m) + ") * x + (" + str(b) + ")") )

    plt.xlabel("Ordnung n")
    plt.ylabel("Durchmesser D² in mm²")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()
    return d;

def green(d):
    data1 = np.loadtxt("115/data_green1.txt",unpack=True)
    data2 = np.loadtxt("115/data_green2.txt",unpack=True)
    #data3 = np.loadtxt("115/data_green3.txt",unpack=True)
    #data4 = np.loadtxt("115/data_green4.txt",unpack=True)

    plt.plot(data1[0], data1[1], "b.-",label="Data1")
    plt.plot(data2[0], data2[1], "r.-",label="Data2")
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")
    #plt.plot(data4[0], data4[1], "y.-",label="Data4")
    plt.xlabel("D in mm")
    plt.ylabel("Intensität")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()

    return;



def uv(d):
    data0 = np.loadtxt("115/date_uv0.txt",unpack=True)
    data1 = np.loadtxt("115/date_uv1.txt",unpack=True)
    data2 = np.loadtxt("115/date_uv2.txt",unpack=True)
    data3 = np.loadtxt("115/date_uv3.txt",unpack=True)
    data4 = np.loadtxt("115/date_uv4.txt",unpack=True)
    data5 = np.loadtxt("115/date_uv5.txt",unpack=True)

    plt.plot(data0[0], data0[1], "m.-",label="Data0")
    plt.plot(data1[0], data1[1], "b.-",label="Data1")
    plt.plot(data2[0], data2[1], "r.-",label="Data2")
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")
    #plt.plot(data4[0], data4[1], "y.-",label="Data4")
    #plt.plot(data5[0], data5[1], "k.-",label="Data5")

    plt.xlabel("D in mm")
    plt.ylabel("Intensität")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()
    return;


def main():
    d = laser()
    #green()
    #uv()
    return;

main()
