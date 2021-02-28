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

    #Data Print ---------------------------
    #print("m = ", m , "b = ", b)
    #print("d = ", d*1e3, "mm")

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
    data3 = np.loadtxt("115/data_green3.txt",unpack=True)
    data4 = np.loadtxt("115/data_green4.txt",unpack=True)

    f   = uc.ufloat(150,1)*1e-3

    d1a = np.array([1055-580,1231-393,1340-266]) /155
    d1i = np.array([996-652,1194-428,1312-295]) /155
    d2a = np.array([1077-586,1265-405,1384-292]) /155
    d2i = np.array([1000-665,1223-442,1353-312]) /155
    d4a = np.array([1060-675,1261-479,1394-361]) /155
    d4i = np.array([950-763,1230-514,1368-383]) /155

    dma = np.mean([d1a,d2a,d4a],axis = 0)**2
    dmi = np.mean([d1i,d2i,d4i],axis = 0)**2
    derra = np.std([d1a,d2a,d4a],axis = 0)**2
    derri = np.std([d1i,d2i,d4i],axis = 0)**2

    ma,ba = uc_lin_reg([1,2,3],0,dma,derra)
    mi,bi = uc_lin_reg([1,2,3],0,dmi,derri)

    lama = (1e-6*ma*d)/(4*f**2)
    lami = (1e-6*mi*d)/(4*f**2)

    lamq = (lama + lami)/2

    Dlam = np.mean(lamq/(8*f**2) * ((dma-dmi)*1e-6))
    Dk = Dlam/lamq


    #Data Print ---------------------------
    print("Außen in mm²: m = ", ma , "b = ", ba)
    print("Innen in mm²: m = ", mi , "b = ", bi)
    print("lambda innen = ", lama*1e9, "nm")
    print("lambda außen = ", lami*1e9, "nm")
    print(Dlam,Dk*lamq**2)

    #plt.plot(data1[0], data1[1], "b.-",label="Data1")
    #plt.plot(data2[0], data2[1], "r.-",label="Data2")
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")
    #plt.plot(data4[0], data4[1], "y.-",label="Data4")

    xw = np.linspace(0.5,3.5,1000)
    plt.plot([1,2,3],dma,"bo", label = "Data Außen")
    plt.plot([1,2,3],dmi,"co", label = "Data Innen")
    plt.plot(xw,ma.n*xw+ba.n,"r-", label = ("Fit Außen: ("+ str(ma) + ") * x + (" + str(ba) + ")") )
    plt.plot(xw,mi.n*xw+bi.n,"m-", label = ("Fit Innen: ("+ str(mi) + ") * x + (" + str(bi) + ")") )

    plt.xlabel("Ordnung n")
    plt.ylabel("Durchmesser D² in mm²")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()
    return;


def uv_old(d):
    data0 = np.loadtxt("115/date_uv0.txt",unpack=True)
    data1 = np.loadtxt("115/date_uv1.txt",unpack=True)
    data2 = np.loadtxt("115/date_uv2.txt",unpack=True)
    data3 = np.loadtxt("115/date_uv3.txt",unpack=True)
    data4 = np.loadtxt("115/date_uv4.txt",unpack=True)
    data5 = np.loadtxt("115/date_uv5.txt",unpack=True)

    f   = uc.ufloat(150,1)*1e-3

    d1a = np.array([1048-342,1110-244]) /155
    d1c = np.array([1000-400,1091-281]) /155
    d1i = np.array([967-432,1087-308])  /155

    d4a = np.array([]) /155
    d4c = np.array([]) /155
    d4i = np.array([]) /155

    d5a = np.array([1095-349,1222-251]) /155
    d5c = np.array([1044-394,1177-289]) /155
    d5i = np.array([995-426,1138-314])  /155

    dma = np.mean([d1a,d5a],axis = 0) **2
    dmc = np.mean([d1c,d5c],axis = 0) **2
    dmi = np.mean([d1i,d5i],axis = 0) **2
    derra = np.std([d1a,d5a],axis = 0) **2
    derrc = np.std([d1c,d5c],axis = 0) **2
    derri = np.std([d1i,d5i],axis = 0) **2

    ma,ba = uc_lin_reg([1,2],0,dma,derra)
    mc,bc = uc_lin_reg([1,2],0,dmc,derrc)
    mi,bi = uc_lin_reg([1,2],0,dmi,derri)

    lama = (1e-6*ma*d)/(4*f**2)
    lamc = (1e-6*mc*d)/(4*f**2)
    lami = (1e-6*mi*d)/(4*f**2)

    lamq = (lama+ lamc+ lami)/3

    Dlamac = np.mean(lamq/(8*f**2) * ((dma-dmc)*1e-6))
    Dlamai = np.mean(lamq/(8*f**2) * ((dma-dmi)*1e-6))
    Dlamci = np.mean(lamq/(8*f**2) * ((dmc-dmi)*1e-6))


    #Data Print ---------------------------
    print("Außen in mm²: m = ", ma , "b = ", ba)
    print("mitte in mm²: m = ", mc , "b = ", bc)
    print("Innen in mm²: m = ", mi , "b = ", bi)
    print("lambda außen = ", lama*1e9, "nm")
    print("lambda mitte = ", lamc*1e9, "nm")
    print("lambda innen = ", lami*1e9, "nm")
    print(Dlamac/lamq, Dlamai/lamq, Dlamci/lamq)

    #plt.plot(data0[0], data0[1], "m.-",label="Data0")   #nope
    #plt.plot(data1[0], data1[1], "b.-",label="Data1")   #good
    #plt.plot(data2[0], data2[1], "r.-",label="Data2")   #maybe (nope)
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")   #nope
    #plt.plot(data4[0], data4[1], "y.-",label="Data4")   #good
    #plt.plot(data5[0], data5[1], "k.-",label="Data5")   #good

    xw = np.linspace(0.5,3.5,1000)
    plt.plot([1,2],dma,"bo", label = "Data Außen")
    plt.plot([1,2],dmc,"ko", label = "Data Mitte")
    plt.plot([1,2],dmi,"co", label = "Data Innen")
    plt.plot(xw,ma.n*xw+ba.n,"r-", label = ("Fit Außen: ("+ str(ma) + ") * x + (" + str(ba) + ")") )
    plt.plot(xw,mc.n*xw+bc.n,"y-", label = ("Fit Mitte: ("+ str(ma) + ") * x + (" + str(ba) + ")") )
    plt.plot(xw,mi.n*xw+bi.n,"m-", label = ("Fit Innen: ("+ str(mi) + ") * x + (" + str(bi) + ")") )

    plt.xlabel("Ordnung n")
    plt.ylabel("Durchmesser D² in mm²")
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

    f   = uc.ufloat(150,1)*1e-3

    d1a = 2*( np.array([93,161,244,342,1048,1110])-695    ) /155
    d1c = 2*( np.array([120,196,281,400,1000,1091])-695    ) /155
    d1i = 2*( np.array([139,214,308,432,967,1087] )-695    ) /155

    d5a = 2*(np.array([95,178,251,349,1095,1222])-722) /155
    d5c = 2*(np.array([117,207,289,394,1044,1177])-722) /155
    d5i = 2*(np.array([154,225,314,426,995,1138])-722)  /155

    dma = np.mean([d1a,d5a],axis = 0) **2
    dmc = np.mean([d1c,d5c],axis = 0) **2
    dmi = np.mean([d1i,d5i],axis = 0) **2
    derra = np.std([d1a,d5a],axis = 0) **2
    derrc = np.std([d1c,d5c],axis = 0) **2
    derri = np.std([d1i,d5i],axis = 0) **2

    ma,ba = uc_lin_reg([4,3,2,1,1,2],0,dma,derra)
    mc,bc = uc_lin_reg([4,3,2,1,1,2],0,dmc,derrc)
    mi,bi = uc_lin_reg([4,3,2,1,1,2],0,dmi,derri)

    lama = (1e-6*ma*d)/(4*f**2)
    lamc = (1e-6*mc*d)/(4*f**2)
    lami = (1e-6*mi*d)/(4*f**2)

    lamq = (lama+ lamc+ lami)/3

    Dlamac = np.mean(lamq/(8*f**2) * ((dma-dmc)*1e-6))
    Dlamai = np.mean(lamq/(8*f**2) * ((dma-dmi)*1e-6))
    Dlamci = np.mean(lamq/(8*f**2) * ((dmc-dmi)*1e-6))


    #Data Print ---------------------------
    print("Außen in mm²: m = ", ma , "b = ", ba)
    print("mitte in mm²: m = ", mc , "b = ", bc)
    print("Innen in mm²: m = ", mi , "b = ", bi)
    print("lambda außen = ", lama*1e9, "nm")
    print("lambda mitte = ", lamc*1e9, "nm")
    print("lambda innen = ", lami*1e9, "nm")
    print(Dlamac/lamq, Dlamai/lamq, Dlamci/lamq)

    #plt.plot(data0[0], data0[1], "m.-",label="Data0")   #nope
    #plt.plot(data1[0], data1[1], "b.-",label="Data1")   #good
    #plt.plot(data2[0], data2[1], "r.-",label="Data2")   #maybe (nope)
    #plt.plot(data3[0], data3[1], "g.-",label="Data3")   #nope
    #plt.plot(data4[0], data4[1], "y.-",label="Data4")   #good
    #plt.plot(data5[0], data5[1], "k.-",label="Data5")   #good

    xw = np.linspace(0.5,4.2,1000)
    plt.plot([4,3,2,1,1,2],dma,"bo", label = "Data Außen")
    plt.plot([4,3,2,1,1,2],dmi,"co", label = "Data Innen")
    plt.plot([4,3,2,1,1,2],dmc,"ko", label = "Data Mitte")
    plt.plot(xw,ma.n*xw+ba.n,"r-", label = ("Fit Außen: ("+ str(ma) + ") * x + (" + str(ba) + ")") )
    plt.plot(xw,mc.n*xw+bc.n,"y-", label = ("Fit Mitte: ("+ str(mc) + ") * x + (" + str(bc) + ")") )
    plt.plot(xw,mi.n*xw+bi.n,"m-", label = ("Fit Innen: ("+ str(mi) + ") * x + (" + str(bi) + ")") )

    plt.xlabel("Ordnung n")
    plt.ylabel("Durchmesser D² in mm²")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()
    return;


def main():
    d = laser()
    #green(d)
    uv_old(d)
    uv(d)
    return;

main()
