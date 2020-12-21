import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const
from scipy import optimize

def plt_corr_data(I,Z):
    plt.plot(I,Z, "xr")
    plt.plot(I,Z, "-b")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Ereignisse (korrigiert)")
    plt.grid(True)
    plt.show()
    plt.clf()
    return;

def gausfitf(I, a1, m1, s1, a2, m2, s2):
    e1 = ((I-m1)**2)/(2*s1**2)
    e2 = ((I-m2)**2)/(2*s2**2)
    f = a1 * np.exp(-e1) + a2 * np.exp(-e2)
    return f;

def etaf(E):
    m = 9.11e-31
    ms = np.float128(m * (const.c)**2)
    eta = np.sqrt(( (E + ms)/ms)**2 - 1)
    return float(eta);

def plt_eichung_x(I, eta):
    m, t, r_v, p_v, std_err = stats.linregress(I, eta)
    x_w = np.linspace(-0.5, 9, 100)

    print(m,t, std_err)
    plt.plot(I,eta, "xr")
    plt.plot(x_w,m*x_w + t, "-b")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Reduzierter Impuls")
    plt.grid(True)
    plt.show()
    plt.clf()
    return;

def eta2eps(eta):
    return np.sqrt(eta**2 + 1)

def eps2eta(eps):
    return np.sqrt(eps**2 - 1)

def yKurie(N,eps):
    eta =eps2eta(eps)
    return np.sqrt( N/(eta*eps*GFermi(eta)) );

def GFermi(eta):
    A= 25.2007
    B= 23.6665,
    eps = eta2eps(eta)
    x = np.sqrt(A+ (B)/(eps-1))
    G = x* (eta)/(eps)
    return G;

def plot_kurie(eps,y):
    a, b, r_v, p_v, std_err = stats.linregress(eps, y)
    xv = np.linspace(1,2,100)
    print(a,b,std_err)

    plt.plot(eps,y, "xr")
    plt.plot(xv,a*xv + b, "-b")
    plt.xlabel("Reduzierte Energie")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
    plt.clf()

def plot_spline(cs,xs,eps,Z):
    plt.plot(eps,Z,"xr")
    plt.plot(xs, cs(xs), "-b")
    plt.xlabel("Reduzierte Energie")
    plt.ylabel("Ereignisse (korrigiert)")
    plt.grid(True)
    plt.show()
    plt.clf()
    return;
