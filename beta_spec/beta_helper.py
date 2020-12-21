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

    print(m,t)
    plt.plot(I,eta, "xr")
    plt.plot(x_w,m*x_w + t, "-b")
    plt.xlabel("Stromstärke in A")
    plt.ylabel("Reduzierter Impuls")
    plt.grid(True)
    plt.show()
    plt.clf()
    return;
