import numpy as np
import PhyPraKit as ppk #wurde von mir verändert
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as const

import uncertainties as uc
from uncertainties.umath import sqrt

def task2():
    data = np.loadtxt("task2_data.csv", delimiter = ",", unpack = True, skiprows = 1)
    # data[x][y] x = colums || y = rows
    return;


def main():
    task2()
    return;


main()
