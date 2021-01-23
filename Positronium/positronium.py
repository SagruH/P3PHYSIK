import numpy as np
import matplotlib.pyplot as plt


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

def aufgabe2():     #zeitliche Auflösung
    data = np.loadtxt("data/A2_zeitliche_Aufloesung.RPT", unpack = True)

    return;

def main():
    #aufgabe1()
    aufgabe2()
    return;

main()
