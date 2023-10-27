import matplotlib.pyplot as plt
import random

random.seed(19260817)

MARKERS = "ov^<>8sphHD"


def plot_0_0(Ds, Ts, names, markers, save_name, title, xlabel, ylabel):
    for D, T, name, marker in zip(Ds, Ts, names, markers):
        plt.plot(T, D, label=name, marker=marker)

    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".pdf")
    plt.close()
