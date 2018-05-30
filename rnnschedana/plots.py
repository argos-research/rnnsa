import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class COLORS:
    white = mcolors.hex2color('#ffffff')
    grey0 = mcolors.hex2color('#bbbbbb')
    grey1 = mcolors.hex2color('#999999')
    grey2 = mcolors.hex2color('#777777')
    grey3 = mcolors.hex2color('#555555')
    grey4 = mcolors.hex2color('#333333')
    black = mcolors.hex2color('#000000')

def activation_functions(ylim, xlim):
    ax = plt.subplot(111)
    #ax.set_title('Activation Functions')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    x = np.arange(xlim[0], xlim[1], 0.05)
    tanh = np.tanh(x)
    sigmoid = np.exp(x)/(1+np.exp(x))
    relu = np.maximum(x, 0)
    ax.plot(x, tanh, label='tanh', color=COLORS.grey0)
    ax.plot(x, sigmoid, label='sigmoid', color=COLORS.grey2)
    ax.plot(x, relu, label='ReLU', color=COLORS.grey4)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    activation_functions((-1.5, 1.5), (-6., 6.))
