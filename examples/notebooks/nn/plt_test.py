#!/usr/bin/env python
# !encoding=utf-8

import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


if __name__ == '__main__':
    plt.figure(1)
    fig_1, axs_1 = plt.subplots(1, 4, figsize=(15, 4), facecolor='w', edgecolor='k')
    fig_1.subplots_adjust(hspace=.5, wspace=.1)
    axs_1 = axs_1.ravel()

    plt.figure(2)
    fig_2, axs_2 = plt.subplots(2, 2, figsize=(8, 7))
    fig_2.subplots_adjust(hspace=.5, wspace=.1)
    axs_2 = axs_2.ravel()
    t1 = np.arange(0, 5, 0.1)

    for i in range(4):
        plt.figure(1)
        axs_1[i].contourf(np.random.rand(10, 10), 5, cmap=plt.cm.Oranges)
        axs_1[i].set_title(str(250 + i))

        plt.figure(2)
        axs_2[i].plot(f(t1), 'bo')
    plt.show()
