from __future__ import print_function, division
#
import sys, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS'] = '1'  # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '1'  # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)

import numpy as np
import matplotlib.pyplot as plt
from examples.notebooks.spectral_function_fermion import spectral_function_fermion


def scale_up(z, x_min, x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return x_min + (z + 1) * (x_max - x_min) / 2


def scale_down(x, x_min, x_max):
    """
    Scales down x \in [x_min,x_max] to z \in [-1,1]
    where z = f(x) = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return (2 * (x - x_min) / (x_max - x_min)) - 1


def solver(order_n, fig_params):

    plot_row, plot_col = fig_params[0], fig_params[1]
    for n in [order_n]:
        plt.figure(14)
        fig_1, axs_1 = plt.subplots(plot_row, plot_col, figsize=(5 * plot_col, 2.5 * plot_row), facecolor='w',
                                    edgecolor='k')
        fig_1.subplots_adjust(hspace=0.5, wspace=0.5)
        axs_1 = axs_1.ravel()
        fig_1.suptitle("coefficients of the Chebyshev polynomial")

        plt.figure(2)
        fig_2, axs_2 = plt.subplots(plot_row, plot_col, figsize=(5 * plot_col, 2.5 * plot_row), facecolor='w',
                                    edgecolor='k')
        fig_2.subplots_adjust(hspace=0.5, wspace=0.5)
        axs_2 = axs_2.ravel()
        fig_2.suptitle("spectral function")

        plt.figure(3)
        fig_3, axs_3 = plt.subplots(plot_row, plot_col, figsize=(5 * plot_col, 2.5 * plot_row), facecolor='w',
                                    edgecolor='k')
        fig_3.subplots_adjust(hspace=0.5, wspace=0.5)
        axs_3 = axs_3.ravel()
        fig_3.suptitle("input parameters")

        # n = 112  # order (degree, highest power) of the approximating polynomial
        # m = 113  # number of Chebyshev nodes (having m > n doesn't matter for the approximation it seems)
        m = n + 1
        x_min = -25
        x_max = 25
        x_grid = np.linspace(x_min, x_max, 1000)
        r_k = np.polynomial.chebyshev.chebpts1(m)

        # builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
        # using the recurrence relation
        T = np.polynomial.chebyshev.chebvander(r_k, n)

        # calculate the Chebyshev coefficients
        x_k = scale_up(r_k, x_min, x_max)
        parameters = [[float(x) for x in d.strip().split(',')[1:]] for d in open('paras.csv').readlines()]
        for NN in range(min(plot_row * plot_col, len(parameters))):
            paras = np.array(parameters[NN])

            # 这里的x_k的值是非均匀的值，所以面积算出来的不准
            y_k = spectral_function_fermion(x_k, paras)
            alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k

            # Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
            T_pred = np.zeros((len(x_grid), n + 1))
            T_pred[:, 0] = np.ones((len(x_grid), 1)).T
            z_grid = scale_down(x_grid, x_min, x_max)
            T_pred[:, 1] = z_grid.T
            for i in range(1, n):
                T_pred[:, i + 1] = 2 * z_grid * T_pred[:, i] - T_pred[:, i - 1]
            Tf = T_pred @ alpha

            # this plot coefficients of the Chebyshev polynomial
            plt.figure(14)
            axs_1[NN].plot(alpha, 'ro', markersize=1)
            axs_1[NN].set_xlabel('i')
            axs_1[NN].set_ylabel('alpha')
            axs_1[NN].set_xlim([0, 256])
            # axs_1[NN].set_ylim([-1, 1])

            # this plot spectral function
            plt.figure(2)
            spectral_values = spectral_function_fermion(x_grid, paras)
            axs_2[NN].plot(x_grid, spectral_values)
            axs_2[NN].plot(x_grid, Tf)
            # axs_2[NN].title(f'm={m},n={n}')
            axs_2[NN].set_xlabel('$\\omega$')
            axs_2[NN].set_ylabel('A')
            axs_2[NN].set_xlim([-25, 25])
            axs_2[NN].set_ylim([0, 0.45])
            area = (x_grid[1] - x_grid[0]) * spectral_values[:].sum()
            print(f"{NN} area is : {area}")

            # this plot input parameters
            plt.figure(3)
            axs_3[NN].plot(np.arange(1, len(paras) + 1), paras, '-.o')
            axs_3[NN].set_xlabel('i')
            axs_3[NN].set_ylabel('parameters')
            axs_3[NN].set_xlim([0, 8])
            axs_3[NN].set_ylim([-4.3, 9.995])
        plt.show()


if __name__ == '__main__':
    # chebyshev parameters
    order_n = 256
    # plot parameters
    plot_row = 1
    plot_col = 2
    solver(order_n, (plot_row, plot_col))
