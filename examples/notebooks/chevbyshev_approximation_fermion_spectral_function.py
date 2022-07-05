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
import scipy.sparse as sp

from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian


def zero_temperature_Green_function(omegas, params):
    print("Mock Green imaginary function.")
    return 1 / 3 * omegas ** 3 + 2 * omegas ** 2 + omegas - 10


# def f(x):
#     return 1 / 3 * x ** 3 + 2 * x ** 2 + x - 10


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


paras = [1.5205, 0.2055, 2.0274, 3.4384, 1.137, 1.2329, 1.4658, 0.4657]
x_min = -1
x_max = 3
x_grid = np.linspace(x_min, x_max, 100)
plt.figure()
plt.plot(x_grid, zero_temperature_Green_function(x_grid, paras))
plt.show()
plt.close()

n = 2  # order (degree, highest power) of the approximating polynomial
m = 3  # number of Chebyshev nodes (having m > n doesn't matter for the approximation it seems)

r_k = np.polynomial.chebyshev.chebpts1(m)

# builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
# using the recurrence relation
T = np.zeros((m, n + 1))

T[:, 0] = np.ones((m, 1)).T

T[:, 1] = r_k.T

for i in range(1, n):
    T[:, i + 1] = 2 * r_k * T[:, i] - T[:, i - 1]

# or numpy's routine
# T = np.polynomial.chebyshev.chebvander(r_k,n)

# calculate the Chebyshev coefficients
x_k = scale_up(r_k, x_min, x_max)
y_k = zero_temperature_Green_function(x_k, paras)
alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k

# Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
T = np.zeros((len(x_grid), n + 1))

T[:, 0] = np.ones((len(x_grid), 1)).T

z_grid = scale_down(x_grid, x_min, x_max)

T[:, 1] = z_grid.T

for i in range(1, n):
    T[:, i + 1] = 2 * z_grid * T[:, i] - T[:, i - 1]

# compute approximation
Tf = T @ alpha
# make sure to use the scaled down grid inside the Chebyshev expansion
plt.figure()
plt.plot(x_grid, zero_temperature_Green_function(x_grid, paras))
plt.plot(x_grid, Tf)
plt.show()
plt.close()
