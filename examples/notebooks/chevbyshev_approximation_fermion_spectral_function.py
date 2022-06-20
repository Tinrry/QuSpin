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


#
#
# define custom LinearOperator object that generates the left hand side of the equation.
#
class LHS(sp.linalg.LinearOperator):
    #
    def __init__(self, H, omega, eta, E0, kwargs={}):
        self._H = H  # Hamiltonian
        self._z = omega + 1j * eta + E0  # complex energy
        self._kwargs = kwargs  # arguments

    #
    @property
    def shape(self):
        return (self._H.Ns, self._H.Ns)

    #
    @property
    def dtype(self):
        return np.dtype(self._H.dtype)

    #
    def _matvec(self, v):
        # left multiplication
        return self._z * v - self._H.dot(v, **self._kwargs)

    #
    def _rmatvec(self, v):
        # right multiplication
        return self._z.conj() * v - self._H.dot(v, **self._kwargs)


L = 6  # system size
part = L // 2  # 根据对称性，进行参数缩减
paras = np.array([9.0, 0.0, 2.0, 1.8, 4.0, 0.0, 0.2, 0.0])
U = paras[0]
ef = paras[1]
eis_part = paras[2:2 + part]
hoppings_part = paras[2 + part:]
eis = np.concatenate((eis_part, eis_part))
hoppings = np.concatenate((hoppings_part, -1 * hoppings_part))

occupancy = L + 1
N_up = occupancy // 2
N_down = occupancy - N_up

# hop_to_xxx， hop_from_xxx都是正符号，同一个系数，在hamiltonian中已经定好了符号
hop_to_impurity = [[hoppings[i], 0, i + 1] for i in range(L)]
hop_from_impurity = [[hoppings[i], i + 1, 0] for i in range(L)]
pot = [[ef, 0]] + [[eis[i], i + 1] for i in range(L)]
interaction = [[U, 0, 0]]

# 在符号上 都用‘-+’ 或者‘+-’，不可以掺杂
static = [
    ['-+|', hop_from_impurity],
    ['-+|', hop_to_impurity],
    ['|-+', hop_from_impurity],
    ['|-+', hop_to_impurity],
    ['n|', pot],  # up on-site potention
    ['|n', pot],  # down on-site potention
    ['n|n', interaction]  # up-down interaction
]
dynamic = []
no_checks = dict(check_pcon=False, check_symm=False, check_herm=True)

# construct basis
basis0 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))
H0 = hamiltonian(static, dynamic, basis=basis0, dtype=np.float64, **no_checks)
# calculate ground state
[E0], psi0 = H0.eigsh(k=1, which="SA")
psi0 = psi0.ravel()

# spectral peaks broadening factor
eta = 0.1
del L
L = occupancy

# 产生算符，会导致电子增加，所以要加1
basisq = spinful_fermion_basis_general(N=L, Nf=(N_up + 1, N_down))
Op_list = [["+|", [0], f(0)]]
Hq = hamiltonian(static, dynamic, basis=basisq, dtype=np.complex128, check_symm=False, check_pcon=False,
                 check_herm=False)
# shift sectors
psiA = basisq.Op_shift_sector(basis0, Op_list, psi0)


# fermion spectral function
def f(omega):
    lhs = LHS(Hq, omega, eta, E0)
    x, *_ = sp.linalg.bicg(lhs, psiA)
    return -np.vdot(psiA, x) / np.pi
    return 1 / 3 * omega ** 3 + 2 * omega ** 2 + omega - 10


x_min = 0
x_max = 4
x_grid = np.linspace(x_min, x_max, 100)
Gpm = np.zeros(x_grid.shape + (1,), dtype=np.complex128)
for i, omega in enumerate(x_grid):
    Gpm[i, 0] = f(omega)

#
#####################################################################
#
# ---------------------------------------------------- #
#            chebyshev approximation                   #
# ---------------------------------------------------- #
#

n = 10  # order (degree, highest power) of the approximating polynomial  ----1,later 2 debug
m = 17  # number of Chebyshev nodes (having m > n doesn't matter for the approximation it seems)


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


alpha = np.zeros((psiA.shape + (n + 1,)))
mu = np.zeros((1, n + 1))

# 这里为了求 mu
alpha[:, 0] = psiA
alpha[:, 1] = Hq.dot(psiA)
mu[:, 0] = alpha[:, 0].T.dot(alpha[:, 0])
mu[:, 1] = alpha[:, 0].T.dot(alpha[:, 1])
for i in range(1, n):
    alpha[:, i + 1] = 2 * Hq.dot(alpha[:, i]) - alpha[:, i - 1]  # overflow
    mu[:, i + 1] = alpha[:, 0].T.dot(alpha[:, i + 1])

z_grid = scale_down(x_grid, x_min, x_max)
T = np.zeros((len(x_grid), n + 1)) + 0.0j

T[:, 0] = np.cos(0 * np.arccos(z_grid))
T[:, 1] = np.cos(1 * np.arccos(z_grid))
for i in range(n + 1):
    T[:, i] = np.cos(i * np.arccos(z_grid))
    if i > 1:
        temp_t = 2 * np.multiply(z_grid, T[:, i - 1]) - T[:, i - 2]
        assert np.allclose(np.real(temp_t), np.real(T[:, i]), rtol=1e-5)

Tf = T @ mu.T
assert Tf.shape[0] == x_grid.shape[0]
assert Tf.shape == Gpm.shape

# make sure to use the scaled down grid inside the Chebyshev expansion
plt.figure()
plt.plot(x_grid, Gpm[:, 0].imag)
plt.plot(x_grid, Tf.imag)
plt.xlabel('$\\omega$')
plt.ylabel('$spectral function')
plt.title('chevbyshev approximation vs fermion')
plt.show()

# plot approximation error
plt.figure()
plt.plot(x_grid, Gpm[:, 0].imag - Tf.imag)
plt.xlabel('$\\omega$')
plt.ylabel('$error')
plt.title('chevbyshev approximation vs fermion')
plt.show()
plt.close()

# chebyshev 的 m 个点
# generate chebyshev nodes (the roots of Chebyshev polynomials, a Chebyshev polynomial of degree m-1 has m roots)
# r_k = -np.cos((2 * np.arange(1, m + 1) - 1) * np.pi / (2 * m))

# or using numpy's routine
# r_k = np.polynomial.chebyshev.chebpts1(m)


# builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
# using the recurrence relation
# T = np.zeros((m, n + 1))
#
# # psiA
# T[:, 0] = psiA @ np.ones((m, 1)).T
# assert T[:, 0].shape == psiA.shape+(m,)            # shape should be psiA.shape+(m,)
#
# T[:, 1] = Hq.dot(psiA).dot(r_k.T)
# assert T[:, 1].shape == psiA.shape+(m,)
#
# for i in range(1, n):
#     T[:, i + 1] = 2 * r_k * T[:, i] - T[:, i - 1]
#
# # or numpy's routine
# # T = np.polynomial.chebyshev.chebvander(r_k,n)
#
# # calculate the Chebyshev coefficients
# # x_k = scale_up(r_k, x_min, x_max)
# # y_k = f(x_k)
# alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k
#
# # 这里为了用 alpha 做预测
# # Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
# T = np.zeros((len(x_grid), n + 1))
#
# T[:, 0] = np.ones((len(x_grid), 1)).T
#
# z_grid = scale_down(x_grid, x_min, x_max)
#
# T[:, 1] = z_grid.T

# for i in range(1, n):
#     T[:, i + 1] = 2 * z_grid * T[:, i] - T[:, i - 1]
#
# # compute approximation
# Tf = T @ alpha
#
# # make sure to use the scaled down grid inside the Chebyshev expansion
# plt.figure()
# plt.plot(x_grid, f(x_grid))
# plt.plot(x_grid, Tf)
# plt.show()
# plt.close()
#
# # plot approximation error
# plt.figure()
# plt.plot(x_grid, f(x_grid) - Tf)
# plt.show()
# plt.close()
