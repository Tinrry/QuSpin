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

from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import matplotlib.pyplot as plt

L = 6  # system size
part = L // 2  # 根据对称性，进行参数缩减
paras = np.array([9.0, 0.0, 2.0, 1.8, 4.0, 0.0, 0.2, 0.0])
U = paras[0]
ef = paras[1]
eis_part = paras[2:2 + part]
hoppings_part = paras[2 + part:]
eis = np.concatenate((eis_part, -1 * eis_part))
hoppings = np.concatenate((hoppings_part, hoppings_part))

# this is for (3, 4) basis
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
basis0_down = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

H0_down = hamiltonian(static, dynamic, basis=basis0_down, dtype=np.float64, **no_checks)
# calculate ground state
[E0_down], psi0_down = H0_down.eigsh(k=1, which="SA")

from numpy.linalg import eig

c = eig(H0_down.toarray())
eigenvalues = sorted(c[0])
assert np.isclose(eigenvalues[:1], [E0_down])

del L

omegas = np.arange(-20, 20, 0.01)
# spectral peaks broadening factor
eta = 1


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


# allocate arrays to store data
# Anderson model for impurity node
Gpm_down = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_down = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
Op_list = [["+|", [0], 1.0]]

Hq_down = hamiltonian(static, [], basis=basisq_down, dtype=np.complex128,
                      check_symm=False, check_pcon=False, check_herm=False)

hqV_1 = Hq_down.eigvalsh()
hqE_2, HqV_2 = Hq_down.eigh()
c = eig(Hq_down.toarray())
assert np.isclose(hqV_1[:1], hqE_2[:1])
# shift sectors
psiA_down = basisq_down.Op_shift_sector(basis0_down, Op_list, psi0_down)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS(Hq_down, omega, eta, E0_down)
    x, exitCode = sp.linalg.bicg(lhs, psiA_down)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_down)
    Gpm_down[i] = -np.vdot(psiA_down, x) / np.pi

# this is for (4, 3) basis
N_down = occupancy // 2
N_up = occupancy - N_down

# construct basis
basis0_up = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

H0_up = hamiltonian(static, dynamic, basis=basis0_up, dtype=np.float64, **no_checks)
# calculate ground state
[E0_up], psi0_up = H0_up.eigsh(k=1, which="SA")

# allocate arrays to store data
# Anderson model for impurity node
Gpm_up = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_up = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
Hq_up = hamiltonian(static, [], basis=basisq_up, dtype=np.complex128,
                    check_symm=False, check_pcon=False, check_herm=False)
# shift sectors
psiA_up = basisq_up.Op_shift_sector(basis0_up, Op_list, psi0_up)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS(Hq_up, omega, eta, E0_up)
    x, exitCode = sp.linalg.bicg(lhs, psiA_up)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_up)
    Gpm_up[i] = -np.vdot(psiA_up, x) / np.pi

#
##### plot results
#
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,10)
mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei']
Gpm = np.zeros_like(Gpm_up, dtype=np.complex128)
Gpm = Gpm_down[:] + Gpm_up[:]
plt.plot(omegas, Gpm[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.grid()
plt.show()
# plt.close()

# # for result valid
# #
# #
# Gpm_down_valid = np.zeros_like(Gpm_up, dtype=np.complex128)
# Gpm_up_valid = np.zeros_like(Gpm_up, dtype=np.complex128)
# Gpm_valid = np.zeros_like(Gpm_up, dtype=np.complex128)
#
# # 产生算符，会导致电子增加，所以要加1
# Op_list_down = [["|+", [0], 1.0]]
#
#
# # this is for (3, 4) basis
# N_up = occupancy // 2
# N_down = occupancy - N_up
#
# basisq_down_valid = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down + 1))
# del Op_list, basisq_down
#
# Hq_down_valid = hamiltonian(static, [], basis=basisq_down_valid, dtype=np.complex128,
#                             check_symm=False, check_pcon=False, check_herm=False)
#
# del Hq_down
# # shift sectors
# psiA_down_valid = basisq_down_valid.Op_shift_sector(basis0_down, Op_list_down, psi0_down)
# del psiA_down
# #
# ### apply vector correction method
# #
# # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
# for i, omega in enumerate(omegas):
#     lhs = LHS(Hq_down_valid, omega, eta, E0_down)
#     x, exitCode = sp.linalg.bicg(lhs, psiA_down_valid)
#     assert exitCode == 0
#     np.allclose(lhs._matvec(x), psiA_down_valid)
#     Gpm_down_valid[i] = -np.vdot(psiA_down_valid, x) / np.pi
#
#
# # this is for (4, 3) basis
# N_down = occupancy // 2
# N_up = occupancy - N_down
#
# basisq_up_valid = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down + 1))
# del basisq_up
# Hq_up_valid = hamiltonian(static, [], basis=basisq_up_valid, dtype=np.complex128,
#                           check_symm=False, check_pcon=False, check_herm=False)
#
# del Hq_up
# # shift sectors
# psiA_up_valid = basisq_up_valid.Op_shift_sector(basis0_up, Op_list_down, psi0_up)
# del psiA_up
# #
# ### apply vector correction method
# #
# # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
# for i, omega in enumerate(omegas):
#     lhs = LHS(Hq_up_valid, omega, eta, E0_up)
#     x, exitCode = sp.linalg.bicg(lhs, psiA_up_valid)
#     assert exitCode == 0
#     np.allclose(lhs._matvec(x), psiA_up_valid)
#     Gpm_up_valid[i] = -np.vdot(psiA_up_valid, x) / np.pi
#
# Gpm_valid = Gpm_down_valid[:] + Gpm_up_valid[:]
# assert np.allclose(Gpm[:1], Gpm_valid[:1])
# print(Gpm[:10])
# print(Gpm_valid[:10])
