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
import numexpr, cProfile
import matplotlib.pyplot as plt

L = 6  # system size
paras = np.array([8.0, -2.0, 2.0, 0.5, 3.0, 1.0, -0.2, 0.2])
U = paras[0]
ef = paras[1]
eis_part = paras[2:2 + L]
hoppings_part = paras[2 + L:]
eis = np.concatenate((eis_part, eis_part))
hoppings = np.concatenate((hoppings_part, -1*hoppings_part))

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
print(static)
dynamic = []
no_checks = dict(check_pcon=False, check_symm=False, check_herm=True)

# ===================================== direct calculation of spectral functions using symmetries =====================

del L
L = occupancy

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

#
##### calculate action without constructing the Hamiltonian matrix
#
on_the_fly = False  # toggles between using a `hamiltnian` or a `quantum_LinearOperator` object
# this example does not work under these conditions because ground-state sector is not q=0
if (L // 2) % 2 != 0:
    raise ValueError("Example requires modifications for Heisenberg chains with L=4*n+2.")
if L % 2 != 0:
    raise ValueError("Example requires modifications for Heisenberg chains with odd number of sites.")
# construct basis
basis0 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))
# construct operator for Hamiltonian in the ground state sector
if on_the_fly:
    H0 = quantum_LinearOperator(static, basis=basis0, dtype=np.float64)
else:
    # basis=不可以省略，因为会变成别的缺省值
    H0 = hamiltonian(static, dynamic, basis=basis0, dtype=np.float64, **no_checks)
# calculate ground state
[E0], psi0 = H0.eigsh(k=1, which="SA")
psi0 = psi0.ravel()
# print(psi0)
# define frequencies to calculate spectral function for
omegas = np.arange(0, 4, 0.02)
# spectral peaks broadening factor
eta = 0.1
# allocate arrays to store data
# Anderson model for impurity node
Gpm = np.zeros(omegas.shape+(1,), dtype=np.complex128)
print(Gpm)
print("computing impurity site 0.")

# 产生算符，会导致电子增加，所以要加1
basisq = spinful_fermion_basis_general(N=L, Nf=(N_up+1, N_down))
Op_list = [["+|", [0], 1.0]]
# define operators in the q-momentum sector
if on_the_fly:
    Hq = quantum_LinearOperator(static, basis=basisq, dtype=np.complex128,
                                check_symm=False, check_pcon=False, check_herm=False)
else:
    Hq = hamiltonian(static, [], basis=basisq, dtype=np.complex128,
                     check_symm=False, check_pcon=False, check_herm=False)
# shift sectors
psiA = basisq.Op_shift_sector(basis0, Op_list, psi0)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS(Hq, omega, eta, E0)
    x, *_ = sp.linalg.bicg(lhs, psiA)
    print(i, x)
    Gpm[i,0] = -np.vdot(psiA, x) / np.pi
#
##### plot results
#
plt.plot(omegas, Gpm[:,0].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.show()
plt.close()
