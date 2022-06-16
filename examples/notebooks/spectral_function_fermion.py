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

from numpy.linalg import eig

c = eig(H0.toarray())
eigenvalues = sorted(c[0])
assert np.isclose(eigenvalues[:1], [E0])

del L
L = occupancy
omegas = np.arange(-20, 20, 0.01)
# spectral peaks broadening factor
eta = 0.1


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
Gpm = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq = spinful_fermion_basis_general(N=L, Nf=(N_up + 1, N_down))
Op_list = [["+|", [0], 1.0]]

Hq = hamiltonian(static, [], basis=basisq, dtype=np.complex128,
                 check_symm=False, check_pcon=False, check_herm=False)

hqV_1 = Hq.eigvalsh()
hqE_2, HqV_2 = Hq.eigh()
c = eig(Hq.toarray())
assert np.isclose(hqV_1[:1], hqE_2[:1])
# shift sectors
psiA = basisq.Op_shift_sector(basis0, Op_list, psi0)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS(Hq, omega, eta, E0)
    x, exitCode = sp.linalg.bicg(lhs, psiA)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA)
    Gpm[i] = -np.vdot(psiA, x) / np.pi
#
##### plot results
#
plt.plot(omegas, Gpm[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.show()
# plt.close()
area = (omegas[1]-omegas[0]) * Gpm[:].imag.sum()

# for result valid
#
#
# define custom LinearOperator object that generates the left hand side of the equation.
#
class LHS_mirror(sp.linalg.LinearOperator):
    #
    def __init__(self, H, omega, eta, E0, kwargs={}):
        self._H = H  # Hamiltonian
        self._z = omega + 1j * eta - E0  # complex energy
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
        return self._z * v + self._H.dot(v, **self._kwargs)

    #
    def _rmatvec(self, v):
        # right multiplication
        return self._z.conj() * v + self._H.dot(v, **self._kwargs)


Gpm_mirror = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_mirror = spinful_fermion_basis_general(N=L, Nf=(N_up - 1, N_down))
Op_list_mirror = [["-|", [0], 1.0]]

Hq_mirror = hamiltonian(static, [], basis=basisq_mirror, dtype=np.complex128,
                        check_symm=False, check_pcon=False, check_herm=False)

hqE_1_mirror = Hq_mirror.eigvalsh()
hqE_2_mirror, HqV_2_mirror = Hq_mirror.eigh()
c_mirror = eig(Hq_mirror.toarray())
assert np.isclose(hqE_1_mirror[:1], hqE_2_mirror[:1])
# shift sectors
psiA_mirror = basisq_mirror.Op_shift_sector(basis0, Op_list_mirror, psi0)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs_mirror = LHS_mirror(Hq_mirror, omega, eta, E0)
    x_mirror, exitCode = sp.linalg.bicg(lhs_mirror, psiA_mirror)
    assert exitCode == 0
    np.allclose(lhs_mirror._matvec(x_mirror), psiA_mirror)
    Gpm_mirror[i] = -np.vdot(psiA_mirror, x_mirror) / np.pi
#
##### plot results
#
plt.plot(omegas, Gpm_mirror[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm_mirror.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.show()
# plt.close()
area_mirror = (omegas[1]-omegas[0]) * Gpm_mirror[:].imag.sum()

print(f"========== area for tow parts : {(area+area_mirror):.2f} ==========")
