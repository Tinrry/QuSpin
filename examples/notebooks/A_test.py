from __future__ import print_function, division
import sys, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS'] = '1'  # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '1'  # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)

import numpy as np

from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import matplotlib.pyplot as plt

from examples.notebooks.utils.fermion_utils import LHS_gen, LHS_anni

L = 1  # system size

paras = np.array([8.0, -2.0, 2.0, 0.5])
U = paras[0]
ef = paras[1]
eis = paras[2:2 + L]
hoppings = paras[2 + L:]

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

# construct operator for Hamiltonian in the ground state sector
H0 = hamiltonian(static, dynamic, basis=basis0, dtype=np.float64, **no_checks)
# calculate ground state
[E0], psi0 = H0.eigsh(k=1, which="SA")
psi0 = psi0.ravel()
print(E0, psi0)

# 计算行列式的特征值，特征向量
from numpy.linalg import eig

c = eig(H0.toarray())
print(c[0], c[1])

del L

# define frequencies to calculate spectral function for
omegas = np.arange(-10, 10, 0.02)
# spectral peaks broadening factor
eta = 0.1
# allocate arrays to store data
# Anderson model for impurity node
Gpm_gen = np.zeros_like(omegas) + 0.0j

basisq_gen = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
Op_list_gen = [["+|", [0], 1.0]]
# define operators in the q-momentum sector
Hq_gen = hamiltonian(static, [], basis=basisq_gen, dtype=np.complex128,
                     check_symm=False, check_pcon=False, check_herm=False)

hqE_1 = Hq_gen.eigvalsh()
hqE_2, HqV_2 = Hq_gen.eigh()
assert hqE_1[0] == hqE_2[0]
c = eig(Hq_gen.toarray())

# shift sectors
psiA_gen = basisq_gen.Op_shift_sector(basis0, Op_list_gen, psi0)
print(psiA_gen)

#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS_gen(Hq_gen, omega, eta, E0)
    x, exitCode = sp.linalg.bicg(lhs, psiA_gen)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_gen)
    Gpm_gen[i] = -np.vdot(psiA_gen, x) / np.pi

plt.plot(omegas, Gpm_gen[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$A(\\omega)$')
plt.show()
plt.close()

# for result valid
#
#
# define custom LinearOperator object that generates the left hand side of the equation.
#

Gpm_anni = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_anni = spinful_fermion_basis_general(N=occupancy, Nf=(N_up - 1, N_down))
Op_list_anni = [["-|", [0], 1.0]]

Hq_anni = hamiltonian(static, [], basis=basisq_anni, dtype=np.complex128,
                        check_symm=False, check_pcon=False, check_herm=False)

hqE_1_anni = Hq_anni.eigvalsh()
hqE_2_anni, HqV_2_anni = Hq_anni.eigh()
c_anni = eig(Hq_anni.toarray())
assert np.isclose(hqE_1_anni[:1], hqE_2_anni[:1])
# shift sectors
psiA_anni = basisq_anni.Op_shift_sector(basis0, Op_list_anni, psi0)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs_anni = LHS_anni(Hq_anni, omega, eta, E0)
    x_anni, exitCode = sp.linalg.bicg(lhs_anni, psiA_anni)
    assert exitCode == 0
    np.allclose(lhs_anni._matvec(x_anni), psiA_anni)
    Gpm_anni[i] = -np.vdot(psiA_anni, x_anni) / np.pi
#
##### plot results
#
plt.plot(omegas, Gpm_anni[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm_mirror.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.show()
# plt.close()

area = (omegas[1]-omegas[0]) * Gpm_gen[:].imag.sum()
area_anni = (omegas[1]-omegas[0]) * Gpm_anni[:].imag.sum()
print("area: ",area)
print("area_anni: ",area_anni)
print(f"========== area for tow parts : {(area+area_anni):.2f} ==========")

