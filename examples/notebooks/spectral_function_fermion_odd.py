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

from examples.notebooks.utils.fermion_utils import LHS_gen, LHS_anni
L = 6  # system size
part = L // 2  # 根据对称性，进行参数缩减
paras = np.array([9.048, 0.2223, 1.7352, 1.1966, 3.7488, 0.2838, 0.4838, 1.1495])
U = paras[0]
ef = paras[1]
eis_part = paras[2:2 + part]
hoppings_part = paras[2 + part:]
eis = np.concatenate((eis_part, -1 * eis_part))
hoppings = np.concatenate((hoppings_part, hoppings_part))

# this is for (3, 4) basis
occupancy = L + 1
type_1 = occupancy // 2
type_2 = occupancy - type_1

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
    ['z|z', interaction]  # up-down interaction
]

dynamic = []
no_checks = dict(check_pcon=False, check_symm=False, check_herm=True)

plt.plot(paras, '-.o')
plt.ylim([-5, 10])
plt.grid()

# construct basis
basis0_A = spinful_fermion_basis_general(N=occupancy, Nf=(type_1, type_2))

H0_A = hamiltonian(static, dynamic, basis=basis0_A, dtype=np.float64, **no_checks)
# calculate ground state
[E0_A], psi0_A = H0_A.eigsh(k=1, which="SA")

from numpy.linalg import eig

c = eig(H0_A.toarray())
eigenvalues = sorted(c[0])
assert np.isclose(eigenvalues[:1], [E0_A])

del L

omegas = np.arange(-20, 20, 0.01)
# spectral peaks broadening factor
eta = 1

# allocate arrays to store data
# Anderson model for impurity node
Gpm_A = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_A = spinful_fermion_basis_general(N=occupancy, Nf=(type_1 + 1, type_2))
Op_list = [["+|", [0], 1.0]]

Hq_A = hamiltonian(static, [], basis=basisq_A, dtype=np.complex128,
                   check_symm=False, check_pcon=False, check_herm=False)

hqV_1 = Hq_A.eigvalsh()
hqE_2, HqV_2 = Hq_A.eigh()
c = eig(Hq_A.toarray())
assert np.isclose(hqV_1[:1], hqE_2[:1])
# shift sectors
psiA_A = basisq_A.Op_shift_sector(basis0_A, Op_list, psi0_A)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS_gen(Hq_A, omega, eta, E0_A)
    x, exitCode = sp.linalg.bicg(lhs, psiA_A)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_A)
    Gpm_A[i] = -np.vdot(psiA_A, x) / np.pi

# this is for (4, 3) basis
# construct basis
basis0_B = spinful_fermion_basis_general(N=occupancy, Nf=(type_2, type_1))

H0_B = hamiltonian(static, dynamic, basis=basis0_B, dtype=np.float64, **no_checks)
# calculate ground state
[E0_B], psi0_B = H0_B.eigsh(k=1, which="SA")

# allocate arrays to store data
# Anderson model for impurity node
Gpm_B = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_B = spinful_fermion_basis_general(N=occupancy, Nf=(type_2 + 1, type_1))
Hq_B = hamiltonian(static, [], basis=basisq_B, dtype=np.complex128,
                   check_symm=False, check_pcon=False, check_herm=False)
# shift sectors
psiA_B = basisq_B.Op_shift_sector(basis0_B, Op_list, psi0_B)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS_gen(Hq_B, omega, eta, E0_B)
    x, exitCode = sp.linalg.bicg(lhs, psiA_B)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_B)
    Gpm_B[i] = -np.vdot(psiA_B, x) / np.pi

#
##### plot results
#
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei']
Gpm = np.zeros_like(Gpm_B, dtype=np.complex128)
Gpm = Gpm_A[:] + Gpm_B[:]
plt.plot(omegas, Gpm[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$G_{+-}(\\omega)$')
plt.grid()
plt.show()
# plt.close()

# for result valid
del Op_list
Gpm_A_valid = np.zeros_like(Gpm_B, dtype=np.complex128)
Gpm_B_valid = np.zeros_like(Gpm_B, dtype=np.complex128)
Gpm_valid = np.zeros_like(Gpm_B, dtype=np.complex128)

# part A
# 产生算符，会导致电子增加，所以要加1
Op_list_down = [["|+", [0], 1.0]]
# 产生算符，会导致电子增加，所以要加1
basisq_A_valid = spinful_fermion_basis_general(N=occupancy, Nf=(type_1, type_2 + 1))

Hq_A_valid = hamiltonian(static, [], basis=basisq_A_valid, dtype=np.complex128,
                         check_symm=False, check_pcon=False, check_herm=False)

# shift sectors
psiA_A_valid = basisq_A_valid.Op_shift_sector(basis0_A, Op_list_down, psi0_A)
#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i, omega in enumerate(omegas):
    lhs = LHS_gen(Hq_A_valid, omega, eta, E0_A)
    x, exitCode = sp.linalg.bicg(lhs, psiA_A_valid)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_A_valid)
    Gpm_A_valid[i] = -np.vdot(psiA_A_valid, x) / np.pi

# part B
basisq_B_valid = spinful_fermion_basis_general(N=occupancy, Nf=(type_2, type_1 + 1))
Hq_B_valid = hamiltonian(static, [], basis=basisq_B_valid, dtype=np.complex128,
                         check_symm=False, check_pcon=False, check_herm=False)

# shift sectors
psiA_B_valid = basisq_B_valid.Op_shift_sector(basis0_B, Op_list_down, psi0_B)
for i, omega in enumerate(omegas):
    lhs = LHS_gen(Hq_B_valid, omega, eta, E0_B)
    x, exitCode = sp.linalg.bicg(lhs, psiA_B_valid)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA_B_valid)
    Gpm_B_valid[i] = -np.vdot(psiA_B_valid, x) / np.pi

Gpm_valid = Gpm_A_valid[:] + Gpm_B_valid[:]
assert np.allclose(Gpm[:1], Gpm_valid[:1])
print(Gpm[:10])
print(Gpm_valid[:10])

# '-' spin , part_A
Gpm_A_anni = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_A_anni = spinful_fermion_basis_general(N=occupancy, Nf=(type_1 - 1, type_2))
Op_list_anni = [["-|", [0], 1.0]]

Hq_A_anni = hamiltonian(static, [], basis=basisq_A_anni, dtype=np.complex128,
                        check_symm=False, check_pcon=False, check_herm=False)
# shift sectors
psiA_A_anni = basisq_A_anni.Op_shift_sector(basis0_A, Op_list_anni, psi0_A)

for i, omega in enumerate(omegas):
    lhs_anni = LHS_anni(Hq_A_anni, omega, eta, E0_A)
    x_anni, exitCode = sp.linalg.bicg(lhs_anni, psiA_A_anni)
    assert exitCode == 0
    np.allclose(lhs_anni._matvec(x_anni), psiA_A_anni)
    Gpm_A_anni[i] = -np.vdot(psiA_A_anni, x_anni) / np.pi

# '-' spin , part_B
Gpm_B_anni = np.zeros_like(omegas, dtype=np.complex128)

# 产生算符，会导致电子增加，所以要加1
basisq_B_anni = spinful_fermion_basis_general(N=occupancy, Nf=(type_2 - 1, type_1))

Hq_B_anni = hamiltonian(static, [], basis=basisq_B_anni, dtype=np.complex128,
                        check_symm=False, check_pcon=False, check_herm=False)
# shift sectors
psiA_B_anni = basisq_B_anni.Op_shift_sector(basis0_B, Op_list_anni, psi0_B)

for i, omega in enumerate(omegas):
    lhs_anni = LHS_anni(Hq_B_anni, omega, eta, E0_B)
    x_anni, exitCode = sp.linalg.bicg(lhs_anni, psiA_B_anni)
    assert exitCode == 0
    np.allclose(lhs_anni._matvec(x_anni), psiA_B_anni)
    Gpm_B_anni[i] = -np.vdot(psiA_B_anni, x_anni) / np.pi

Gpm_anni = np.zeros_like(omegas, dtype=np.complex128)
Gpm_anni = Gpm_A_anni + Gpm_B_anni

Green = np.zeros_like(omegas, dtype=np.complex128)
Green = 0.5*(Gpm + Gpm_anni)

area = (omegas[1]-omegas[0]) * Green[:].imag.sum()
print(f"========== area for tow parts : {area:.2f} ==========")
