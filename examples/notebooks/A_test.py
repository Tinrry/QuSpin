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

from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian,quantum_LinearOperator
import scipy.sparse as sp
import numexpr,cProfile
import matplotlib.pyplot as plt

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
hop_to_impurity = [[hoppings[i], 0, i+1] for i in range(L)]
hop_from_impurity = [[hoppings[i], i+1, 0] for i in range(L)]
pot = [[ef, 0]] + [[eis[i], i+1] for i in range(L)]
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
basis0 = spinful_fermion_basis_general(N=occupancy,Nf=(N_up, N_down))

# construct operator for Hamiltonian in the ground state sector
H0 = hamiltonian(static,dynamic,basis=basis0,dtype=np.float64, **no_checks)
# calculate ground state
[E0],psi0 = H0.eigsh(k=1,which="SA")
psi0 = psi0.ravel()
print(E0,psi0)

# 计算行列式的特征值，特征向量
from numpy.linalg import eig
c = eig(H0.toarray())
c = sorted(c)
print(c[0], c[1])

# ===================================== direct calculation of spectral functions using symmetries =====================

del L
L = occupancy
#######################################################################
#                            example 26                               #
# This example shows how to use the `Op-shit_sector` method of the    #
# general basis class to compute spectral functions using symmetries. #
#######################################################################

#
#
# define custom LinearOperator object that generates the left hand side of the equation.
#
class LHS(sp.linalg.LinearOperator):
    #
    def __init__(self,H,omega,eta,E0,kwargs={}):
        self._H = H # Hamiltonian
        self._z = omega +1j*eta + E0 # complex energy
        self._kwargs = kwargs # arguments
    #
    @property
    def shape(self):
        return (self._H.Ns,self._H.Ns)
    #
    @property
    def dtype(self):
        return np.dtype(self._H.dtype)
    #
    def _matvec(self,v):
        # left multiplication
        return self._z * v - self._H.dot(v,**self._kwargs)
    #
    def _rmatvec(self,v):
        # right multiplication
        return self._z.conj() * v - self._H.dot(v,**self._kwargs)


# define frequencies to calculate spectral function for
omegas = np.arange(0, 1, 0.2)
# spectral peaks broadening factor
eta = 0.1
# allocate arrays to store data
# Anderson model for impurity node
Gpm = np.zeros_like(omegas)+0.0j

basisq = spinful_fermion_basis_general(N=L, Nf=(N_up+1, N_down))
Op_list = [["+|", [0], 1.0 ]]
# define operators in the q-momentum sector
Hq = hamiltonian(static, [], basis=basisq, dtype=np.complex128,
        check_symm=False, check_pcon=False, check_herm=False)

hqV_1 = Hq.eigvalsh()
hqE_2, HqV_2 = Hq.eigh()

c = eig(Hq.toarray())

# shift sectors
psiA = basisq.Op_shift_sector(basis0, Op_list, psi0)
print(psiA)

#
### apply vector correction method
#
# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
for i,omega in enumerate(omegas):
    lhs = LHS(Hq,omega,eta,E0)
    x, exitCode = sp.linalg.bicg(lhs,psiA)
    assert exitCode == 0
    np.allclose(lhs._matvec(x), psiA)
    Gpm[i] = -np.vdot(psiA,x)/np.pi


plt.plot(omegas,Gpm[:].imag)
plt.xlabel('$\\omega$')
plt.ylabel('$Gpm.imag$')
plt.title('$A(\\omega)$')
plt.show()
plt.close()
