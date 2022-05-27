import numpy as np

from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian

L = 3  # system size
paras = np.array([7.0, -3.5, 2.0, 2.3, 2.1, 0.6, -0.2, 0.5])
# yang
# paras = np.array([8.0, -2.0, 2.0, 0.5, 3.0, 1.0, -0.2, 0.2])
U = paras[0]
ef = paras[1]
eis = paras[2:2 + L]
hoppings = paras[2 + L:]

occupancy = L + 1
N_up = occupancy // 2
N_down = occupancy - N_up

basis = spinful_fermion_basis_1d(L=occupancy, Nf=(N_up, N_down))
basis_general = spinful_fermion_basis_general(N=occupancy,Nf=(N_up, N_down))
print("================ spinful_fermion_basis_1d ================")
print(basis)

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
print(static)
dynamic = []
no_checks = dict(check_pcon=False, check_symm=False, check_herm=True)
# basis=不可以省略，因为会变成别的缺省值
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64, **no_checks)
print("================ spinful_fermion_basis_general ================")
print(basis_general)
H_general = hamiltonian(static, dynamic, basis=basis_general, dtype=np.float64, **no_checks)

# ===================================== direct calculation of spectral functions using symmetries =====================


