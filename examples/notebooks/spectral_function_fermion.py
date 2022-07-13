import numpy as np
from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import numexpr, cProfile
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei']

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 6)


class LHS(sp.linalg.LinearOperator):
    #
    def __init__(self, H, omega, eta, E0, isparticle=True, kwargs={}):
        if isparticle:
            self._H = H  # Hamiltonian
            self._z = omega + 1j * eta + E0  # complex energy
        else:
            self._H = -H  # Hamiltonian
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
        return self._z * v - self._H.dot(v, **self._kwargs)

    #
    def _rmatvec(self, v):
        # right multiplication
        return self._z.conj() * v - self._H.dot(v, **self._kwargs)


def spectral_function_fermion(omegas, paras):
    # config
    L = 6  # system size
    # L = 4  # system size
    part = L // 2  # 根据对称性，进行参数缩减
    # paras = np.array([9.0, 0.0, 2.0, 1.8, 4.0, 0.0, 0.2, 0.0])

    # paras = np.array([9.0, 0.0, 2.0, 1.8, 0.0, 0.2])
    U = paras[0]
    ef = paras[1]
    eis_part = paras[2:2 + part]
    hoppings_part = paras[2 + part:]
    eis = np.concatenate((eis_part, -1 * eis_part))
    hoppings = np.concatenate((hoppings_part, hoppings_part))

    # hop_to_xxx， hop_from_xxx都是正符号，同一个系数，在hamiltonian中已经定好了符号
    hop_to_impurity = [[hoppings[i], 0, i + 1] for i in range(L)]
    hop_from_impurity = [[hoppings[i], i + 1, 0] for i in range(L)]
    pot = [[ef, 0]] + [[eis[i], i + 1] for i in range(L)]
    interaction = [[U, 0, 0]]
    # end config

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

    # spectral peaks broadening factor
    eta = 0.9

    Green = np.zeros_like(omegas, dtype=np.complex128)

    cdagger_op = [["+|", [0], 1.0]]
    c_op = [["-|", [0], 1.0]]

    # this is for (3, 4) basis
    occupancy = L + 1
    N_up = occupancy // 2
    N_down = occupancy - N_up

    dynamic = []
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

    # construct basis
    basis_GS = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

    H0 = hamiltonian(static, dynamic, basis=basis_GS, dtype=np.float64, **no_checks)
    # calculate ground state
    [E0], GS = H0.eigsh(k=1, which="SA")

    # 产生算符，会导致电子增加，所以要加1
    basis_H1 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
    H1 = hamiltonian(static, [], basis=basis_H1, dtype=np.complex128, **no_checks)

    # shift sectors, |A> = c^\dagger |GS>
    psiA = basis_H1.Op_shift_sector(basis_GS, cdagger_op, GS)
    # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
    #   |x> = (z+E0-H)^-1 c^\dagger |GS>
    for i, omega in enumerate(omegas):
        lhs = LHS(H1, omega, eta, E0)
        x, exitCode = sp.linalg.bicg(lhs, psiA)
        assert exitCode == 0
        np.allclose(lhs._matvec(x), psiA)
        Green[i] += -np.vdot(psiA, x) / np.pi

    # 湮灭算符，会导致电子减少，所以要减1
    basis_H2 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up - 1, N_down))
    H2 = hamiltonian(static, [], basis=basis_H2, dtype=np.complex128, **no_checks)

    # shift sectors, |A> = c |GS>
    psiA = basis_H2.Op_shift_sector(basis_GS, c_op, GS)
    # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
    #   |x> = (z-E0+H)^-1 c |GS>
    for i, omega in enumerate(omegas):
        lhs = LHS(H2, omega, eta, E0, isparticle=False)
        x, exitCode = sp.linalg.bicg(lhs, psiA)
        assert exitCode == 0
        np.allclose(lhs._matvec(x), psiA)
        Green[i] += -np.vdot(psiA, x) / np.pi

    # this is for (4, 3) basis

    # if 2 *(occupancy//2) != occupancy:
    if 0:
        N_down = occupancy // 2
        N_up = occupancy - N_down

        dynamic = []
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

        # construct basis
        basis_GS = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

        H0 = hamiltonian(static, dynamic, basis=basis_GS, dtype=np.float64, **no_checks)
        # calculate ground state
        [E0], GS = H0.eigsh(k=1, which="SA")

        # 产生算符，会导致电子增加，所以要加1
        basis_H1 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
        H1 = hamiltonian(static, [], basis=basis_H1, dtype=np.complex128, **no_checks)

        # shift sectors, |A> = c^\dagger |GS>
        psiA = basis_H1.Op_shift_sector(basis_GS, cdagger_op, GS)
        # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
        #   |x> = (z+E0-H)^-1 c^\dagger |GS>
        for i, omega in enumerate(omegas):
            lhs = LHS(H1, omega, eta, E0)
            x, exitCode = sp.linalg.bicg(lhs, psiA)
            assert exitCode == 0
            np.allclose(lhs._matvec(x), psiA)
            Green[i] += -np.vdot(psiA, x) / np.pi

        # 湮灭算符，会导致电子减少，所以要减1
        basis_H2 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up - 1, N_down))
        H2 = hamiltonian(static, [], basis=basis_H2, dtype=np.complex128, **no_checks)

        # shift sectors, |A> = c^ |GS>
        psiA = basis_H2.Op_shift_sector(basis_GS, c_op, GS)
        # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
        #   |x> = (z-E0+H)^-1 c |GS>
        for i, omega in enumerate(omegas):
            lhs = LHS(H2, omega, eta, E0, isparticle=False)
            x, exitCode = sp.linalg.bicg(lhs, psiA)
            assert exitCode == 0
            np.allclose(lhs._matvec(x), psiA)
            Green[i] += -np.vdot(psiA, x) / np.pi

        Green *= 0.5

    return Green[:].imag


if __name__ == '__main__':
    NN = 4
    parameters = [[float(x) for x in d.strip().split(',')[1:]] for d in open('paras.csv').readlines()]
    print(parameters[2])
    paras = np.array(parameters[NN])

    omegas = np.arange(-10, 10, 0.03)

    spectral_values = spectral_function_fermion(omegas, paras)
    plt.plot(omegas, spectral_values)
    plt.xlabel('$\\omega$')
    plt.ylabel('$Gpm.imag$')
    plt.title('$G_{+-}(\\omega)$')
    plt.ylim([0, 0.45])
    plt.xlim([-25, 25])
    plt.grid()
    plt.show()
