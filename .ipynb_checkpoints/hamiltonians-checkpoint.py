from qutip import (basis, tensor, sigmax, sigmay, sigmaz, qeye, Qobj)
import numpy as np
import itertools
from collections import deque

np.random.seed(42)


def ising(N, g, Jx, Jy=0, Jz=0):
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian - Energy splitting terms
    H = 0
    for i in range(N):
        H += g * sz_list[i]

    for n in range(N - 1):
        H += -Jx * sx_list[n] * sx_list[n + 1]
        H += -Jy * sy_list[n] * sy_list[n + 1]
        H += -Jz * sz_list[n] * sz_list[n + 1]
    return H, sx_list, sy_list, sz_list


def schwinger(N, g, a, theta, J, m=0):

    # Set the system parameters
    J = 1/4*a
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian - Energy splitting terms
    H = 0
    for i in range(N):
        H += (m/2) * (-1)**i * sz_list[i]

    for n in range(N - 1):
        H += J * sx_list[n] * sx_list[n + 1]
        H += J * sy_list[n] * sy_list[n + 1]

    # Electromagnetic coupling term
    for n in range(1, N):
        temp = 0
        for i in range(1, n):
            temp += (sz_list[i] + (-1)**i)/2 + theta/(2*np.pi)
        H += (temp**2)*8*J * g**2
    return H


def basis_syk(N):
    if (N == 1):
        prev = [sigmay()/np.sqrt(2), sigmax()/np.sqrt(2)]
        return prev
    else:
        prev = basis_syk(N-1)
    ops = []
    sn = 1/np.sqrt(2) * tensor(qeye(2**(N-1)), sigmay())
    snm1 = 1/np.sqrt(2) * tensor(qeye(2**(N-1)), sigmax())
    if N > 1:
        ops.append(sn)
        ops.append(snm1)
        for i in prev:
            ops.append(tensor(i, sigmaz()))
    else:
        prev.extend(ops)
    for i in ops:
        i.dims = ops[0].dims
    return ops


def syk_full(N):
    psis = basis_syk(N)
    k = 2*N
    H = 0
    std = np.sqrt(6)*2/(N)**1.5
    J = std*np.random.rand(2*N, 2*N, 2*N, 2*N)
    for i in range(0, k):
        for j in range(i+1, k):
            for l in range(j+1, k):
                for m in range(l+1, k):
                    H += J[i, j, l, m]*psis[i]*psis[j]*psis[l]*psis[m]
    return H


def genMajorana(N_val, nrm_cff):
    n_dirac = int(N_val/2)  # Number of Dirac fermions
    n_Level = int(2**n_dirac)  # Number of energy levels

    # Pauli matrices & identity
    p1 = sigmax()
    p2 = sigmay()
    p3 = sigmaz()
    id2 = qeye(2)

    xi_Lst = []  # List to store Majorana operators
    xi1_Lst = []  # List to store p1 operators
    xi2_Lst = []  # List to store p2 operators

    xi1_Lst.append(p1)
    xi2_Lst.append(p2)

    for i in range(n_dirac-1):
        xi1_Lst.append(id2)
        xi2_Lst.append(id2)

    # Generate Majorana operators by tensor product
    xi_Lst.append(multTensor(xi1_Lst, n_Level, nrm_cff))
    xi_Lst.append(multTensor(xi2_Lst, n_Level, nrm_cff))

    xi1_deq = deque(xi1_Lst)
    xi2_deq = deque(xi2_Lst)

    tms = int((N_val-2)/2)  # Number of times to rotate the deque
    for j in range(tms):
        xi1_deq.rotate(1)
        xi1_deq[0] = p3
        xi_Lst.append(multTensor(list(xi1_deq), n_Level, nrm_cff))
        xi2_deq.rotate(1)
        xi2_deq[0] = p3
        xi_Lst.append(multTensor(list(xi2_deq), n_Level, nrm_cff))

    return xi_Lst


# Generate a tensor product of matrices
def multTensor(mat_List, n_Level, norm_cff):
    prod = mat_List[0]
    m = 1
    while m < len(mat_List):
        prod = tensor(prod, mat_List[m])
        m += 1

    prod = norm_cff * prod
    prod = prod.full()
    prod = prod.reshape((n_Level, n_Level))

    return Qobj(prod)


# Print the parameters
def print_parms(N_val, _no_coupls):
    print(f"N: {N_val}")
    print(f"# of non-zero couplings: {_no_coupls}")
    print()


# Generate the Hamiltonian of the binary-coupling model
def syk_binary(N_val, no_plus_coupls, no_minus_coupls, chi_List):

    N_list = []
    for i in range(N_val):
        N_list.append(i)

    print_parms(N_val, no_plus_coupls+no_minus_coupls)

    H_p_m_1 = 0
    combs_list = list(itertools.combinations(N_list, 4))
    plus_coupls_combs = np.random.sample(combs_list, no_plus_coupls)
    for c in range(no_plus_coupls):
        combs_list.remove(plus_coupls_combs[c])

    minus_coupls_combs = np.random.sample(combs_list, no_minus_coupls)
    for i in range(no_plus_coupls):
        H_p_m_1 += chi_List[plus_coupls_combs[i][0]]*chi_List[plus_coupls_combs[i]
                                                              [1]] * chi_List[plus_coupls_combs[i][2]]*chi_List[plus_coupls_combs[i][3]]

    for ii in range(no_minus_coupls):
        H_p_m_1 -= chi_List[minus_coupls_combs[ii][0]]*chi_List[minus_coupls_combs[ii][
            1]] * chi_List[minus_coupls_combs[ii][2]]*chi_List[minus_coupls_combs[ii][3]]

    # Overall normalization factor of the Hamiltonian
    C_Np = 1

    return C_Np*H_p_m_1
