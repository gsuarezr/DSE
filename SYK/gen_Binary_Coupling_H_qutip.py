from qutip import *
from collections import deque
import itertools
import math
import random
import time

# Generate a list of Majorana fermion operators
def genMajorana(N_val, nrm_cff):
    n_dirac = int(N_val/2)  # Number of Dirac fermions
    n_Level = int(2**n_dirac)  # Number of energy levels

    # Pauli matrices & identity
    p1 = sigmax(); p2 = sigmay()
    p3 = sigmaz(); id2 = qeye(2)

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
        xi_Lst.append(multTensor(list(xi1_deq),n_Level,nrm_cff))
        xi2_deq.rotate(1)
        xi2_deq[0] = p3
        xi_Lst.append(multTensor(list(xi2_deq),n_Level,nrm_cff))

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
def gen_H_pm1(N_val, no_plus_coupls, no_minus_coupls, chi_List):
    random.seed(time.time())

    N_list = []
    for i in range(N_val):
        N_list.append(i)

    print_parms(N_val, no_plus_coupls+no_minus_coupls)

    H_p_m_1 = 0
    combs_list = list(itertools.combinations(N_list, 4))
    plus_coupls_combs = random.sample(combs_list, no_plus_coupls)
    for c in range(no_plus_coupls):
        combs_list.remove(plus_coupls_combs[c])

    minus_coupls_combs = random.sample(combs_list, no_minus_coupls)
    for i in range(no_plus_coupls):
        H_p_m_1 += chi_List[plus_coupls_combs[i][0]]*chi_List[plus_coupls_combs[i][1]]*\
                   chi_List[plus_coupls_combs[i][2]]*chi_List[plus_coupls_combs[i][3]]

    for ii in range(no_minus_coupls):
        H_p_m_1 -= chi_List[minus_coupls_combs[ii][0]]*chi_List[minus_coupls_combs[ii][1]]*\
                   chi_List[minus_coupls_combs[ii][2]]*chi_List[minus_coupls_combs[ii][3]]

    # Overall normalization factor of the Hamiltonian
    C_Np = 1

    return C_Np*H_p_m_1


#################################################################################################

# Number of Majorana fermions
N = 8

# Number of +1 couplings
no_plus_coupls = 5
# Number of -1 couplings
no_minus_coupls = 5

# Normalization of the Majorana fermions 
norm_cff = 1/math.sqrt(2)
# Generate the Majorana operators
chi_List = genMajorana(N, norm_cff)

# Generate the binary-coupling model Hamiltonian
H_pm1 = gen_H_pm1(N, no_plus_coupls, no_minus_coupls, chi_List)

print(H_pm1)  # Hamiltonian of the binary-coupling model
