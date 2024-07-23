import numpy as np

import qutip as qt
from qutip.solver.heom import HEOMSolver, BosonicBath, CorrelationFitter, UnderDampedBath

import matplotlib.pyplot as plt
from mpl_setup import *


############ Normal driven qubit, time-dependent example 1 for paper ############


# hamiltonian parameters
Delta = 2 * np.pi  # qubit splitting
omega_d = Delta  # drive frequency
A = 0.01 * Delta  # drive amplitude

# driving field


def f(t):
    return np.sin(omega_d * t)


H0 = Delta / 2.0 * qt.sigmaz()
H1 = [A / 2.0 * qt.sigmax(), f]
H = [H0, H1]

# bath parameters
gamma = 0.005 * Delta / (2 * np.pi)  # dissipation strength
temp = 0  # temperature

# simulation parameters
psi0 = qt.basis(2, 0)  # initial state
e_ops = [qt.sigmaz()]

T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 1000 * T, 400)

# --- HEOM ---

wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi

gamma_heom = 1.9 * w0
Gamma = gamma_heom/2
Omega = np.sqrt(w0**2 - Gamma**2)
lambd = np.sqrt(
    0.5 * gamma *
    ((w0 ** 2 - wsamp ** 2) ** 2 + (gamma_heom ** 2) * ((wsamp) ** 2)) /
    (gamma_heom * wsamp))

bath = UnderDampedBath(Q=qt.sigmax(), lam=lambd, w0=w0,
                       gamma=gamma_heom, T=0, Nk=5)
times2 = np.linspace(0, 15, 1000)
cfiitter2 = CorrelationFitter(
    qt.sigmax(),
    0, times2, bath.correlation_function)
fit2 = cfiitter2.get_fit(Ni=1, Nr=2)
print(fit2[1]['summary'])


max_depth = 5

HEOM_corr_fit = HEOMSolver(
    qt.QobjEvo(H), fit2[0], max_depth=max_depth
)


results_corr_fit = (HEOM_corr_fit.run(psi0*psi0.dag(), tlist, e_ops=e_ops))


# --- brmesolve ---
def nth(w):
    if temp > 0:
        return 1 / (np.exp(w / temp) - 1)
    else:
        return 0

# power spectrum


def noise_spectrum_br(w):
    if w > 0:
        return gamma * (nth(w) + 1)
    elif w == 0:
        return 0
    else:
        return gamma * nth(-w)


a_ops = [[qt.sigmax(), noise_spectrum_br]]

brme_result = qt.brmesolve(
    H, psi0, tlist, a_ops=a_ops, e_ops=e_ops, sec_cutoff=-1)


# --- mesolve ---

# c_ops_me = [np.sqrt(gamma0*(nth(delta)+1))*qt.destroy(2).dag(),np.sqrt(gamma0*(nth(delta)))*qt.destroy(2)]
c_ops_me = [np.sqrt(gamma) * qt.sigmam()]
me_result = qt.mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)

c_ops_me_RWA = [np.sqrt(gamma) * qt.sigmam()]
H_RWA = (Delta - omega_d) * 0.5 * qt.sigmaz() + A / 4 * qt.sigmax()
me_result_RWA = qt.mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)


plt.figure()

plt.plot(tlist, me_result.expect[0], '-', label=r'mesolve (time-dep)')
plt.plot(tlist, me_result_RWA.expect[0], '-.', label=r'mesolve (rwa)')
plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
plt.plot(tlist, brme_result.expect[0], ':', linewidth=6, label=r'brmesolve')

plt.xlabel(r'$t\, /\, \Delta^{-1}$', fontsize=18)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=18)
plt.legend()
plt.text(200, 0.7, "(a)", fontsize=18)

plt.savefig("mesolve_driven_1.pdf")
plt.show()


############ Frequency modulation, Example 2 ############


# hamiltonian parameters
omega_d = 0.05 * Delta  # drive frequency
A = Delta  # drive amplitude

# driving field


def f(t):
    return np.sin(omega_d * t)


H0 = [A / 2.0 * qt.sigmaz(), f]
H = [H0]

# bath parameters
gamma = 0.05 * Delta / (2 * np.pi)  # dissipation strength

# simulation parameters
psi0 = qt.basis(2, 0)  # initial state
e_ops = [qt.sigmaz()]

T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 2 * T, 400)

# --- HEOM ---

wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi

gamma_heom = 1.9 * w0
# gamma = 1.5 * w0


lambd = np.sqrt(
    0.5 * gamma *
    ((w0 ** 2 - wsamp ** 2) ** 2 + (gamma_heom ** 2) * ((wsamp) ** 2)) /
    (gamma_heom * wsamp))

bath = UnderDampedBath(Q=qt.sigmax(), lam=lambd, w0=w0,
                       gamma=gamma_heom, T=1e-30, Nk=5)
times2 = np.linspace(0, 15, 2000)
cfiitter2 = CorrelationFitter(
    qt.sigmax(),
    0, times2, bath.correlation_function)
fit2 = cfiitter2.get_fit(Ni=1, Nr=2)
print(fit2[1]['summary'])

HEOM_corr_fit = HEOMSolver(
    qt.QobjEvo(H),
    fit2[0],
    max_depth=max_depth, options=qt.Options(
        nsteps=15000, store_states=True, rtol=1e-12, atol=1e-12))


results_corr_fit = (HEOM_corr_fit.run(psi0*psi0.dag(), tlist, e_ops=e_ops))


# --- brmesolve ---


a_ops = [[qt.sigmax(), bath.power_spectrum]]

brme_result = qt.brmesolve(
    H, psi0, tlist, a_ops=a_ops, e_ops=e_ops, sec_cutoff=-1)
# bose einstein distribution


a_ops = [[qt.sigmax(), noise_spectrum_br]]

brme_result2 = qt.brmesolve(
    H, psi0, tlist, a_ops=a_ops, e_ops=e_ops, sec_cutoff=-1)

# --- mesolve ---

# c_ops_me = [np.sqrt(gamma0*(nth(delta)+1))*qt.destroy(2).dag(),np.sqrt(gamma0*(nth(delta)))*qt.destroy(2)]
c_ops_me = [np.sqrt(gamma) * qt.sigmam()]
me_result = qt.mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)

c_ops_me_RWA = [np.sqrt(gamma) * qt.sigmam()]
H_RWA = (Delta - omega_d) * 0.5 * qt.sigmaz() + A / 4 * qt.sigmax()
me_result_RWA = qt.mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)


plt.figure()

plt.plot(tlist, me_result.expect[0], '-', label=r'mesolve')
# plt.plot(tlist, me_result_RWA.expect[0], '-.', label=r'mesolve (rwa)')
plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
plt.plot(
    tlist, brme_result.expect[0],
    ':', linewidth=6, label=r'brmesolve non-flat')
plt.plot(tlist, brme_result2.expect[0], ':', linewidth=6, label=r'brmesolve')

plt.xlabel(r'$t\, /\, \Delta^{-1}$', fontsize=18)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=18)
plt.legend()
plt.text(8, 0.7, "(b)", fontsize=18)

plt.savefig("mesolve_driven_2.pdf")
plt.show()


exit()
# --- Alternative ways of specifying time dependence ---

f = f"sin({omega_d} * t)"

f = np.sin(omega_d * tlist)
