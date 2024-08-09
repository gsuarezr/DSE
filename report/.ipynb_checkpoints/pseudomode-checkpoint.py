import matplotlib.pyplot as plt
import numpy as np
from qutip import (basis, expect, mesolve, qeye, destroy,
                   tensor, brmesolve)
from qutip import liouvillian
from nmm import csolve
from qutip.solver.heom import BathExponent
from qutip.solver import heom
from scipy.integrate import quad


class pseudomode:
    def __init__(self, Hsys, Q, bath):
        """
        TODO I usually have problems with the bath exponents class so MAybe I'll just fit inside here and that's it
        Also need to add different levels for each mode
        """
        self.Hsys = Hsys
        self.Q = Q
        self.bath = bath
        self.coefficients()

    def coefficients(self):
        cks = np.zeros(len(self.bath.exponents), dtype=np.complex128)
        vks = np.zeros(len(self.bath.exponents), dtype=np.complex128)
        seen = set()
        i = 0
        for exp in self.bath.exponents:
            if exp.vk not in seen:
                if exp.type == BathExponent.types["R"]:
                    cks[i] = 2*exp.ck
                else:
                    cks[i] = -2j*exp.ck
                vks[i] = -exp.vk

                seen.add(exp.vk.conjugate())
                i = i+1

        self.cks = cks[cks != 0]
        self.vks = vks[vks != 0]

    def tensor_id(self, pos, cutoff=2, op=None):
        temp = [qeye(cutoff)]*(len(self.cks)+1)
        if pos != 0:
            temp[0] = qeye(self.Hsys.shape[0])
            temp[pos] = destroy(cutoff)
        else:
            temp[0] = op
        return tensor(temp)

    def hamiltonian(self, cutoff=2):
        Hsys = self.tensor_id(0, cutoff, self.Hsys)
        Q = self.tensor_id(0, cutoff, self.Q)
        destroys = [self.tensor_id(i+1, cutoff) for i in range(len(self.cks))]
        for i in destroys:
            i.dims = Hsys.dims
        Hpm = sum([np.imag(i)*destroys[k].dag()*destroys[k]
                  for k, i in enumerate(self.vks)])
        Hsys_pm = sum([np.sqrt(i + 0j) * (destroys[k].dag() + destroys[k]) * Q
                       for k, i in enumerate(self.cks)])
        Heff = Hsys_pm+Hpm+Hsys
        return Heff, destroys

    def power_spectrum(self, w):
        S = 0
        for i in range(len(self.cks)):
            S += 2 * np.real(self.cks[i] / (self.vks[i] - 1j*w))
        return S

    def correlation_function(self, t):
        return self.bath.correlation_function(t)

    def prepare(self, cutoff, initial):
        init = [initial]+[basis(cutoff, 0)]*(len(self.cks))
        psi02 = tensor(init)
        psi02 = psi02*psi02.dag()
        return psi02.to("CSR")
    def lindbladian(self,cutoff):
        Heff, d = self.hamiltonian(cutoff)
        c_ops = [np.sqrt(-2*np.real(i) + 0j)*d[k]
                 for k, i in enumerate(self.vks)]
        return liouvillian(H=Heff,c_ops=c_ops)
    def evolution(self, initial, cutoff, t, e_ops=[], options={}):
        Heff, d = self.hamiltonian(cutoff)
        initial = self.prepare(cutoff, initial)
        e_ops = [self.tensor_id(0, cutoff, i) for i in e_ops]
        c_ops = [np.sqrt(-2*np.real(i) + 0j)*d[k]
                 for k, i in enumerate(self.vks)]
        return mesolve(
            Heff, initial, t, c_ops=c_ops, e_ops=e_ops, options=options)


class zero_temp_bath(heom.CorrelationFitter):
    def __init__(
            self, Q, time, lam, gamma, w0, N=2, lower=None, upper=None,
            sigma=None, guesses=None):
        Gamma = gamma/2
        Omega = np.sqrt(w0**2 - Gamma**2)
        C = np.array([self._matsubara_zero_integrand(
            i, lam, gamma, w0) for i in time])
        self.correlation_function = C+self.C0(time, lam, gamma, w0)
        super().__init__(Q, 0, time, C)
        self.fbath, self.finfo = self.get_fit(
            Ni=1, Nr=N, lower=lower, upper=upper, sigma=sigma, guesses=guesses)
        ck = (lam**2)/(2*Omega)
        vk = (-Gamma+1j*Omega)
        C0_contrib = [
            BathExponent(
                type="R", dim=None, Q=Q, ck=ck / 2, vk=-vk),
            BathExponent(
                type="R", dim=None, Q=Q, ck=ck / 2, vk=-np.conjugate(vk))]
        self.fbath.exponents.extend(C0_contrib)
        self.exponents = self.fbath.exponents
        self.bath = self.fbath

    def _matsubara_zero_integrand(
            self, t, coup_strength, bath_broad, bath_freq):
        """
        Integral for the zero temperature Matsubara exponentials.
        """
        lam = coup_strength
        gamma = bath_broad
        w0 = bath_freq

        omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2 + 0j)
        a = omega + 1j * gamma/2
        aa = np.conjugate(a)

        prefactor = -(lam ** 2 * gamma) / np.pi
        def integrand(x): return prefactor * ((x * np.exp(-x * t)
                                               ) / ((a ** 2 + x ** 2) * (aa ** 2 + x ** 2)))
        return quad(integrand, 0, np.inf, limit=5000, complex_func=True)[0]

    def C0(self, t, coupling, gamma, w0, beta=np.inf):
        Gamma = gamma/2
        Omega = np.sqrt(w0**2 - Gamma**2 + 0j)
        if beta != np.inf:
            tempc0r = (1/np.tanh(beta*(Omega + Gamma*1j)/2))*np.exp(1j*Omega*t)
        else:
            # Tookl the limit analytically because there seems to be some issues with numpy
            tempc0r = np.exp(1j*Omega*t)
        c0r = tempc0r+np.conjugate(tempc0r)
        c0i = -2j*np.sin(Omega*t)
        return (coupling**2)/(2*Omega)*np.exp(-Gamma*t)*np.exp(-1j*Omega*t)


def rotation(data, H, t):
    rotated = [
        (-1j * H * t[i]).expm()
        * data[i]
        * (1j * H * t[i]).expm()
        for i in range(len(t))
    ]
    return rotated
