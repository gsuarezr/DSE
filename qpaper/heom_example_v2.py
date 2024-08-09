# ---IMPORT SECTION---
from mpl_setup import BLUE,ORANGE,GREEN,PURPLE,GRAY,SKY_BLUE,VERMILLION
from qutip import Qobj,sigmaz,sigmax,brmesolve,sigmay
from qutip import sigmam,sigmap,mesolve,expect,basis
import numpy as np
from qutip.solver.heom import SpectralFitter,CorrelationFitter
from qutip.solver.heom import HEOMSolver,UnderDampedBath
import matplotlib.pyplot as plt
import mpmath as mp
from qutip.solver.heom import OhmicBath

colors=[BLUE,ORANGE,GREEN,PURPLE,GRAY,SKY_BLUE,VERMILLION]
P11p = basis(2, 0) * basis(2, 0).dag()
P12p = basis(2, 0) * basis(2, 1).dag()

# -- FIRST EXAMPLE PARAMETERS --- 

lam = 0.5
w0 = 2
gamma = 3 
T = 0.5
t = np.linspace(0,40,1000)
rho0 = 0.5 * Qobj([[1,1],[1,1]])
Hsys = sigmax()/2 +sigmaz()/4
Q = sigmaz()
bath = UnderDampedBath(Q=Q, lam=lam, gamma=gamma,T=T, Nk=5,w0=w0)


# -- VISUALIZATION OF CORRELATION FUNCTION AND EXPONENTS ---
t_viz = np.linspace(0,4,100)
C=bath.correlation_function(t_viz)
C_approx=bath.correlation_function_approx(t_viz)
fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.54))
ax[0].plot(t_viz,np.imag(C),color=colors[0],label='Full')
lines = ["--","-.",":"]
c=0
for i in [1]:
    bath = UnderDampedBath(Q=Q, lam=lam, gamma=gamma, T=T, Nk=i,w0=w0)
    ax[0].plot(t_viz,np.imag(bath.correlation_function_approx(t_viz)),lines[c],
               color=colors[c+1],label=f'k={i}')
    c+=1
ax[0].legend()

ax[1].plot(t_viz,np.real(C),label='Full')
lines = ["--","-.",":"]
c=0
for i in [1,2,5]:
    bath = UnderDampedBath(Q=Q, lam=lam, gamma=gamma, T=T, Nk=i,w0=w0)
    ax[1].plot(t_viz,np.real(bath.correlation_function_approx(t_viz)),lines[c],
               label=f'k={i}')
    c+=1
ax[1].legend()
ax[0].set_ylabel(r"$C_{I}(t)$")
ax[1].set_ylabel(r"$C_{R}(t)$")
ax[0].set_xlabel(r"$t$")
ax[1].set_xlabel(r"$t$")
plt.savefig('./heom_corr_func.pdf')

# -- SIMULATION --

# -- HEOM --
solver = HEOMSolver(Hsys, bath, max_depth=6)
result_h = solver.run(rho0, t)
# APPROPIATE JUMP OPS IN THIS CASE
sp=Qobj([[0.4,0.2472],[-0.6472,-0.4]])
sm=sp.dag()
sz=Qobj([[0.2,0.4],[0.4,-0.2]])
print(bath.power_spectrum(0))
# -- LINDBLAD --
c_ops=[np.sqrt(bath.power_spectrum(1.118))*sp,np.sqrt(bath.power_spectrum(-1.118))*sm,np.sqrt(bath.power_spectrum(0))*sz]
result_lindblad = mesolve(Hsys, rho0, t, c_ops)

# -- BLOCH-REDFIELD --
a_ops=[[Q, bath.power_spectrum]]
resultBR = brmesolve(Hsys, rho0, t, a_ops=a_ops,sec_cutoff=-1)



# -- DYNAMICS SIMULATION --
fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.54))
ax[0].plot(lam*t,expect(P11p,result_h.states),color=colors[0],label='HEOMSolver')
ax[0].plot(lam*t,expect(P11p,resultBR.states),color=colors[2],label='brmesolve')
ax[0].plot(lam*t,expect(P11p,result_lindblad.states),'-.',color=colors[1],label='mesolve')
ax[0].set_ylabel(r"$\rho_{11}$")
ax[0].set_xlabel(r"$\lambda t$")
ax[0].legend()
ax[1].plot(lam*t,expect(sigmap(),result_h.states),color=colors[0],label='HEOMSolver')
ax[1].plot(lam*t,expect(sigmap(),result_lindblad.states),'-.',color=colors[1],label='mesolve')
ax[1].plot(lam*t,expect(sigmap(),resultBR.states),'x',color=colors[2],label='brmesolve',markevery=10)

ax[1].set_ylabel(r"$Re(\rho_{12})$")
ax[1].set_xlabel(r"$\lambda t$")
ax[1].legend()
plt.savefig('./heom_qubit_underdamped.pdf')
# -- SECOND EXAMPLE PARAMETERS --
lam=0.1
gamma=5
T=1

# -- AUXILIARY FUNCTIONS -- 
def J(w, lam=lam,gamma=gamma):
    """ 
    Ohmic spectral density
    """
    return lam*w*np.exp(-abs(w)/gamma)
def ohmic_correlation(t, lam=lam, wc=gamma, beta=1/(T), s=1):
    """ The Ohmic bath correlation function as a function of t
        (and the bath parameters).
    """
    corr = (
        (1 / np.pi) * lam * wc**(1 - s) * beta**(-(s + 1)) * mp.gamma(s + 1)
    )
    z1_u = (1 + beta * wc - 1.0j * wc * t) / (beta * wc)
    z2_u = (1 + 1.0j * wc * t) / (beta * wc)
    # Note: the arguments to zeta should be in as high precision as possible.
    # See http://mpmath.org/doc/current/basics.html#providing-correct-input
    return np.array([
        complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
        for u1, u2 in zip(z1_u, z2_u)
    ], dtype=np.complex128)

# --FITTING--
w = np.linspace(0, 100, 2000)
fs=SpectralFitter(T,Q,w,J)
bath_fs,_=fs.get_fit(N=8,Nk=5)

t=np.linspace(0,10,1000)
fc=CorrelationFitter(Q=Q,T=T,t=t,C=ohmic_correlation)
bath_fc,_=fc.get_fit(Ni=5,Nr=4)

fo=OhmicBath(Q=Q,T=T,alpha=lam, wc=gamma,s=1)

bath_fos,_=fo.make_spectral_fit(t,N=8,Nk=5)
bath_foc,_=fo.make_correlation_fit(w,Ni=5,Nr=4)
# -- DEPENDENCE ON NK --
figfit, axfit = plt.subplots(1, 2, figsize=(13.6, 4.54))
full=bath_fs.spectral_density(w)
axfit[0].plot(w,full,color=colors[0],label="Original")
j=0
markers=["-.","--","dotted"]
for i in [1,2,5]:
    bath_fs,fitinfo=fs.get_fit(N=8,Nk=i)
    axfit[0].plot(w,bath_fs.spectral_density_approx(w),linestyle=markers[j],color=colors[j],label=f"k={i}")
    axfit[1].plot(w,np.abs(full-bath_fs.spectral_density_approx(w)),'-.',color=colors[j],label=f"k={i}")
    j+=1

axfit[0].legend()
axfit[1].legend()
axfit[0].set_ylabel(r"$J(\omega)$")
axfit[0].set_xlabel(r"$\omega$")
axfit[1].set_ylabel(r"$|J(\omega)-J_{approx}(\omega)|$")
axfit[1].set_xlabel(r"$\omega$")
plt.savefig('./heom_spec_k.pdf')
print("done")


# -- SOLVING DYNAMICS --
tlist = np.linspace(0, 10 , 1000)
HEOM_corr_fit = HEOMSolver(Hsys, bath_fc, max_depth=5)
result_corr=HEOM_corr_fit.run(rho0,tlist)

HEOM_spec_fit = HEOMSolver(Hsys, bath_fs, max_depth=5)
result_spec=HEOM_spec_fit.run(rho0,tlist)



HEOM_fos = HEOMSolver(Hsys, bath_fos, 
                      max_depth=5)
result_fos=HEOM_fos.run(rho0,tlist)

HEOM_foc = HEOMSolver(Hsys, bath_foc, 
                      max_depth=5)
result_foc=HEOM_foc.run(rho0,tlist)

# -- OHMIC BATH DYNAMICS -- 
fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.54))
ax[0].plot(lam*t,expect(P11p,result_corr.states),color=colors[0],label='CorrelationFitter')
ax[0].plot(lam*t,expect(P11p,result_spec.states),"-.",color=colors[1],label='SpectralFitter')
ax[0].plot(lam*t,expect(P11p,result_fos.states),'o',color=colors[2],label='OhmicBath-Spectral',markevery=15)
ax[0].plot(lam*t,expect(P11p,result_foc.states),'x',color=colors[4],label='OhmicBath-Correlation',markevery=10)
ax[0].set_ylabel(r"$\rho_{11}$")
ax[0].set_xlabel(r"$\lambda t$")
ax[0].legend()
ax[1].plot(lam*t,expect(P12p,result_corr.states),color=colors[0],label='CorrelationFitter')
ax[1].plot(lam*t,expect(P12p,result_spec.states),"-.",color=colors[1],label='SpectralFitter')
ax[1].plot(lam*t,expect(P12p,result_fos.states),'o',color=colors[2],label='OhmicBath-Spectral',markevery=15)
ax[1].plot(lam*t,expect(P12p,result_foc.states),'x',color=colors[4],label='OhmicBath-Correlation',markevery=10)
ax[1].set_ylabel(r"$Re(\rho_{12})$")
ax[1].set_xlabel(r"$\lambda t$")
ax[1].legend()
plt.savefig('./heom_qubit_ohmic.pdf')


