
import numpy as np
from matplotlib import pyplot as plt
from mpl_setup import BLUE, ORANGE, GREEN, PURPLE, GRAY, SKY_BLUE, VERMILLION
from qutip import *
from qutip.solver.heom import (
    HEOMSolver,
)
from environment import BosonicEnvironment
colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY, SKY_BLUE, VERMILLION]
eps = 0.1  # small system energy

Hsys = 0.5 * eps * sigmax()


def plot_result_expectations(result, color, label, m_op, axes=None, factor=1,
                             linestyle="solid"):
    """ Plot the expectation values of operators as functions of time.

        Each plot in plots consists of: (solver_result, measurement_operation,
        color, label).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
        fig_created = True
    else:
        fig = None
        fig_created = False

    # add kw arguments to each plot if missing
    exp = np.real(expect(result.states, m_op))
    axes.plot(np.array(result.times)*factor, exp, color=color, label=label,
              linestyle=linestyle)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig


# ALPHA
alphas = [0.1, 0.4, 0.6, 0.9, 1.2]
NCs = [8, 8, 8, 13, 13]

wc = 1

t = np.linspace(0, 80, 1000)
T = 0


def J(w, alpha=1):
    """ The Ohmic bath spectral density as a function
    of w (and the bath parameters). """
    return (np.pi/2) * w * alpha * 1 / (1+(w/wc)**2)**2


Q = sigmaz()
Ttot = 300
# Times to solve for:
tlist = np.linspace(0, Ttot, 1000)
rho0 = basis(2, 0) * basis(2, 0).dag()
try:  # Load precomputed models if existing
    models = [qload("transition_"+str(i)) for i in alphas]
except:
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))
    models = []
    for i in range(len(alphas)):
        try:
            models.append(qload("transition_"+str(alphas[i])))
        except:
            env = BosonicEnvironment.from_spectral_density(
                lambda w: J(w, alphas[i]), wMax=50, T=T)
            cfit, fitinfo = env.approx_by_cf_fit(
                tlist=t, Ni_max=1, Nr_max=2, target_rsme=None, full_ansatz=True
            )
            bath = cfit.to_bath(Q)
            print(fitinfo["summary"])
            options = {'nsteps': 15000, 'store_states': True,
                       'rtol': 1e-14, 'atol': 1e-14}
            # Number of levels of the hierarchy to retain:
            NC = NCs[i]  # this is an importance covnergence parameter
            HEOMMats = HEOMSolver(Hsys, [bath], NC, options=options)
            resultMats = HEOMMats.run(rho0, tlist)
            qsave(data=resultMats, name="transition_"+str(alphas[i]))
            models.append(resultMats)


fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))
plt.axhline(y=0, label=r"$\langle \sigma_{z} \rangle$ =0" +
            " localized ", color=colors[len(alphas)], linestyle='--')
linestyles=['solid',"dashed","-.","dotted","solid"]

for k, i in enumerate(models):
    plot_result_expectations(
        i, colors[k], rf"$\\alpha={alphas[k]} \quad N_c={NCs[k]}$",
        sigmaz(), axes=axes, factor=eps,linestyle=linestyles[k]
    )
# plt.xlim(0,1)
# axes.locator_param
axes.legend(loc=0, fontsize=18)
axes.xaxis.set_tick_params(labelsize=20)
axes.yaxis.set_tick_params(labelsize=20)
plt.ylabel(r"$\langle \sigma_{z}(t) \rangle$", fontsize=20)
plt.xlabel(r"$\Delta t$", fontsize=20)
fig.tight_layout()
plt.savefig('./heom_transition.pdf')
# FOR The main text
def J(w, alpha=1):
    """ The Ohmic bath spectral density as a function
    of w (and the bath parameters). """
    return (np.pi/2) * w * alpha * 1 / (1+(w/wc)**2)**2

env = BosonicEnvironment.from_spectral_density(lambda w: J(w, alpha), wMax=50, T=T)
