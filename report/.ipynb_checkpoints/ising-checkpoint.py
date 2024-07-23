#!/usr/bin/env python
# coding: utf-8

# ---
# title: Reproducing results from the DSE paper
# ---
# 
# This is just the same example you already had in the paper, The main issue/delay in obtaining it was the normalized_ouput parameter in mesolve which should have been set to false.
# 
# In order to run the notebook unfortunately both my version of [qutip]() with the fitting stuff and this poorly done package for the [cumulant equation](https://github.com/gsuarezr/NonMarkovianMethods/tree/multiplebaths) needs to be installed  

# In[1]:
import sys
import matplotlib.pyplot as plt
import numpy as np
from qutip import ( basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz,destroy,
                   tensor,fidelity,tracedist,brmesolve,Qobj)
from qutip.solver import heom
from scipy.integrate import quad
from pseudomode import pseudomode,zero_temp_bath,rotation
from hamiltonians import ising
from nmm import csolve


if __name__ == "__main__":
    N = int(sys.argv[1])
    Jx = float(sys.argv[2])
    g = float(sys.argv[3])
    
    H,sx,sy,sz=ising(N=N,g=g,Jx=Jx)
    print(N,Jx,g)
    
    # In[3]:
    
    
    Q=sx[-1]+ 1.1*sy[-1]+0.9*sz[-1]
    
    
    # bath parameters
    
    # In[4]:
    
    
    E01=1#H.eigenenergies()[2]-H.eigenenergies()[0]# Raise this question about the paper 
    gamma=3.8*g
    w0=12*E01
    Gamma=gamma/2
    Omega=np.sqrt(w0**2 -Gamma**2)
    lam=1.15*np.sqrt(Omega)
    
    
    # In[5]:
    
    
    state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
    psi0 = tensor(state_list)
    rho0=psi0*psi0.dag()
    times=np.linspace(0,50,500)
    tfit=np.linspace(0, 25, 5000)
    
    
    #  Using HEOM
    
    # In[6]:
    
    
    bath = heom.UnderDampedBath(
            Q=Q,
            lam=lam, gamma=gamma, w0=w0, T=0, Nk=5) # fix runtime warning
    cfiitter2 = heom.CorrelationFitter(
        Q, 0, tfit, bath.correlation_function)
    bath1, fit2info = cfiitter2.get_fit(Ni=1, Nr=2)
    # notice one mode is also a pretty good approximation
    print(fit2info['summary'])
    
    
    # In[ ]:

    
    
    # In[ ]:
    
    
    solver = heom.HEOMSolver(H,
                              [bath1], max_depth=5, options={"atol": 1e-14})
    result = solver.run(rho0, times)
    
    
    # In[ ]:
    
    
    cum = csolve(
        Hsys=H, t=times, baths=[bath],
        Qs=[Q],
        eps=2, cython=False)
    result_cum = cum.evolution(rho0)
    
    
    result_cum = rotation(result_cum, H, times)
    
    
    # In[ ]:
    
    
    a_ops = [[Q, bath.power_spectrum]]
    resultBR = brmesolve(H, rho0, times, a_ops=a_ops, options={
        "atol": 1e-14}, sec_cutoff=-1)
    
    a_ops = [[Q, bath.power_spectrum]]
    resultBR2 = brmesolve(H, rho0, times, a_ops=a_ops, options={
        "atol": 1e-14})
    
    
    # In[ ]:
    
    
    Ncutoff=3
    modes=2
    bathu = zero_temp_bath(Q, tfit, lam, gamma, w0, N=modes)
    example = pseudomode(Hsys=H, Q=Q, bath=bathu)
    print(bathu.finfo["summary"])

    
    # In[ ]:
    
    
    ans = example.evolution(rho0, Ncutoff, times, e_ops=[H], options={
                            "atol": 1e-14, "normalize_output": False, "store_states": True})
    ans = [i.ptrace(range(N))for i in ans.states]
    
    
    # In[ ]:
    
    
    from qutip import qload
    
    
    # In[ ]:
    
    
    results=[result,result_cum,resultBR,resultBR2,ans]
    
    
    # In[ ]:
    
    
    from qutip import qsave,qload
    
    
    # In[ ]:
    
    
    qsave(results,f"N={N}_ising_{lam}_nocheating_goodq")




