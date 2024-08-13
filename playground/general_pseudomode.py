
class pseudomode:
    def _init__(self,Hsys,Q,cutoff=2):
        self.Hsys=Hsys
        self.Q=Q
        self.cutoff=cutoff
def tensor_id(op,pos,cutoff=cutoff):
    if pos==0:
        return tensor([op,qeye(cutoff),qeye(cutoff),qeye(cutoff)])
    if pos==1:
        return tensor([qeye(H.shape[0]),op,qeye(cutoff),qeye(cutoff)])
    if pos==2:
        return tensor([qeye(H.shape[0]),qeye(cutoff),op,qeye(cutoff)])
    else:
        return tensor([qeye(H.shape[0]),qeye(cutoff),qeye(cutoff),op])

Gamma= gamma/2
Omega= np.sqrt(w0**2 -Gamma**2)
Hsys=tensor_id(H,0)
Qeff=tensor_id(Q,0)
a=tensor_id(destroy(cutoff),1)
a.dims=Qeff.dims
b=tensor_id(destroy(cutoff),2)
b.dims=Qeff.dims
c=tensor_id(destroy(cutoff),3)
c.dims=Qeff.dims
Hpm= Omega*a.dag()*a+finfo['params_real'][2][0]*b.dag()*b+finfo['params_real'][2][1]*c.dag()*c
Hsys_pm=np.sqrt(((lam**2)/(2*Omega)))*Qeff*(a+a.dag())+1j*np.sqrt(-finfo['params_real'][0][0])*Qeff*(b+b.dag())+1j*np.sqrt(-finfo['params_real'][0][1]+0j)*Qeff*(c+c.dag())
Heff=Hsys+Hsys_pm+Hpm