import sys
sys.path.append("../src/")

import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity
import numpy as np
from matplotlib import pyplot as plt  
from two_atoms_src import get_true_dynamics_2_atom, examine_Trotter

iis=np.linspace(2,6,3)
mode="0"
mode2="L"

g=(0.5*1e-3)*2*np.pi
Gamma_max=8*g
v0=0.2*2*np.pi
omega_base=1.6*2*np.pi
  
sigma_m=(sigmax()-1j*sigmay())/2
sigma_p=(sigmax()+1j*sigmay())/2

psi0=tensor([basis(2, 1),basis(2, 0)])
  
t_tot=3000
n_rabi=101
times=np.linspace(0.0, t_tot, n_rabi) 

n1=tensor(sigmaz(),qeye(2))
n2=tensor(qeye(2),sigmaz())

for i in range(3):
  ii=iis[i]
  gamma_1=ii*g
  result=get_true_dynamics_2_atom(g,gamma_1,times,psi0,n1,n2,mode2)
  result_n1=(result[0]+1)/2
  result_n2=(result[1]+1)/2
  data=np.array([times,result_n1,result_n2])

  if mode2=="L":  
    np.savetxt(str("evolution_L_gamma="+str(ii)+"g.OUT"),np.transpose(data))
  else:
    np.savetxt(str("evolution_H_gamma="+str(ii)+"g.OUT"),np.transpose(data))

