import sys
sys.path.append("../src/")

import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity
import numpy as np
from matplotlib import pyplot as plt  
from two_atoms_src import get_true_dynamics_2_atom, examine_Trotter

ii=6
mode="0"
mode2="L"

g=(0.5*1e-3)*2*np.pi
Gamma_max=8*g
v0=0.2*2*np.pi
omega_base=1.6*2*np.pi
gamma_1=ii*g

sigma_m=(sigmax()-1j*sigmay())/2
sigma_p=(sigmax()+1j*sigmay())/2

psi0=tensor([basis(2, 1),basis(2, 0)])
  
t_tot=5000
n_rabi=101
times=np.linspace(0.0, t_tot, n_rabi) 

#ls=np.linspace(2,50,25)
#ls=range(1,31)
ls=[50]
n_steps=np.shape(ls)[0]
#print(n_steps)
  
N=n_rabi*n_steps
data_times=np.zeros(N)
data_ls=np.zeros(N)
data_n1=np.zeros(N)
data_n2=np.zeros(N)
result_n2s=np.zeros(N)

n1=tensor(sigmaz(),qeye(2))
n2=tensor(qeye(2),sigmaz())

true_result=get_true_dynamics_2_atom(g,gamma_1,times,psi0,n1,n2,mode2)
true_result_n1=(true_result[0]+1)/2
true_result_n2=(true_result[1]+1)/2
  
for i in range(n_steps):
  l=int(ls[i])
  result=examine_Trotter(l,g,gamma_1,times,psi0,n1,n2,Gamma_max,v0,omega_base,mode,mode2)
  result_n1=(result[0]+1)/2
  result_n2=(result[1]+1)/2
  for j in range(n_rabi):
    data_times[n_rabi*i+j]=times[j]
    data_ls[n_rabi*i+j]=l
    data_n1[n_rabi*i+j]=(result_n1[j]-true_result_n1[j])/(true_result_n1[j]+1e-15)
    data_n2[n_rabi*i+j]=(result_n2[j]-true_result_n2[j])/(true_result_n2[j]+1e-15)
    result_n2s[n_rabi*i+j]=result_n2[j]
    
data=np.array([data_times,data_ls,result_n2s,data_n2])
    
if mode2=="L":  
  np.savetxt(str("Trotter_ls_Gamma="+str(ii)+"Omega.OUT"),np.transpose(data))
else:
  np.savetxt(str("Trotter_ls_H_Gamma="+str(ii)+"Omega.OUT"),np.transpose(data))
