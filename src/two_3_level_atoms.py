import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops
import numpy as np
from matplotlib import pyplot as plt  

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()

Hada=n2+1/np.sqrt(2)*(n1-n0+s01+s10)
print(Hada)

def get_dynamics_CZ(psi0,mode):
  options=Options()
  options.atol=1e-3
  #options.rtol=1e-10
  #options.nsteps=200000
  #options.norm_tol=1e-10  
  times=np.linspace(0,np.pi,100)
  
  H=tensor(s10,s12)+tensor(s01,s21)
  L=[2*tensor(s21,id3)]
  
  if mode=="CNOT":
    psi0=tensor(id3,Hada)*psi0    
  
  #psi_t = sesolve(H, psi0, times).states
  psi_t = mesolve(H, psi0, times, L).states
  
  n_t=times.shape[0]  
  result=np.zeros((3,n_t))

  #N1=tensor(n1,n1)
  #N2=tensor(n0,n2)
  N1=tensor(n1,n0)
  N2=tensor(n1,n1)
  N3=tensor(n0,n1)
  for i in range(n_t):
    psi_f=psi_t[i]
    if mode=="CNOT":
      psi_f=tensor(id3,Hada)*psi_f*tensor(id3,Hada)
      # in case of sesolve iwth H, use psi_f=tensor(id3,Hada)*psi_f

    result[0,i]=np.real(expect(N1,psi_f))
    result[1,i]=np.real(expect(N2,psi_f))
    result[2,i]=np.real(expect(N3,psi_f))
  
  print(psi_t[n_t-1])  
  plt.figure(figsize=(10,10),dpi=100) 
  #plt.title(f'$g={g}GHz, \\Gamma_1={gamma_1}GHz$',fontsize=30) 
  plt.plot(times,result[0],label='10')
  plt.plot(times,result[1],label='11')
  plt.plot(times,result[2],label='01')
  #plt.xlabel("t [ns]",fontsize=30)
  #plt.ylabel("n",fontsize=30)
  #plt.xticks(fontsize=30)
  #plt.yticks(fontsize=30)
  plt.legend(fontsize=30)
  #plt.locator_params(axis='y', nbins=6)
  #plt.locator_params(axis='x', nbins=6)
  #plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
  plt.show()
  
  return result
    
#def get_dynamics_GA_CNOT():
             
mode="CNOT"
psi0=tensor(p1,p0)
print(psi0)

get_dynamics_CZ(psi0,mode)
