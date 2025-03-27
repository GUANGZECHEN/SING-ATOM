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

Deph=4*n2+2*n1
print(Deph)

def no_jump(a, b=None):
    if b is None:
        b = a
    ad_b = a.dag() * b

    D = - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)

    return D

def jump(a, b=None):
    if b is None:
        b = a
    ad_b = a.dag() * b

    D = spre(a) * spost(b.dag())

    return D
    
def get_dynamics_noisy_CZ(psi0,mode,Gamma,Gamma_phi):    # g=1
  options=Options()
  options.atol=1e-3
  #options.rtol=1e-10
  #options.nsteps=200000
  #options.norm_tol=1e-10  
  times=np.linspace(0,np.pi,100)
  
  H=tensor(s10,s12)+tensor(s01,s21)
  L=[np.sqrt(2*Gamma)*tensor(s21,id3), np.sqrt(2*Gamma)*tensor(id3,s21), np.sqrt(Gamma)*tensor(s10,id3), np.sqrt(Gamma)*tensor(s10,id3), np.sqrt(Gamma_phi)*tensor(Deph,id3), np.sqrt(Gamma_phi)*tensor(id3,Deph)]
  
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

def two_qubit_op(A,B,idd,m,n,N): # A on qubit m, B on n, total N qubits
  O=idd
  if m==1:
    O=A
  if n==1:
    O=B
    
  for j in range(2,N+1):
    if j==m:
      O=tensor(O,A)
    elif j==n:
      O=tensor(O,B)
    else:
      O=tensor(O,idd)
      
  return O

def test_two_qubit_op():  
  O=two_qubit_op(n2,n0,id3,3,1,3)
  psi=tensor(tensor(p2,p0),p0)
  print(expect(O,psi))

def get_dynamics_noisy_CZ_many_qubit(psi0,mode,Gamma,Gamma_phi,m,n,N): # m: control qubit, n: X qubit, N: total number of qubits
  options=Options()
  options.atol=1e-3
  #options.rtol=1e-10
  #options.nsteps=200000
  #options.norm_tol=1e-10  
  times=np.linspace(0,np.pi,10)  
  
  H1=two_qubit_op(s10,s12,id3,m,n,N)
  H2=two_qubit_op(s01,s21,id3,m,n,N)
  L1=two_qubit_op(s21,id3,id3,m,n,N)
  L2=two_qubit_op(id3,s21,id3,m,n,N)
  L3=two_qubit_op(s10,id3,id3,m,n,N)
  L4=two_qubit_op(id3,s10,id3,m,n,N)
  L5=two_qubit_op(Deph,id3,id3,m,n,N)
  L6=two_qubit_op(id3,Deph,id3,m,n,N)
  
  H=H1+H2  # g=1
  L=[np.sqrt(2*Gamma)*L1, np.sqrt(2*Gamma)*L2, np.sqrt(Gamma)*L3, np.sqrt(Gamma)*L4, np.sqrt(Gamma_phi)*L5, np.sqrt(Gamma_phi)*L6]


  
  if mode=="CNOT":
    psi0=two_qubit_op(id3,Hada,id3,m,n,N)*psi0   # in case of psi0 being a density matrix, use two_qubit_op(id3,Hada,id3,m,n,N)*psi0*two_qubit_op(id3,Hada,id3,m,n,N)
    
    

  #psi_t = sesolve(H, psi0, times).states
  psi_t = mesolve(H, psi0, times, L).states
  
  n_t=len(psi_t)
  
  for i in range(n_t):
    psi_t[i]=two_qubit_op(id3,Hada,id3,m,n,N)*psi_t[i]*two_qubit_op(id3,Hada,id3,m,n,N)
    # in case of sesolve iwth H, use psi_f=tensor(id3,Hada)*psi_f
  
  return psi_t

def test_CZ_many_qubit():
  psi0=tensor(tensor(p1,p0),p1)
  mode="CNOT"
  Gamma=0.0
  Gamma_phi=0.0
  times=np.linspace(0,np.pi,10) 

  m=3
  n=2
  N=3
  psi_t=get_dynamics_noisy_CZ_many_qubit(psi0,mode,Gamma,Gamma_phi,m,n,N)
    
  n_t=len(psi_t)  
  result=np.zeros((3,n_t))

  N1=two_qubit_op(n1,n0,id3,m,n,N)
  N2=two_qubit_op(n1,n1,id3,m,n,N)
  N3=two_qubit_op(n0,n1,id3,m,n,N)
  for i in range(n_t):
    psi_f=psi_t[i]     

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

#test_two_qubit_op()  
test_CZ_many_qubit()
