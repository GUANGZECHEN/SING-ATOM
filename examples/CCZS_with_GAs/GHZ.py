import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity, ket2dm
import numpy as np
from matplotlib import pyplot as plt  

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()
Sz=n0-n1+n2


Hada=n2+1/np.sqrt(2)*(n1-n0+s01+s10)

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1

def get_fidelity_noisy_GHZ(Gamma,Gamma_phi):    # g=1
  options=Options()
  times=[0,np.pi/np.sqrt(2)]

  psi0=1/np.sqrt(2)*(tensor([p1,p0,p0])+tensor([p1,p1,p0]))
    
  H=tensor([s10,s12,id3])+tensor([s01,s21,id3])+(tensor([id3,s12,s10])+tensor([id3,s21,s01]))
  L=[np.sqrt(Gamma)*tensor([Decay,id3,id3]), np.sqrt(Gamma)*tensor([id3,Decay,id3]),np.sqrt(Gamma)*tensor([id3,id3,Decay]), np.sqrt(2*Gamma_phi)*tensor([Deph,id3,id3]), np.sqrt(2*Gamma_phi)*tensor([id3,Deph,id3]),np.sqrt(2*Gamma_phi)*tensor([id3,id3,Deph])]
  
  psi_t = mesolve(H, psi0, times, L).states[1]
  
  Rz = tensor(id3,id3,Sz)
  X = tensor(s10+s01,id3,id3)
  psi_t = Rz*X*psi_t*X*Rz
  
  psi_f = 1/np.sqrt(2)*(tensor([p0,p0,p0])+tensor([p1,p1,p1]))
  
  psi_f = ket2dm(psi_f)
  
  F = fidelity(psi_t,psi_f)**2
         
  return F
  
n_G1=4  
n_Gp=51
G1s=np.linspace(0,0.01,n_G1)  # in units of g=1
Gps=np.linspace(0,0.01,n_Gp)

N=n_Gp
r1=np.zeros(N)
r2=np.zeros(N)
r3=np.zeros(N)
r4=np.zeros(N)

R1=np.zeros(N)
R2=np.zeros(N)
R3=np.zeros(N)
R4=np.zeros(N)

for j in range(n_Gp):
  print(j)
  
  Gp=Gps[j]
  F1=get_fidelity_noisy_GHZ(G1s[0],Gp)   
  r1[j]=F1
  
  print(F1)

  F2=get_fidelity_noisy_GHZ(G1s[1],Gp)    
  r2[j]=F2
  
  F3=get_fidelity_noisy_GHZ(G1s[2],Gp)  
  r3[j]=F3
  
  F4=get_fidelity_noisy_GHZ(G1s[3],Gp)     
  r4[j]=F4

  F1=get_fidelity_noisy_GHZ(Gp,G1s[0])    
  R1[j]=F1
  
  #print(F1)

  F2=get_fidelity_noisy_GHZ(Gp,G1s[1])   
  R2[j]=F2
  
  F3=get_fidelity_noisy_GHZ(Gp,G1s[2])    
  R3[j]=F3
  
  F4=get_fidelity_noisy_GHZ(Gp,G1s[3])    
  R4[j]=F4  
  
np.savetxt("Fidelity_GHZ_vs_Gp.OUT", np.transpose([Gps,r1,r2,r3,r4]))
np.savetxt("Fidelity_GHZ_vs_G1.OUT", np.transpose([Gps,R1,R2,R3,R4]))
