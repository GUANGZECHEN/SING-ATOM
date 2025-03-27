import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity
import numpy as np
from matplotlib import pyplot as plt  

p0=basis(2, 1)
p1=basis(2, 0)
id2=qeye(2)

print(p0)
print(p1)
print(id2)

sm=(sigmax()-1j*sigmay())/2
sp=(sigmax()+1j*sigmay())/2

print(sm)

Decay=sm
Deph=(sigmaz()+qeye(2))/2

def get_Choi_noisy_iSWAP(Gamma,Gamma_phi):    # g=1
  options=Options() 
  times=[0,np.pi/2]
  
  Phi=tensor([p0,p0,p0,p0])+tensor([p0,p1,p0,p1])+tensor([p1,p0,p1,p0])+tensor([p1,p1,p1,p1])
  Phi=Phi/2
  
  H=tensor([id2,id2,sm,sp])+tensor([id2,id2,sp,sm])
  
  L=[np.sqrt(Gamma)*tensor([id2,id2,Decay,id2]), np.sqrt(Gamma)*tensor([id2,id2,id2,Decay]), np.sqrt(2*Gamma_phi)*tensor([id2,id2,Deph,id2]), np.sqrt(2*Gamma_phi)*tensor([id2,id2,id2,Deph])]
 
  psi_t = mesolve(H, Phi, times, L).states[1]
       
  return psi_t
  
n_G1=4  
n_Gp=51
G1s=np.linspace(0,0.01,n_G1)
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

P=get_Choi_noisy_iSWAP(0,0)
print(P.tr())
print(fidelity(P,P))


for j in range(n_Gp):
  Gp=Gps[j]
  Q1=get_Choi_noisy_iSWAP(G1s[0],Gp)
  F1=fidelity(P,Q1)**2   
  r1[j]=F1
  
  print(F1)

  Q2=get_Choi_noisy_iSWAP(G1s[1],Gp)
  F2=fidelity(P,Q2)**2    
  r2[j]=F2
  
  Q3=get_Choi_noisy_iSWAP(G1s[2],Gp)
  F3=fidelity(P,Q3)**2    
  r3[j]=F3
  
  Q4=get_Choi_noisy_iSWAP(G1s[3],Gp)
  F4=fidelity(P,Q4)**2    
  r4[j]=F4

  Q1=get_Choi_noisy_iSWAP(Gp,G1s[0])
  F1=fidelity(P,Q1)**2    
  R1[j]=F1
  
  print(F1)

  Q2=get_Choi_noisy_iSWAP(Gp,G1s[1])
  F2=fidelity(P,Q2)**2    
  R2[j]=F2
  
  Q3=get_Choi_noisy_iSWAP(Gp,G1s[2])
  F3=fidelity(P,Q3)**2    
  R3[j]=F3
  
  Q4=get_Choi_noisy_iSWAP(Gp,G1s[3])
  F4=fidelity(P,Q4)**2    
  R4[j]=F4  
  
np.savetxt("Fidelity_iSWAP_2_qubit_vs_Gp.OUT", np.transpose([Gps,r1,r2,r3,r4]))
np.savetxt("Fidelity_iSWAP_2_qubit_vs_G1.OUT", np.transpose([Gps,R1,R2,R3,R4]))
