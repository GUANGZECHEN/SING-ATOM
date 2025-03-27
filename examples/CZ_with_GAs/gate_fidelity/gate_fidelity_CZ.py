import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity
import numpy as np
from matplotlib import pyplot as plt  

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()

Hada=n2+1/np.sqrt(2)*(n1-n0+s01+s10)
print(Hada)

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1
print(Deph)
    
def test_Choi_noisy_CZ(Gamma,Gamma_phi):    # g=1
  options=Options() 
  times=[0,np.pi]
  
  # for testing
  Phi=tensor([p0,p0,p1,p1])
  N1=tensor([n0,n0,n0,n0])
  N2=tensor([n0,n0,n0,n1])
  N3=tensor([n0,n0,n1,n0])
  N4=tensor([n0,n0,n1,n1])
  
  
  H=tensor([id3,id3,s10,s12])+tensor([id3,id3,s01,s21])
  L=[np.sqrt(Gamma)*tensor([id3,id3,Decay,id3]), np.sqrt(Gamma)*tensor([id3,id3,id3,Decay]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,Deph,id3]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,id3,Deph])]

  # for testing  
  Phi=tensor([id3,id3,id3,Hada])*Phi 
  
  psi_t = mesolve(H, Phi, times, L).states[1]
  
  # for testing
  psi_t = tensor([id3,id3,id3,Hada])*psi_t*tensor([id3,id3,id3,Hada]) 
  print(expect(N1,psi_t))
  print(expect(N2,psi_t))  
  print(expect(N3,psi_t))
  print(expect(N4,psi_t))
      
  return psi_t

def get_Choi_noisy_CZ(Gamma,Gamma_phi):    # g=1
  options=Options() 
  times=[0,np.pi]
  
  Phi=tensor([p0,p0,p0,p0])+tensor([p0,p1,p0,p1])+tensor([p1,p0,p1,p0])+tensor([p1,p1,p1,p1])
  Phi=Phi/2
  
  H=tensor([id3,id3,s10,s12])+tensor([id3,id3,s01,s21])
  L=[np.sqrt(Gamma)*tensor([id3,id3,Decay,id3]), np.sqrt(Gamma)*tensor([id3,id3,id3,Decay]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,Deph,id3]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,id3,Deph])]
  
  psi_t = mesolve(H, Phi, times, L).states[1]
       
  return psi_t
  
def get_Fg_CZ(Gamma,Gamma_phi):
  return 1-np.pi*(4/5*Gamma+23/20*Gamma_phi)
  
n_G1=4  
n_Gp=51
G1s=np.linspace(0,0.01/np.sqrt(2),n_G1)  # in units of g=sqrt(2)g_ij=sqrt(2)
Gps=np.linspace(0,0.01/np.sqrt(2),n_Gp)

N=n_Gp
r1=np.zeros(N)
r2=np.zeros(N)
r3=np.zeros(N)
r4=np.zeros(N)

R1=np.zeros(N)
R2=np.zeros(N)
R3=np.zeros(N)
R4=np.zeros(N)

P=get_Choi_noisy_CZ(0,0)
print(P.tr())


for j in range(n_Gp):
  Gp=Gps[j]
  Q1=get_Choi_noisy_CZ(G1s[0],Gp)
  F1=fidelity(P,Q1)**2    
  r1[j]=F1
  
  print(F1)

  Q2=get_Choi_noisy_CZ(G1s[1],Gp)
  F2=fidelity(P,Q2)**2    
  r2[j]=F2
  
  Q3=get_Choi_noisy_CZ(G1s[2],Gp)
  F3=fidelity(P,Q3)**2    
  r3[j]=F3
  
  Q4=get_Choi_noisy_CZ(G1s[3],Gp)
  F4=fidelity(P,Q4)**2    
  r4[j]=F4

  Q1=get_Choi_noisy_CZ(Gp,G1s[0])
  F1=fidelity(P,Q1)**2    
  R1[j]=F1
  
  print(F1)

  Q2=get_Choi_noisy_CZ(Gp,G1s[1])
  F2=fidelity(P,Q2)**2    
  R2[j]=F2
  
  Q3=get_Choi_noisy_CZ(Gp,G1s[2])
  F3=fidelity(P,Q3)**2    
  R3[j]=F3
  
  Q4=get_Choi_noisy_CZ(Gp,G1s[3])
  F4=fidelity(P,Q4)**2    
  R4[j]=F4  
  
np.savetxt("Fidelity_CZ_vs_Gp.OUT", np.transpose([Gps,r1,r2,r3,r4]))
np.savetxt("Fidelity_CZ_vs_G1.OUT", np.transpose([Gps,R1,R2,R3,R4]))
