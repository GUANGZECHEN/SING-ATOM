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
#print(Hada)

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1
#print(Deph)

def test_RZZ_phi(phi):    # g=1
  options=Options() 
    
  times=[0,phi]
  
  # for testing
  Phi=tensor([p1,p1])+tensor([p0,p0])+tensor([p0,p1])+tensor([p1,p0])
  print(Phi)
  N1=tensor([n0,n0])
  N2=tensor([n0,n1])
  N3=tensor([n1,n0])
  N4=tensor([n1,n1])
  
  psi_t=test_CZ_phi(4*phi)
  
  #H=tensor([n1-n0,n1-n0])
  H=-tensor([n1+n0,n1-n0])-tensor([n1-n0,n1+n0])-tensor(n1+n0,n1+n0)
  print(H)
  
  psi_t = sesolve(-H, psi_t, times).states[1]   # evolution is given by exp(-iHt)
  
  # for testing
  print(psi_t) 
  #print(expect(N1,psi_t))
  #print(expect(N2,psi_t))  
  #print(expect(N3,psi_t))
  #print(expect(N4,psi_t))
      
  return psi_t
  
def test_CZ_phi(phi):    # g=1
  options=Options() 
  
  beta=(phi/np.pi)%2-1
  print("beta=",beta)
  if 1-beta**2<1e-5:
    return tensor([p1,p1])+tensor([p0,p0])+tensor([p0,p1])+tensor([p1,p0])

  alpha=np.sqrt(4*beta**2/(1-beta**2))
  if beta>0:
    Delta=alpha
  else:
    Delta=-alpha 
    
  times=[0,np.pi/np.sqrt(1+(Delta/2)**2)]
  
  # for testing
  Phi=tensor([p1,p1])+tensor([p0,p0])+tensor([p0,p1])+tensor([p1,p0])
  print(Phi)
  N1=tensor([n0,n0])
  N2=tensor([n0,n1])
  N3=tensor([n1,n0])
  N4=tensor([n1,n1])
  
  
  H=tensor([s10,s12])+tensor([s01,s21])-Delta*tensor([n1,n1])
  
  psi_t = sesolve(H, Phi, times).states[1]
  
  # for testing
  print(psi_t) 
  #print(expect(N1,psi_t))
  #print(expect(N2,psi_t))  
  #print(expect(N3,psi_t))
  #print(expect(N4,psi_t))
      
  return psi_t
  
def get_Choi_perfect_CZp(phi):    # g=1
  options=Options()
  
  # get Delta, -Delta is the detuning on level 1 of qubit 1
  
  beta=phi/np.pi-1
  if 1-beta**2<1e-5:
    return tensor([p0,p0,p0,p0])+tensor([p0,p1,p0,p1])+tensor([p1,p0,p1,p0])+tensor([p1,p1,p1,p1])
  
  alpha=np.sqrt(4*beta**2/(1-beta**2))
  if beta>0:
    Delta=alpha
  else:
    Delta=-alpha  
   
  times=[0,np.pi/np.sqrt(1+(Delta/2)**2)]
  
  Phi=tensor([p0,p0,p0,p0])+tensor([p0,p1,p0,p1])+tensor([p1,p0,p1,p0])+tensor([p1,p1,p1,p1])
  Phi=Phi/2
  
  H=tensor([id3,id3,s10,s12])+tensor([id3,id3,s01,s21])-Delta*tensor([id3,id3,n1,n1])
  psi_t = mesolve(H, Phi, times, [0*tensor([id3,id3,s10,s12])]).states[1]
       
  return psi_t
  

def get_Choi_noisy_CZp(Gamma,phi,Gamma_phi):    # g=1
  options=Options()
  
  # get Delta, -Delta is the detuning on level 1 of qubit 1
  beta=phi/np.pi-1
  alpha=np.sqrt(4*beta**2/(1-beta**2))
  if beta>0:
    Delta=alpha
  else:
    Delta=-alpha  
  
  # get decay due to Delta
  gamma=1 # g=gamma
  a=-Delta/(566) # Delta in units of omega_0, omega_0=566g
  phase= (2/3*np.pi+a*4*np.pi) # omega_0 corresponds to 4 pi
  Gamma_waveguide= gamma*(np.abs(1+np.exp(1j*phase)+np.exp(1j*2*phase)))**2 # extra decay on qubit 1, level 1
  Decay2=np.sqrt(2*Gamma)*s21+np.sqrt(Gamma+Gamma_waveguide)*s10
   
  times=[0,np.pi/np.sqrt(1+(Delta/2)**2)]
  
  Phi=tensor([p0,p0,p0,p0])+tensor([p0,p1,p0,p1])+tensor([p1,p0,p1,p0])+tensor([p1,p1,p1,p1])
  Phi=Phi/2
  
  H=tensor([id3,id3,s10,s12])+tensor([id3,id3,s01,s21])-Delta*tensor([id3,id3,n1,n1])
  L=[tensor([id3,id3,Decay2,id3]), np.sqrt(Gamma)*tensor([id3,id3,id3,Decay]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,Deph,id3]), np.sqrt(2*Gamma_phi)*tensor([id3,id3,id3,Deph])]
 
  psi_t = mesolve(H, Phi, times, L).states[1]
       
  return psi_t


n_G1=4
n_phi=51

Gp=0.0028

G1s=np.linspace(0,0.01/np.sqrt(2),n_G1)   # g=sqrt(2)g_0
phis=np.linspace(0.02*np.pi,1.98*np.pi,n_phi)

N=n_phi
r1=np.zeros(N)
r2=np.zeros(N)
r3=np.zeros(N)
r4=np.zeros(N)

for j in range(n_phi):
  phi=phis[j]
  P=get_Choi_perfect_CZp(phi)
  print(P.tr())

  Q1=get_Choi_noisy_CZp(G1s[0],phi,Gp)
  F1=fidelity(P,Q1)**2
  r1[j]=F1
    
  Q2=get_Choi_noisy_CZp(G1s[1],phi,Gp)
  F2=fidelity(P,Q2)**2
  r2[j]=F2

  Q3=get_Choi_noisy_CZp(G1s[2],phi,Gp)
  F3=fidelity(P,Q3)**2
  r3[j]=F3
  
  Q4=get_Choi_noisy_CZp(G1s[3],phi,Gp)
  F4=fidelity(P,Q4)**2
  r4[j]=F4

np.savetxt("Fidelity_CZ_phi.OUT", np.transpose([phis,r1,r2,r3,r4]))
