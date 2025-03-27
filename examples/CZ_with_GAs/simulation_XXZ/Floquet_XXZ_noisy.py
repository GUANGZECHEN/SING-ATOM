import sys
sys.path.append("../../src/")

import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity
import numpy as np
from matplotlib import pyplot as plt  

import three_level_qubits
from three_level_qubits import O_i,O_ij,RXY3,RZZ3

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1

J=1
Jz=5
Gamma=0
N=4

g=1
g2=2.05/1.79    # the larger coupling
g_CZ=np.sqrt(2)
g_CZ2=g2*np.sqrt(2)

Gamma_1=0.02/(1.79*2*2*np.pi) #(in units of g)
Gamma_phi=0.05/(1.79*2*2*np.pi) #(in units of g)

psi0=tensor([p1,p0,p0,p0])
  
t_tot=2
nt=51
times=np.linspace(0, t_tot, nt)
l=10 # number of trotter steps, 50 allows a time of arounnd 7.5 to have high precision

N1=O_i(n1,0,N)
N2=O_i(n1,1,N)
N3=O_i(n1,2,N)
N4=O_i(n1,3,N)

options=Options()
#options.atol=1e-20
#options.rtol=1e-5
#options.nsteps=100000
#options.norm_tol=1e-3
  
def get_psi_Trotter(t,psi0,l):   # need to go to 3-level space, g=1
  theta=2*J*t/l
  phi0=Jz*t/l
  
  # get Delta, -Delta is the detuning 11
  phi0=phi0/np.pi
  phi0=phi0%2
  phi0=phi0*np.pi  # rotation of RZZ
  
  phi=2*np.pi-4*phi0  # this gives a rotation of CZ(-4*phi0)
  
  beta=(phi/np.pi)%2-1
  t_CZ=0
  Delta=0
    
  if 1-beta**2>1e-5:
    alpha=np.sqrt(4*beta**2/(1-beta**2))
    if beta>0:
      Delta=alpha
    else:
      Delta=-alpha  # Delta in units of g
  
    t_CZ=np.pi/np.sqrt(1+(Delta/2)**2)
  
  print(Delta)
  
  # get decay due to Delta
  gamma=1/1.79 # 1/1.78*g
  omega_0=1600*gamma # omega_0 in units of g
  a=Delta/(omega_0) # Delta in units of omega_0
  
  
  phase1= (0.0887*2*np.pi+a*2*np.pi) # omega_0 corresponds to 2 pi, this is phase shift for a distance of Delta_x at frequency \omega_{DF,n1}
  
  phase2= (0.3387*2*np.pi+a*2*np.pi) # omega_0 corresponds to 2 pi, this is phase shift for a distance of Delta_x at frequency \omega_{DF,n4}
    
  Gamma_CZ1=gamma*(np.abs(1+np.exp(1j*2*phase1)+np.sqrt(1.5)*np.exp(1j*4*phase1)+np.sqrt(1.5)*np.exp(1j*6*phase1)+np.exp(1j*8*phase1)+np.exp(1j*10*phase1)))**2 # extra decay of the detuned CZ qubit at \omega_{DF,n1}
  
  
  Gamma_CZ2=gamma*(np.abs(1+np.exp(1j*2*phase2)+np.sqrt(1.5)*np.exp(1j*4*phase2)+np.sqrt(1.5)*np.exp(1j*6*phase2)+np.exp(1j*8*phase2)+np.exp(1j*10*phase2)))**2 # extra decay of the detuned CZ qubit at \omega_{DF,n4} 
  
  # These are a major error resource.

  Decay1=np.sqrt(2*Gamma_1)*s21+np.sqrt(Gamma_1+Gamma_CZ1)*s10 
  Decay2=np.sqrt(2*Gamma_1)*s21+np.sqrt(Gamma_1+Gamma_CZ2)*s10
  
  Gamma_RZ=0.03*0.02  # effective decay rate when performing RZ, =30ns*Gamma_1
  Gp_RZ=0.03*0.05 # effective dephasing rate when performing RZ, =30ns*Gamma_phi
  
  if t==0:
    Gamma_RZ=0
    Gp_RZ=0
  
  # iSWAP
  Hi1 = O_ij(s10,s01,0,1,N)+O_ij(s01,s10,0,1,N)
  Hi2 = O_ij(s10,s01,2,3,N)+O_ij(s01,s10,2,3,N)
  
  # CZ
  HC1 = O_ij(s12,s10,0,1,N) + O_ij(s21,s01,0,1,N) - Delta*O_ij(n1,n1,0,1,N)
  HC2 = O_ij(s12,s10,2,3,N) + O_ij(s21,s01,2,3,N) - Delta*O_ij(n1,n1,2,3,N)
    
  # iSWAP  
  Hi3=(O_ij(s10,s01,1,2,N)+O_ij(s01,s10,1,2,N))
  
  # CZ
  HC3=O_ij(s10,s12,1,2,N) + O_ij(s01,s21,1,2,N) - Delta*O_ij(n1,n1,1,2,N) 

  # RZ gates
  H0Z=-phi0*O_ij(n1-n0,n1+n0,0,1,N)
  H1Z=-phi0*O_ij(n1-n0,n1+n0,1,0,N)
  H01Z=-phi0*O_ij(n1+n0,n1+n0,1,0,N)
  H2Z=-phi0*O_ij(n1-n0,n1+n0,2,3,N)
  H3Z=-phi0*O_ij(n1-n0,n1+n0,3,2,N)
  H23Z=-phi0*O_ij(n1+n0,n1+n0,2,3,N)
  H4Z=-phi0*O_ij(n1-n0,n1+n0,1,2,N)
  H5Z=-phi0*O_ij(n1-n0,n1+n0,2,1,N)
  H45Z=-phi0*O_ij(n1+n0,n1+n0,1,2,N)
  
  # Decay
  L_Decay=[np.sqrt(1.36/1.79)*O_i(s10,3,N)] # extra decay, in units of g=1
  
  ## unwanted extra decay 
  # when performing Rz
  L_RZ=[np.sqrt(Gamma_RZ)*O_i(Decay,0,N),np.sqrt(Gamma_RZ)*O_i(Decay,1,N),np.sqrt(Gamma_RZ)*O_i(Decay,2,N),np.sqrt(Gamma_RZ)*O_i(Decay,3,N),np.sqrt(Gp_RZ)*O_i(Deph,0,N),np.sqrt(Gp_RZ)*O_i(Deph,1,N),np.sqrt(Gp_RZ)*O_i(Deph,2,N),np.sqrt(Gp_RZ)*O_i(Deph,3,N)]

  # when performing iSWAP
  L_iSWAP=[np.sqrt(Gamma_1)*O_i(Decay,0,N), np.sqrt(2*Gamma_phi)*O_i(Deph,0,N), np.sqrt(Gamma_1)*O_i(Decay,1,N), np.sqrt(2*Gamma_phi)*O_i(Deph,1,N), np.sqrt(Gamma_1)*O_i(Decay,2,N), np.sqrt(2*Gamma_phi)*O_i(Deph,2,N), np.sqrt(Gamma_1)*O_i(Decay,3,N), np.sqrt(2*Gamma_phi)*O_i(Deph,3,N)] # error when doing iSWAP
  
  # when performing CZ
  L_CZ1=[np.sqrt(Gamma_1)*O_i(Decay,0,N),np.sqrt(Gamma_1)*O_i(Decay,1,N),np.sqrt(Gamma_1)*O_i(Decay,2,N),O_i(Decay2,3,N),np.sqrt(2*Gamma_phi)*O_i(Deph,0,N),np.sqrt(2*Gamma_phi)*O_i(Deph,1,N),np.sqrt(2*Gamma_phi)*O_i(Deph,2,N),np.sqrt(2*Gamma_phi)*O_i(Deph,3,N)] # error when doing first CZ
  
  L_CZ2=[np.sqrt(Gamma_1)*O_i(Decay,0,N),O_i(Decay1,1,N),np.sqrt(Gamma_1)*O_i(Decay,2,N),O_i(Decay2,3,N),np.sqrt(2*Gamma_phi)*O_i(Deph,0,N),np.sqrt(2*Gamma_phi)*O_i(Deph,1,N),np.sqrt(2*Gamma_phi)*O_i(Deph,2,N),np.sqrt(2*Gamma_phi)*O_i(Deph,3,N)] # error when doing first CZ
  
  # when performing the other CZ
  L_CZ4=[np.sqrt(Gamma_1)*O_i(Decay,0,N),O_i(Decay2,1,N),np.sqrt(Gamma_1)*O_i(Decay,2,N),O_i(Decay1,3,N),np.sqrt(2*Gamma_phi)*O_i(Deph,0,N),np.sqrt(2*Gamma_phi)*O_i(Deph,1,N),np.sqrt(2*Gamma_phi)*O_i(Deph,2,N),np.sqrt(2*Gamma_phi)*O_i(Deph,3,N)] # error when doing second CZ
  
  psi=psi0
  for i in range(l):
    # iSWAP
    psi=mesolve(Hi1, psi, [0,theta-theta/g2], [L_iSWAP], options=options).states[1]
    psi=mesolve(Hi1+Hi2*g2, psi, [0,theta/g2], [L_iSWAP], options=options).states[1]

    # CZ
    psi=mesolve(HC1*g_CZ2+HC2*g_CZ, psi, [0,t_CZ/g_CZ2], [L_CZ2], options=options).states[1]
    psi=mesolve(HC2, psi, [0,t_CZ/g_CZ-t_CZ/g_CZ2], [L_CZ1], options=options).states[1]
    
    # RZ
    psi=mesolve(H0Z+H1Z+H2Z+H3Z+H01Z+H23Z, psi, [0,1], [L_RZ], options=options).states[1]     # perform Rz to align phase
    
    # iSWAP
    psi=mesolve(Hi3-Hi3, psi, [0,theta-theta/g2], [L_iSWAP], options=options).states[1] # for 4 sites, this term is 0
    psi=mesolve(Hi3*g2, psi, [0,theta/g2], [L_iSWAP], options=options).states[1]

    # CZ    
    psi=mesolve(HC3*g_CZ, psi, [0,t_CZ/g_CZ2], [L_CZ4], options=options).states[1]
    
    # CZ and Decay
    psi=mesolve(HC3*g_CZ, psi, [0,t_CZ/g_CZ-t_CZ/g_CZ2], [L_CZ4,L_Decay], options=options).states[1]
    
    # Decay 
    psi=mesolve(H4Z-H4Z, psi, [0,Gamma*t/l/(1.36/1.79)-(t_CZ/g_CZ-t_CZ/g_CZ2)], [L_Decay,L_iSWAP], options=options).states[1] 
    
    # RZ
    psi=mesolve(H4Z+H5Z+H45Z, psi, [0,1], [L_RZ], options=options).states[1]
     
  return psi

psis=[]
for i in range(nt):
  t=times[i]
  psis.append(get_psi_Trotter(t,psi0,l))
  
C1s=np.zeros(nt)
C2s=np.zeros(nt)
C3s=np.zeros(nt)
C4s=np.zeros(nt)

for i in range(nt):
  psi=psis[i]
  C1s[i]=expect(N1,psi)
  C2s[i]=expect(N2,psi)
  C3s[i]=expect(N3,psi)
  C4s[i]=expect(N4,psi)

data=np.array([times,C1s,C2s,C3s,C4s])
np.savetxt(str("Jz="+str(Jz)+"_Gamma="+str(Gamma)+"/l="+str(l)+".OUT"),np.transpose(data))

plt.plot(times,C1s,label='C1')
plt.plot(times,C2s,label='C2')
plt.plot(times,C3s,label='C3')
plt.plot(times,C4s,label='C4')
plt.legend()
plt.savefig(str("Jz="+str(Jz)+"_Gamma="+str(Gamma)+"/l="+str(l)+".png"))

