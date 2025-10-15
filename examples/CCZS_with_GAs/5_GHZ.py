import sys
sys.path.append("../../src/")

import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity, ket2dm
import numpy as np
from matplotlib import pyplot as plt  
from three_level_qubits import O_ij, O_i

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()
Sz=n0-n1+n2

Hada=n2+1/np.sqrt(2)*(n1-n0+s01+s10)

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1

# 2 qubit things
q0=basis(2, 1)
q1=basis(2, 0)

#print(q0,q1)

sm=(sigmax()-1j*sigmay())/2
sp=(sigmax()+1j*sigmay())/2
sx=sigmax()
sz=sigmaz()
deph=(sz+qeye(2))/2
id2=qeye(2)

#print(sp*q0,sp*q1)

def O_ij_mixed(O1, O2, i, j, dims):
    """
    Constructs a tensor product operator acting as O1 on qudit i,
    O2 on qudit j, and identities elsewhere.

    Parameters:
    - O1, O2: Qobj operators
    - i, j: indices (integers) of target subsystems
    - dims: list of subsystem dimensions, e.g. [2,3,2,3,2]

    Returns:
    - Qobj: the full operator as a tensor product
    """
    op_list = []
    for idx, d in enumerate(dims):
        if idx == i:
            op_list.append(O1)
        elif idx == j:
            op_list.append(O2)
        else:
            op_list.append(qeye(d))
    
    return tensor(op_list)
    
def O_i_mixed(O, i, dims):
    """
    Constructs a tensor product operator acting as O on subsystem i,
    and identity operators on all other subsystems.

    Parameters:
    - O: Qobj operator to insert
    - i: index of the target subsystem
    - dims: list of subsystem dimensions (e.g. [2, 3, 2, 3, 2])

    Returns:
    - Qobj: the full tensor product operator
    """
    assert isinstance(dims, list), "'dims' must be a list of integers"

    op_list = []
    for idx, d in enumerate(dims):
        if idx == i:
            op_list.append(O)
        else:
            op_list.append(qeye(d))
    return tensor(op_list)

dims=[2,3,2,3,2]

def get_fidelity_noisy_GHZ(Gamma,Gamma_phi):    # g=1     # need to further check with test code on 2,3,2,3,2
  options=Options()
  times=[0,np.pi/np.sqrt(2)]

  psi0=1/np.sqrt(2)*(tensor([q1,p0,q0,p1,q0])+tensor([q1,p1,q0,p1,q0])) # initial state
  
  
  ####### 
  H1 = O_ij_mixed(sm,s12,0,1,dims)+O_ij_mixed(sp,s21,0,1,dims)+O_ij_mixed(s12,sm,1,2,dims)+O_ij_mixed(s21,sp,1,2,dims)
  H2 = O_ij_mixed(sm,s12,2,3,dims)+O_ij_mixed(sp,s21,2,3,dims)+O_ij_mixed(s12,sm,3,4,dims)+O_ij_mixed(s21,sp,3,4,dims)
  H_iSWAP = O_ij_mixed(sm,s01,2,3,dims)+O_ij_mixed(sp,s10,2,3,dims)
  #######
  
  L=[np.sqrt(Gamma)*O_i_mixed(Decay,1,dims), np.sqrt(Gamma)*O_i_mixed(Decay,3,dims), np.sqrt(Gamma)*O_i_mixed(sm,0,dims), np.sqrt(Gamma)*O_i_mixed(sm,2,dims), np.sqrt(Gamma)*O_i_mixed(sm,4,dims), np.sqrt(2*Gamma_phi)*O_i_mixed(Deph,1,dims), np.sqrt(2*Gamma_phi)*O_i_mixed(Deph,3,dims), np.sqrt(2*Gamma_phi)*O_i_mixed(deph,0,dims), np.sqrt(2*Gamma_phi)*O_i_mixed(deph,2,dims), np.sqrt(2*Gamma_phi)*O_i_mixed(deph,4,dims)] 
  # The 1st CCZS
  
  psi_t = mesolve(H1, psi0, times, L).states[1]
  
  Rz = O_i_mixed(sz,2,dims)
  X = O_i_mixed(sx,0,dims)
  psi_t = Rz*X*psi_t*X*Rz    # in practice there should be decay related to this, fix later, single gate 30ns, need to have both Rz for qubits 2 and 3 in practise to align the phase
  
  # iSWAP
  
  times2=[0,np.pi/2]
  psi_t = mesolve(H_iSWAP, psi_t, times2, L).states[1]

  H_Rz = O_i_mixed(sz,0,dims)
  times3=[0,np.pi/4]
  psi_t = mesolve(H_Rz, psi_t, times3, []).states[1]   # assume negligible single-gate time, add later
  
  # The 2nd CCZS
  
  psi_t = mesolve(H2, psi_t, times, L).states[1]
  Rz = O_i_mixed(sz,4,dims)
  X = O_i_mixed(sx,2,dims)
  psi_t = Rz*X*psi_t*X*Rz    # in practice there should be decay related to this, fix later
    
  psi_f = 1/np.sqrt(2)*(tensor([q0,p0,q0,p0,q0])+tensor([q1,p1,q1,p1,q1]))
  
  psi_f = ket2dm(psi_f)
  
  F = fidelity(psi_t,psi_f)**2
         
  return F
  
n_G1=4  
n_Gp=21
G1s=np.linspace(0,0.005,n_G1)  # in units of g=1
Gps=np.linspace(0,0.005,n_Gp)

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
  
np.savetxt("Fidelity_5GHZ_vs_Gp.OUT", np.transpose([Gps,r1,r2,r3,r4]))
np.savetxt("Fidelity_5GHZ_vs_G1.OUT", np.transpose([Gps,R1,R2,R3,R4]))
