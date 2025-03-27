import sys
import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity
import numpy as np
from matplotlib import pyplot as plt

sx=sigmax()
sy=sigmay()
sz=sigmaz()
sm=(sx-1j*sy)/2
sp=(sx+1j*sy)/2
id2=qeye(2)
res=sm+(-sz+id2)/2 # reset gate, maps anything to [0,1]

pd=basis(2, 1)
pu=basis(2, 0)

def O_i(O,i,N,idd):
  X=idd
  if i==0:
    X=O 
  
  for ii in range(1,N):
    if ii==i:
      X=tensor(X,O)
    else:
      X=tensor(X,idd)

  return X
  
def O_ij(O1,O2,i,j,N,idd):
  X=idd
  if i==0:
    X=O1
  elif j==0:
    X=O2 
  
  for ii in range(1,N):
    if ii==i:
      X=tensor(X,O1)
    elif ii==j:
      X=tensor(X,O2)
    else:
      X=tensor(X,idd)

  return X
  
def RXY(i,j,N,theta,psi):  # rotation along xy for -theta (e^-iHt)
  H=O_ij(sp,sm,i,j,N,id2)+O_ij(sm,sp,i,j,N,id2)
  psi=sesolve(H,psi,[0,theta]).states[1]
  return psi

def test_RXY():
  N=3
  psi=tensor([pu,pd,pu])
  psi=RXY(2,0,N,1/4*np.pi,psi)
  print(psi)
  
def RZZ(i,j,N,theta,psi):  # rotation along ZZ for -theta (e^-iHt)
  H=O_ij(sz,sz,i,j,N,id2)
  psi=sesolve(H,psi,[0,theta]).states[1]
  return psi
  
def test_RZZ():
  N=2
  psi=tensor([pu,pu])+tensor([pu,pd])+tensor([pd,pu])+tensor([pd,pd])
  psi=RZZ(1,0,N,1/4*np.pi,psi)
  print(psi)
  
#test_RZZ()
