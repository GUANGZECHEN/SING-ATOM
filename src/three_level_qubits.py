import sys
import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost, qutrit_basis, three_level_ops, fidelity
import numpy as np
from matplotlib import pyplot as plt  

p2,p1,p0=qutrit_basis()
n2,n1,n0,s12,s10=three_level_ops()
id3=n0+n1+n2
s01=s10.dag()
s21=s12.dag()

Decay=np.sqrt(2)*s21+s10
Deph=2*n2+n1

def O_i(O,i,N,idd=id3):
  X=idd
  if i==0:
    X=O 
  
  for ii in range(1,N):
    if ii==i:
      X=tensor(X,O)
    else:
      X=tensor(X,idd)

  return X
  
def O_ij(O1,O2,i,j,N,idd=id3):
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
  
def RXY3(i,j,N,theta,psi):  # rotation along xy for -theta (e^-iHt)
  H=O_ij(s01,s10,i,j,N,id3)+O_ij(s10,s01,i,j,N,id3)
  psi=mesolve(H,psi,[0,theta]).states[1]
  return psi

def test_RXY3():
  N=2
  psi=tensor([p1,p0])
  psi=RXY3(1,0,N,1/4*np.pi,psi)
  print(psi)
  
def RZZ3(i,j,N,phi0,psi):  # rotation along ZZ for -phi0 (e^-iHt)
  # get Delta, -Delta is the detuning 11
  phi0=phi0/np.pi
  phi0=phi0%2
  phi0=phi0*np.pi  # rotation of RZZ
  
  phi=2*np.pi-4*phi0   # this gives a rotation of CZ(-4*phi0)
  beta=(phi/np.pi)%2-1
  t_CZ=0
  Delta=0
    
  if 1-beta**2>1e-5:
    alpha=np.sqrt(4*beta**2/(1-beta**2))
    if beta>0:
      Delta=alpha
    else:
      Delta=-alpha  
  
    t_CZ=np.pi/np.sqrt(1+(Delta/2)**2)
    
  H=O_ij(s12,s10,i,j,N) + O_ij(s21,s01,i,j,N) - Delta*O_ij(n1,n1,i,j,N)
  psi=mesolve(H,psi,[0,t_CZ]).states[1]
  
  H2=-phi0*O_ij(n1-n0,n1+n0,i,j,N)-phi0*O_ij(n1-n0,n1+n0,j,i,N)-phi0*O_ij(n1+n0,n1+n0,j,i,N)
  psi=mesolve(H2, psi, [0,1]).states[1]
  return psi
  
def test_RZZ3():
  psi=tensor([p1,p0,p0,p0])
  N=4
  
  psi=RZZ3(2,3,N,1/2*np.pi,psi)
  print(psi)
  
#test_RZZ3()
# This is tested to be good
