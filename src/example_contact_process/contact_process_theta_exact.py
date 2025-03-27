import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost
import numpy as np
from matplotlib import pyplot as plt  

sx=sigmax()
sy=sigmay()
sz=sigmaz()
id2=qeye(2)
sp=(sx+1j*sy)/2
sm=(sx-1j*sy)/2
res=sm+(-sz+id2)/2 # reset gate, maps anything to [0,1]

q0=basis(2, 1)
q1=basis(2, 0)

CX=tensor((sz+id2)/2,sx)+tensor((-sz+id2)/2,id2)

def one_qubit_op(A,idd,m,N):
  O=idd
  if m==1:
    O=A
    
  for j in range(2,N+1):
    if j==m:
      O=tensor(O,A)
    else:
      O=tensor(O,idd)
      
  return O
  
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

def CX_mn(m,n,N):
  O1=two_qubit_op((sz+id2)/2,sx,id2,m,n,N)
  O2=two_qubit_op((-sz+id2)/2,id2,id2,m,n,N)
  
  return O1+O2
  
def CRxy(m,n,N,theta,R):
  SR=sx
  if R=="y":
    SR=sy
    
  alpha=np.cos(theta/2)*id2-1j*np.sin(theta/2)*SR
    
  O1=two_qubit_op((sz+id2)/2,alpha,id2,m,n,N)
  O2=two_qubit_op((-sz+id2)/2,id2,id2,m,n,N)
  
  return O1+O2
  

psi0=tensor(tensor(tensor(q0,q0),q1),q0)

psi=psi0
N=4
theta=np.pi*3/4

ns=[]
for i in range(4):
  ns.append(one_qubit_op((sz+id2)/2,id2,i+1,N))

Ls=[]
for i in range(4):
  Ls.append(one_qubit_op(sm,id2,i+1,N))

t=0.2
options=Options()
options.atol=1e-4
#options.rtol=1e-10
#options.nsteps=200000
#options.norm_tol=1e-10
psi = mesolve(Ls[0]-Ls[0], psi, [0,t], [Ls[2]], options=options).states[1]

psi=CRxy(3,4,N,theta,"x")*psi*CRxy(3,4,N,theta,"x").dag()  # due to the phase, CR(theta)!=CR(theta).dag()
psi=CRxy(3,2,N,theta,"x")*psi*CRxy(3,2,N,theta,"x").dag()
psi=CRxy(2,3,N,theta,"x")*psi*CRxy(2,3,N,theta,"x").dag()
psi=CRxy(2,1,N,theta,"x")*psi*CRxy(2,1,N,theta,"x").dag()
  
psi = mesolve(Ls[0]-Ls[0], psi, [0,t], [Ls[0], Ls[1]], options=options).states[1]
  
psi=CRxy(1,2,N,theta,"y")*psi*CRxy(1,2,N,theta,"y").dag()   # density matrix, act on both sides
  
# end of first layer, now resetting qubit 1, and qubit 5 =  qubit 1

Resets=[]
for i in range(4):
  Resets.append(one_qubit_op(res,id2,i+1,N))
  
psi=Resets[0]*psi*Resets[0].dag()
print("layer 1 finished")
for i in range(4):
  print(expect(ns[i],psi))
  
# end of first layer, qubit 5 =  qubit 1

# layer 2
psi=CRxy(4,1,N,theta,"x")*psi*CRxy(4,1,N,theta,"x").dag()
psi=CRxy(4,3,N,theta,"x")*psi*CRxy(4,3,N,theta,"x").dag()

psi = mesolve(Ls[0]-Ls[0], psi, [0,t], [Ls[2], Ls[3]], options=options).states[1]

psi=CRxy(3,4,N,theta,"y")*psi*CRxy(3,4,N,theta,"y").dag()
psi=CRxy(3,2,N,theta,"y")*psi*CRxy(3,2,N,theta,"y").dag()
psi=CRxy(2,3,N,theta,"y")*psi*CRxy(2,3,N,theta,"y").dag()

# reset qubit 2
psi=Resets[1]*psi*Resets[1].dag()

print("layer 2 finished")
for i in range(4):
  print(expect(ns[i],psi))

# end of second layer, qubit 6 =  qubit 2

psi = mesolve(Ls[0]-Ls[0], psi, [0,t], [Ls[0]], options=options).states[1]

psi=CRxy(1,2,N,theta,"y")*psi*CRxy(1,2,N,theta,"y").dag()
psi=CRxy(1,4,N,theta,"y")*psi*CRxy(1,4,N,theta,"y").dag()                          # due to this, even if GA has qubit reset, it needs long-range CNOT, which is not possible with the trivial setup
psi=CRxy(4,1,N,theta,"y")*psi*CRxy(4,1,N,theta,"y").dag()
psi=CRxy(4,3,N,theta,"y")*psi*CRxy(4,3,N,theta,"y").dag()

datas=np.zeros(5) # final outcome
datas[0]=expect(ns[2],psi)
datas[1]=expect(ns[3],psi)

# reset qubit 3
psi=Resets[2]*psi*Resets[2].dag()

print("layer 3 finished")
for i in range(4):
  print(expect(ns[i],psi))

# end of third layer, qubit 7 =  qubit 3, qubit 4 is useless now
psi=CRxy(2,3,N,theta,"y")*psi*CRxy(2,3,N,theta,"y").dag()
psi=CRxy(2,1,N,theta,"y")*psi*CRxy(2,1,N,theta,"y").dag()

datas[2]=expect(ns[0],psi)
datas[3]=expect(ns[1],psi)
datas[4]=expect(ns[2],psi)

print(datas)
