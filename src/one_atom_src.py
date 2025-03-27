import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost
import numpy as np
from matplotlib import pyplot as plt  

def get_Liouvillian_spectrum(Omega,Gamma):
  H = Omega * sigmax()
  sigma_m=(sigmax()-1j*sigmay())/2
  dissipator=[np.sqrt(Gamma)*sigma_m]
  L=-1j*(tensor(H,qeye(2))-tensor(qeye(2),H))+Gamma*(tensor(sigma_m,sigma_m.conj().dag())-1/2*(tensor(sigma_m*sigma_m.dag(),qeye(2))+tensor(qeye(2),sigma_m*sigma_m.dag())))
  print(L)
  print(L.eigenstates())
        
def get_true_dynamics_1_atom(Omega,Gamma,times,psi0,mode="L"):

  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2

  if mode=="L":
    H = Omega * sigmax()
    L=[np.sqrt(Gamma)*sigma_m]
    psi_true = mesolve(H, psi0, times, L).states
  else:
    H = Omega * sigmax()-1j*Gamma/4*(sigmaz()+qeye(2))
    psi_true = sesolve(H, psi0, times).states
        
  n_rabi=times.shape[0]  
  result=np.zeros(n_rabi) 
  
  for i in range(n_rabi):
    result[i]=expect(sigmaz(),psi_true[i])    
    
  return result

def compute_simulation_time(Omega,Gamma,t,l,Gamma_max,v0,omega_base):
  theta=Omega*t/l
  
  v_eff=2*np.pi*v0/omega_base
  f1=Gamma*t/l
  f2=2*np.pi*Gamma_max/(v_eff)
  a=f1/f2
  Delta_max=0
  
  if a<1:                                     # linearly growing frequency, requires t0=30ns
    from scipy.optimize import fsolve
    def func(x,*args):
      return np.pi*args[0]-x+np.sin(x)
    root=fsolve(func,1.5,args=a)
    t0=root[0]/v_eff*4                             # time for decay simulation
    t1=0
    Delta_max=v0*t0/4                           # maximum frequency change
  else:
    Delta_max=omega_base/2
    t0=np.pi/v_eff*4
    t1=(f1-f2)/Gamma_max
    
  t_drive=30
  
  return (2*t_drive+t0+t1)*l
    
def Trotter_1(Omega,Gamma,t,l,psi0,Gamma_max,v0,omega_base,mode,mode2,decay):   # 1=1GHz=1ns, omega_base=pi
  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2
  
  psi=psi0
  
  theta=Omega*t/l
  
  v_eff=2*np.pi*v0/omega_base
  f1=Gamma*t/l
  f2=2*np.pi*Gamma_max/(v_eff)
  a=f1/f2
  Delta_max=0
  
  if a<1:                                     # linearly growing frequency, requires t0=30ns
    from scipy.optimize import fsolve
    def func(x,*args):
      return np.pi*args[0]-x+np.sin(x)
    root=fsolve(func,1.5,args=a)
    t0=root[0]/v_eff*4                             # time for decay simulation
    t1=0
    Delta_max=v0*t0/4                           # maximum frequency change
  else:
    Delta_max=omega_base/2
    t0=np.pi/v_eff*4
    t1=(f1-f2)/Gamma_max 
    
  def omega_z(t,args):
    if t<=t0/4:
      return Delta_max*t/(t0/4)*(1+0)
    elif t0/4<t<=t0/4+t1/2:
      return Delta_max
    elif t0/4+t1/2<t<=3*t0/4+t1/2:
      return Delta_max*(2-(t-t1/2)/(t0/4))
    elif 3*t0/4+t1/2<t<=3*t0/4+t1:
      return -Delta_max
    else:
      return Delta_max*((t-t1)/(t0/4)-4)
    
  def gamma_z(t,args):
    xx=omega_z(t,args)
    return np.sqrt((np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max)   # dissipator coefficient should be square root
        
  #def omega_z_imp(t,args):
  #  xx=omega_z(t,args)
  #  if t<2*t0/4+t1/2:
  #    xx=xx*(1+0)
  #  return xx
  
  t_drive=30      
  H0=[theta/2*sigmax()/t_drive]
  times_0=[0,t_drive]
  times = [0,t0+t1] 
  
  if mode2=="L":

    def gamma_z(t,args):
      xx=omega_z(t,args)
      return np.sqrt((np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max)   # dissipator coefficient should be square root 
    
    if mode=="decay":
      L2=[np.sqrt(decay)*sigma_m]
      L=[[sigma_m,gamma_z],np.sqrt(decay)*sigma_m]        
    else:
      L2=[]
      L=[[sigma_m,gamma_z]] 
    
    for i in range(l):
      psi=mesolve(H0, psi, times_0, L2).states[1]
      psi = mesolve(sigmax()-sigmax(), psi, times, L, options=Options(nsteps=50000)).states[1]
      psi=mesolve(H0, psi, times_0, L2).states[1]
      
  else:
  
    def gamma_z(t,args):
      xx=omega_z(t,args)
      return (np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max   # dissipator coefficient should be square root
  
    if mode=="decay":
      H=[[sigmaz(),omega_z],[-1j/4*(sigmaz()+qeye(2)),gamma_z],-1j/4*(sigmaz()+qeye(2))*decay]
      H0=theta/2*sigmax()/t_drive-1j/4*(sigmaz()+qeye(2))*decay
    else:
      H=[[sigmaz(),omega_z],[-1j/4*(sigmaz()+qeye(2)),gamma_z]]
      H0=theta/2*sigmax()/t_drive

    for i in range(l):
      psi = sesolve(H0, psi, times_0).states[1]
      psi = sesolve(H, psi, times, options=Options(nsteps=100000)).states[1]
      psi = sesolve(H0, psi, times_0).states[1]
      
  t_tot=(2*t_drive+t0+t1)*l
  print(t_tot)
 
  return psi, t_tot

def examine_Trotter_1(l,Omega,Gamma,times,psi0,Gamma_max,v0,omega_base,mode="0",mode2="L",decay=0,return_t=False):
  n_steps=l  
  n_rabi=times.shape[0]
  exp_Trotter=np.zeros(n_rabi)
  t_sim=np.zeros(n_rabi)

  for i in range(n_rabi):
    t_rabi=times[i]
    psi_t, t_tot=Trotter_1(Omega,Gamma,t_rabi,n_steps,psi0,Gamma_max,v0,omega_base,mode,mode2,decay)
    exp_Trotter[i]=np.real(expect(sigmaz(),psi_t))
    t_sim[i]=t_tot
  
  if return_t==True:
    return exp_Trotter, t_sim
    
  return exp_Trotter
