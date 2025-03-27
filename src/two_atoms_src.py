import qutip
from qutip import sigmax, basis, sesolve, sigmaz, sigmay, mesolve, expect, qeye, tensor, Options, fidelity, liouvillian, spre, spost
import numpy as np
from matplotlib import pyplot as plt  

def no_jump(a, b=None):
    if b is None:
        b = a
    ad_b = a.dag() * b

    D = - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)

    return D

def jump(a, b=None):
    if b is None:
        b = a
    ad_b = a.dag() * b

    D = spre(a) * spost(b.dag())

    return D

def compute_simulation_time_2(g,gamma_1,t,l,Gamma_max,v0,omega_base):
  v_eff=2*np.pi*v0/omega_base
  f1=gamma_1*t/l
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

  t_drive=t/(2*l)
  t_tot=(2*t_drive+t0+t1)*l
  ave_decay=gamma_1*t/t_tot/g
  
  return t_tot, ave_decay

def get_Liouvillian_matrix_2(g,gamma_1):
  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2

  H = g * (tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))
  idd= tensor(qeye(2),qeye(2))
  L=tensor(sigma_m,qeye(2))
    
  Liouvi=-1j*(tensor(H,idd)-tensor(idd,H.conj().dag()))+gamma_1*(tensor(L,L.conj())-1/2*(tensor(L.dag()*L,idd)+tensor(idd,L.conj().dag()*L.conj())))
  M=Liouvi.full()

  return M

def get_Liouvillian_matrix_with_Bx_2(g,gamma_1,Bx):
  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2

  H = g * (tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))+Bx*tensor(sigmax(),qeye(2))
  idd= tensor(qeye(2),qeye(2))
  L=tensor(sigma_m,qeye(2))
    
  Liouvi=-1j*(tensor(H,idd)-tensor(idd,H.conj().dag()))+gamma_1*(tensor(L,L.conj())-1/2*(tensor(L.dag()*L,idd)+tensor(idd,L.conj().dag()*L.conj())))
  M=Liouvi.full()

  return M
            
def get_true_dynamics_2_atom(g,gamma_1,times,psi0,n1,n2,mode="L",Bx=0):

  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2
  
  options=Options()
  options.atol=1e-12
  options.rtol=1e-10
  options.nsteps=200000
  options.norm_tol=1e-10

  if mode=="L":
    H = g * (tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m)) + Bx*tensor(sigmax(),qeye(2))
    L=[np.sqrt(gamma_1)*tensor(sigma_m,qeye(2))]
    psi_true = mesolve(H, psi0, times, L, options=options).states
  else:
    H = g * (tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))-1j*gamma_1/4*tensor(sigmaz(),qeye(2)) + Bx*tensor(sigmax(),qeye(2))
    psi_true = sesolve(H, psi0, times).states
        
  n_rabi=times.shape[0]  
  result=np.zeros((2,n_rabi)) 
  
  for i in range(n_rabi):
    result[0,i]=expect(n1,psi_true[i])
    result[1,i]=expect(n2,psi_true[i])     

  #plt.figure(figsize=(10,10),dpi=100) 
  #plt.title(f'$g={g}GHz, \\Gamma_1={gamma_1}GHz$',fontsize=30) 
  #plt.plot(times,(result[0]+1)/2,label='Trotter_n1')
  #plt.plot(times,(result[1]+1)/2,label='Trotter_n2')
  #plt.xlabel("t [ns]",fontsize=30)
  #plt.ylabel("n",fontsize=30)
  #plt.xticks(fontsize=30)
  #plt.yticks(fontsize=30)
  #plt.legend(fontsize=30)
  #plt.locator_params(axis='y', nbins=6)
  #plt.locator_params(axis='x', nbins=6)
  #plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
  #plt.show()
  
  return result
    
def Trotter(g,gamma_1,t,l,psi0,Gamma_max,v0,omega_base,mode,mode2="L",decay=0,Bx=0):   # 1=1GHz=1ns, omega_base=pi
  #print(Bx)
  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2
  
  psi=psi0
  
  v_eff=2*np.pi*v0/omega_base
  f1=gamma_1*t/l
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

  print(t0,t1,t+l*(t0+t1))
    
  def omega_z(t,args):
    if t<=t0/4:
      return Delta_max*t/(t0/4)
    elif t0/4<t<=t0/4+t1/2:
      return Delta_max
    elif t0/4+t1/2<t<=3*t0/4+t1/2:
      return Delta_max*(2-(t-t1/2)/(t0/4))
    elif 3*t0/4+t1/2<t<=3*t0/4+t1:
      return -Delta_max
    else:
      return Delta_max*((t-t1)/(t0/4)-4)
  
  H0=g*(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))+Bx*tensor(sigmax(),qeye(2))
  times_0=[0,t/(2*l)]
  times = [0,t0+t1] 
  
  options=Options()
  options.atol=1e-12
  options.rtol=1e-10
  options.nsteps=200000
  #options.norm_tol=1e-10
  
  if mode2=="L":

    def gamma_z(t,args):
      xx=omega_z(t,args)
      return np.sqrt((np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max)    # dissipator coefficient should be square root  
    
    if mode=="g":     
      H=[H0,[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[]
    elif mode=="decay":
      H=[[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[np.sqrt(decay)*tensor(sigma_m,qeye(2)),np.sqrt(decay)*tensor(qeye(2),sigma_m)]   
    elif mode=="dephasing":
      H=[[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[np.sqrt(decay)*tensor(sigmaz(),qeye(2)),np.sqrt(decay)*tensor(qeye(2),sigmaz())]            
    else:
      H=[[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[]
    
    if L2==[]:  
      L=[[tensor(sigma_m,qeye(2)),gamma_z]]
    else:
      L=[[tensor(sigma_m,qeye(2)),gamma_z],L2]    
    
    for i in range(l):
      psi=mesolve(H0, psi, times_0, L2, options=options).states[1]
      psi = mesolve(H, psi, times, L, options=options).states[1]
      psi=mesolve(H0, psi, times_0, L2, options=options).states[1]
      
  else:
  
    def gamma_z(t,args):
      xx=omega_z(t,args)
      return (np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max 
  
    if mode=="g":
      H=[H0,[-1j/4*tensor(sigmaz(),qeye(2)),gamma_z],[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[]
      
    elif mode=="decay":
      print(decay)
      #L2=[np.sqrt(decay)*tensor(qeye(2),sigma_m)]
      A=liouvillian(tensor(sigmaz(),qeye(2)))
      B=no_jump(tensor(sigma_m,qeye(2)))
      C=no_jump(tensor(qeye(2),sigma_m))
      D=liouvillian(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))
      H=[(B+C)*decay,[A,omega_z],[B,gamma_z]]
      H0=[g*D,(B+C)*decay]      
      for i in range(l):
        psi=mesolve(H0, psi, times_0, options=options).states[1]
        psi = mesolve(H, psi, times, options=options).states[1]
        psi=psi/psi.norm()
        psi=mesolve(H0, psi, times_0, options=options).states[1]
      return psi

    elif mode=="dephasing":
      print("dephasing=",decay)
      #L2=[np.sqrt(decay)*tensor(qeye(2),sigma_m)]
      A=liouvillian(tensor(sigmaz(),qeye(2)))
      B=no_jump(tensor(sigma_m,qeye(2)))
      E=no_jump(tensor(sigmaz(),qeye(2)))+jump(tensor(sigmaz(),qeye(2)))
      C=no_jump(tensor(qeye(2),sigmaz()))+jump(tensor(qeye(2),sigmaz()))                              # we substracted a constant term
      D=liouvillian(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))
      H=[(E+C)*decay,[A,omega_z],[B,gamma_z]]
      H0=[g*D,(E+C)*decay]      
      for i in range(l):
        psi=mesolve(H0, psi, times_0, options=options).states[1]
        psi = mesolve(H, psi, times, options=options).states[1]
        psi=psi/psi.norm()
        psi=mesolve(H0, psi, times_0, options=options).states[1]
      return psi
          
    else:
      H=[[-1j/4*tensor(sigmaz(),qeye(2)),gamma_z],[tensor(sigmaz(),qeye(2)),omega_z]]
      L2=[]

    for i in range(l):
      psi=sesolve(H0, psi, times_0).states[1]
      psi = sesolve(H, psi, times, options=Options(nsteps=200000)).states[1]
      psi=sesolve(H0, psi, times_0).states[1]

  return psi
         
def examine_Trotter(l,g,gamma_1,times,psi0,n1,n2,Gamma_max,v0,omega_base,mode="g",mode2="L",decay=0,Bx=0):
  print(Bx)
  n_steps=l  
  n_rabi=times.shape[0]
  exp_Trotter=np.zeros((2,n_rabi))
  
  for i in range(n_rabi):
    print(i)
    t_rabi=times[i]
    psi_t=Trotter(g,gamma_1,t_rabi,n_steps,psi0,Gamma_max,v0,omega_base,mode,mode2,decay,Bx)
    psi_t=psi_t/psi_t.norm()
    exp_Trotter[0,i]=expect(n1,psi_t)
    exp_Trotter[1,i]=expect(n2,psi_t)
    

  #plt.figure(figsize=(10,10),dpi=100) 
  #plt.title(f'$g={g}GHz, \\Gamma_1={gamma_1}GHz, steps={n_steps}$',fontsize=30) 
  #plt.plot(times,(exp_Trotter[0]+1)/2,label='Trotter_n1')
  #plt.plot(times,(exp_Trotter[1]+1)/2,label='Trotter_n2')
  #plt.xlabel("t [ns]",fontsize=30)
  #plt.ylabel("n",fontsize=30)
  #plt.xticks(fontsize=30)
  #plt.yticks(fontsize=30)
  #plt.legend(fontsize=30)
  #plt.locator_params(axis='y', nbins=6)
  #plt.locator_params(axis='x', nbins=6)
  #plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
  #plt.show()
  
  return exp_Trotter

  
def get_Tr_rho(g,gamma_1,t,l,psi0,Gamma_max,v0,omega_base,mode,decay=0,Bx=0):   # 1=1GHz=1ns, omega_base=pi
  #print(Bx)
  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2
  
  psi=psi0
  
  v_eff=2*np.pi*v0/omega_base
  f1=gamma_1*t/l
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

  #print(t0,t1,t+l*(t0+t1))
    
  def omega_z(t,args):
    if t<=t0/4:
      return Delta_max*t/(t0/4)
    elif t0/4<t<=t0/4+t1/2:
      return Delta_max
    elif t0/4+t1/2<t<=3*t0/4+t1/2:
      return Delta_max*(2-(t-t1/2)/(t0/4))
    elif 3*t0/4+t1/2<t<=3*t0/4+t1:
      return -Delta_max
    else:
      return Delta_max*((t-t1)/(t0/4)-4)
  
  H0=g*(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))+Bx*tensor(sigmax(),qeye(2))
  times_0=[0,t/(2*l)]
  times = [0,t0+t1] 
  
  options=Options()
  #options.atol=1e-12
  #options.rtol=1e-10
  options.nsteps=200000
  options.normalize_output=False
  options.norm_tol=1e-10

  def gamma_z(t,args):
    xx=omega_z(t,args)
    return (np.cos(np.pi+xx*2*np.pi/omega_base)+1)/2*Gamma_max 
  
  if mode=="g":
    H=[H0,[-1j/4*tensor(sigmaz(),qeye(2)),gamma_z],[tensor(sigmaz(),qeye(2)),omega_z]]
    L2=[]
      
  elif mode=="decay":
    print(decay)
    #L2=[np.sqrt(decay)*tensor(qeye(2),sigma_m)]
    A=liouvillian(tensor(sigmaz(),qeye(2)))
    B=no_jump(tensor(sigma_m,qeye(2)))
    C=no_jump(tensor(qeye(2),sigma_m))
    D=liouvillian(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))
    H=[(B+C)*decay,[A,omega_z],[B,gamma_z]]
    H0=[g*D,(B+C)*decay]      
    for i in range(l):
      psi=mesolve(H0, psi, times_0, options=options).states[1]
      psi = mesolve(H, psi, times, options=options).states[1]
      psi=psi/psi.norm()
      psi=mesolve(H0, psi, times_0, options=options).states[1]
    return psi

  elif mode=="dephasing":
    print("dephasing=",decay)
    #L2=[np.sqrt(decay)*tensor(qeye(2),sigma_m)]
    A=liouvillian(tensor(sigmaz(),qeye(2)))
    B=no_jump(tensor(sigma_m,qeye(2)))
    E=no_jump(tensor(sigmaz(),qeye(2)))+jump(tensor(sigmaz(),qeye(2)))
    C=no_jump(tensor(qeye(2),sigmaz()))+jump(tensor(qeye(2),sigmaz()))                              # we substracted a constant term
    D=liouvillian(tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))
    H=[(E+C)*decay,[A,omega_z],[B,gamma_z]]
    H0=[g*D,(E+C)*decay]      
    for i in range(l):
      psi=mesolve(H0, psi, times_0, options=options).states[1]
      psi = mesolve(H, psi, times, options=options).states[1]
      psi=psi/psi.norm()
      psi=mesolve(H0, psi, times_0, options=options).states[1]
    return psi
          
  else:
    H=[[-1j/4*tensor(sigmaz(),qeye(2))-1j/4*tensor(qeye(2),qeye(2)),gamma_z],[tensor(sigmaz(),qeye(2)),omega_z]]
    L2=[]
    for i in range(l):
      
      psi=sesolve(H0, psi, times_0, options=options).states[1]

      psi = sesolve(H, psi, times, options=options).states[1]
      #psi = sesolve(H, psi, [0,100], options=options).states[1]
     
      psi=sesolve(H0, psi, times_0, options=options).states[1]
    
    print(psi.norm())
    return psi.norm()

def get_tr_rho_no_Trotter(g,gamma_1,times,psi0,n1,n2,mode="L",Bx=0):

  sigma_m=(sigmax()-1j*sigmay())/2
  sigma_p=(sigmax()+1j*sigmay())/2
  
  options=Options()
  #options.atol=1e-12
  #options.rtol=1e-10
  options.nsteps=200000
  options.norm_tol=1e-10
  options.normalize_output=False

  H = g * (tensor(sigma_m,sigma_p)+tensor(sigma_p,sigma_m))-1j*gamma_1/4*tensor(sigmaz(),qeye(2)) + Bx*tensor(sigmax(),qeye(2)) - 1j*gamma_1/4*tensor(qeye(2),qeye(2))
  psi_true = sesolve(H, psi0, times, options=options).states
        
  n_rabi=times.shape[0]  
  result=np.zeros(n_rabi) 
  
  for i in range(n_rabi):
    result[i]=psi_true[i].norm()   
    
  return result
