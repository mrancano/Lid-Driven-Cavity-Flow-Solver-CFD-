#%%
import numpy as np
import scipy as sp

import momentumstep
import pressurestep
import divergenceoperator
import solutiontools
from plotsolution import plotx,ploty,plotp
import copy

#PARAMETERS
N = 20
G = N+2
Re = 100
k = 0.05
n_time = 200
h=1/N

fouriernumber = k/(Re*h**2)
#print(fouriernumber)
CFL = k/h


#p_init = 1 
#Pearson Flow ICs



uN = solutiontools.utilsClass(N)
u_int,v_int = uN.genPearSol(0)
phi_int = uN.genPearPhi(0)

u_int = np.zeros((1,(N+1)*(N)))
v_int = np.zeros((1,(N+1)*(N)))
phi_int = np.zeros((1,(N)*(N)))



u_initial = uN.applyUBCs(u_int)
v_initial =  uN.applyVBCs(v_int)
#phi_initial =  uN.applyPHIBCs(phi_int)



#print(u_initial.reshape(G,G+1))
#print(v_initial.reshape(G+1,G))
#print(phi_initial.reshape(G,G))


#FOLLOWING PSEUDO ALGORITHM
mu = momentumstep.Ustarstep(N,k,Re)
mv = momentumstep.Vstarstep(N,k,Re)
c = pressurestep.pressureCalc(N,k)
p = pressurestep.pressureStep(N)
#print("1")
u_star = mu.step(N,k,Re,u_initial.T,v_initial.T,u_initial.T,v_initial.T)
v_star = mv.step(N,k,Re,u_initial.T,v_initial.T,u_initial.T,v_initial.T)
#print('2')
phinplus1 = c.step(N,u_star,v_star)

#print(phinplus1.reshape(G,G))
#print(u_star.reshape((G),G+1))

u_nplus1,v_nplus1 = p.step(N,k,u_star,v_star,phinplus1)

u_nplus1 = uN.applyUBCs(u_nplus1.reshape((G,G+1))[1:-1,1:-1].flatten())
v_nplus1 =  uN.applyVBCs(v_nplus1.reshape((G+1,G))[1:-1,1:-1].flatten())
#phinplus1 =  uN.applyPHIBCs(phinplus1.reshape((G,G))[1:-1,1:-1].flatten())

#print(u_star.reshape((G),G+1))
#print(u_nplus1.reshape((G),G+1))
#print(v_nplus1.reshape((G+1),G))
#print(uN.genPearPhi(0.1))
#CHECK PRESSURE GRADS

u_prev = copy.copy(u_initial)
v_prev = copy.copy(v_initial)

u_n = copy.copy(u_nplus1)
v_n = copy.copy(v_nplus1)

for i in range(n_time):
    #print((i+1)*k)
    u_star = mu.step(N,k,Re,u_n.T,v_n.T,u_prev.T,v_prev.T)
    v_star = mv.step(N,k,Re,u_n.T,v_n.T,u_prev.T,v_prev.T)

    phinplus1 = c.step(N,u_star,v_star)

    u_nplus1,v_nplus1 = p.step(N,k,u_star,v_star,phinplus1)

    #u_nplus1 = uN.applyUBCs(u_nplus1.reshape((G,G+1))[1:-1,1:-1].flatten())
    #v_nplus1 =  uN.applyVBCs(v_nplus1.reshape((G+1,G))[1:-1,1:-1].flatten())
    #phinplus1 =  uN.applyPHIBCs(phinplus1.reshape((G,G))[1:-1,1:-1].flatten())

    u_prev = copy.copy(u_n)
    v_prev = copy.copy(v_n)

    u_n = copy.copy(u_nplus1)
    v_n = copy.copy(v_nplus1)

    if i%(n_time/10) == 0:
        print(i)
        plotx(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)
        

plotx(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)
ploty(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)
plotp(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)

u_ex , v_ex = uN.genPearSol((i+1)*k)
phi_ex = uN.genPearPhi((i+1)*k)


with np.printoptions(precision=4, suppress=True):
    z=0
    #print(u_nplus1.reshape(G,G+1))
    #print(v_nplus1.reshape(G+1,G))
    #print(phinplus1.reshape(G,G))

#print(np.mean((u_ex-u_nplus1.reshape((G,G+1))[1:-1,1:-1].flatten())**2))
#print(np.mean((v_ex-v_nplus1.reshape((G+1,G))[1:-1,1:-1].flatten())**2))
#print(np.mean((phi_ex-phinplus1.reshape((G,G))[1:-1,1:-1].flatten())**2))



'''

D = divergenceoperator.DivergenceOp(G,1/N)
div = D.apply(u_nplus1,v_nplus1).reshape(G,G)[1:-1,1:-1].flatten()
print(div)


plotp(N,0,0,D.apply(u_nplus1,v_nplus1),Re,(i+1)*k,fouriernumber,CFL)

plotx(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)
ploty(N,u_nplus1,v_nplus1,phinplus1,Re,(i+1)*k,fouriernumber,CFL)

N=3
test = solutiontools.utilsClass(N)
u_int,v_int = test.genPearSol(0)
test.applyUBCs(u_int)

'''

# %%
