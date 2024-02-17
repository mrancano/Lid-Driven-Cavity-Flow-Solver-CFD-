#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import divergenceoperator,solutiontools,meshgen


N=3
#div =divergenceoperator.DivergenceOp(N,1/N)
phi_vec = np.zeros((1,(N)*(N)))
utils = solutiontools.utilsClass(N)
i=0
for y in utils.y_vecdual:
    for x in utils.x_vecdual:
        phi_vec[0,i] = x**2-y**2
        i+=1

def DIVCONV(N):
    div = divergenceoperator.DivergenceOp(N,1/N)
    m = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    u_vec = np.zeros((1,(N+1)*(N)))
    v_vec = np.zeros((1,(N+1)*(N)))
    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vec:
            u_vec[0,i] = np.sin(x)*np.cos(y)
            i+=1
    i=0
    for y in utils.y_vec:
        for x in utils.x_vecdual:
            v_vec[0,i] = np.cos(x)*np.sin(y)
            i+=1

    exact = 2*np.cos(m.x_meshdual)*np.cos(m.y_meshdual)
    #exact = exact[1:-1,1:-1]

    numeric = div.apply(u_vec,v_vec).reshape((N,N))

    return np.mean((exact-numeric)**2)

N_list = np.logspace(2,7,5,endpoint=False,base=2,dtype=int)


#N_list = [3]
errorlist1=[]


for N in N_list:
    errorlist1.append(DIVCONV(N))

    
    print(N)

plt.plot(N_list,errorlist1,'x',label="Divergence")
plt.plot(N_list,1e-3/N_list**4,label="Second Order Slope")


plt.legend()
plt.xlabel("1/h")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log')
plt.show()


# %%
