#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import gradientoperator,solutiontools,meshgen




#print(g.phigrad)

N=3
xg = gradientoperator.phigradientX(N,1/N)
yg = gradientoperator.phigradientY(N,1/N)
utils = solutiontools.utilsClass(N)
phi_vec = np.zeros((1,(N)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vecdual:
        phi_vec[0,i] = x**2-y**2
        i+=1

def XgradConv(N):
    xg = gradientoperator.phigradientX(N,1/N)
    m = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    phi_vec = np.zeros((1,(N)*(N)))
    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vecdual:
            phi_vec[0,i] = np.sin(x)*np.cos(y)
            i+=1

    exact = np.cos(m.x_meshU)*np.cos(m.y_meshU)
    exact = exact[1:-1,1:-1]

    numeric = xg.applyandinteriorize(phi_vec).reshape((N-2,N-1))

    return np.mean((exact-numeric)**2)

def YgradConv(N):
    yg = gradientoperator.phigradientY(N,1/N)
    m = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    phi_vec = np.zeros((1,(N)*(N)))
    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vecdual:
            phi_vec[0,i] = np.sin(x)*np.cos(y)
            i+=1

    exact = -np.sin(m.x_meshV)*np.sin(m.y_meshV)
    exact = exact[1:-1,1:-1]

    numeric = yg.applyandinteriorize(phi_vec).reshape((N-1,N-2))

    return np.mean((exact-numeric)**2)
    
N_list = np.logspace(2,7,5,endpoint=False,base=2,dtype=int)




#N_list = [3]
errorlist1=[]
errorlist2=[]

for N in N_list:
    errorlist1.append(XgradConv(N))
    errorlist2.append(YgradConv(N))
    
    print(N)

#plt.plot(-np.log(1/N_list),errorlist1,'x',label="X Gradient")
plt.plot(N_list,errorlist1,'x',label="X Gradient")
plt.plot(N_list,errorlist2,'x',label="Y Gradient")
plt.plot(N_list,1e-5/N_list**4,label="Second Order Slope")



plt.legend()
plt.xlabel("1/h")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log')
plt.show()

# %%
