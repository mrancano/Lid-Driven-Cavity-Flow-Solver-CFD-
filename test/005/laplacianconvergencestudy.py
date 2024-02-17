#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import solutiontools
from src import meshgen
from src import laplacianoperator

N = 3

lu = laplacianoperator.Ulaplacian(N,1/N)
lv = laplacianoperator.Vlaplacian(N,1/N)
lphi = laplacianoperator.Philaplacian(N,1/N)
meshop = meshgen.meshGenClass(N)

def u(x,y):
    return np.sin(x)*np.cos(y)
def v(x,y):
    return np.cos(x)*np.sin(y)
def Ulaplacian(x,y):
    return (-2*np.sin(x)*np.cos(y))
def Vlaplacian(x,y):
    return (-2*np.cos(x)*np.sin(y))






def ULaplacianConvergenceStudy(N,u,v,laplacian):
    m = meshgen.meshGenClass(N)
    lu = laplacianoperator.Ulaplacian(N,1/N)
    utils = solutiontools.utilsClass(N)

    u_vec = np.zeros((1,(N+1)*(N)))
    v_vec = np.zeros((1,(N+1)*(N)))

    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vec:
            u_vec[0,i] = u(x,y)
            i+=1
    i=0
    for y in utils.y_vec:
        for x in utils.x_vecdual:
            v_vec[0,i] = v(x,y)
            i+=1

    numeric = lu.applyUlap(u_vec)
    numeric = numeric.reshape((N-2,N-1))
    
    exact = laplacian(m.x_meshU,m.y_meshU)[1:-1,1:-1]

    return np.mean((exact-numeric))**2

def VLaplacianConvergenceStudy(N,u,v,laplacian):
    m = meshgen.meshGenClass(N)
    lv = laplacianoperator.Vlaplacian(N,1/N)
    utils = solutiontools.utilsClass(N)

    u_vec = np.zeros((1,(N+1)*(N)))
    v_vec = np.zeros((1,(N+1)*(N)))

    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vec:
            u_vec[0,i] = u(x,y)
            i+=1
    i=0
    for y in utils.y_vec:
        for x in utils.x_vecdual:
            v_vec[0,i] = v(x,y)
            i+=1

    numeric = lv.applyVlap(v_vec)
    numeric = numeric.reshape((N-1,N-2))
    
    exact = laplacian(m.x_meshV,m.y_meshV)[1:-1,1:-1]

    return np.mean((exact-numeric))**2

def PhiLaplacianConvergenceStudy(N,u,v,laplacian):
    m = meshgen.meshGenClass(N)
    lphi = laplacianoperator.Philaplacian(N,1/N)
    utils = solutiontools.utilsClass(N)

    phi_vec = np.zeros((1,(N)*(N)))

    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vecdual:
            phi_vec[0,i] = u(x,y)
            i+=1

    numeric = lphi.applyPhilap(phi_vec)
    numeric = numeric.reshape((N-2,N-2))
    
    exact = laplacian(m.x_meshdual,m.y_meshdual)[1:-1,1:-1]

    return np.mean((exact-numeric))**2


N_list = np.logspace(2,7,5,endpoint=False,base=2,dtype=int)




#N_list = [3]
errorlist1=[]
errorlist2=[]
errorlist3=[]

for N in N_list:
    errorlist1.append(ULaplacianConvergenceStudy(N,u,v,Ulaplacian))
    errorlist2.append(VLaplacianConvergenceStudy(N,u,v,Vlaplacian))
    errorlist3.append(PhiLaplacianConvergenceStudy(N,u,v,Ulaplacian))
    print(N)

plt.plot(N_list,errorlist1,'x',label="U Laplacian")
plt.plot(N_list,errorlist2,'.',label="V Laplacian")
plt.plot(N_list,errorlist3,'o',fillstyle='none',label="Phi Laplacian")
plt.plot(N_list,1e-3/N_list**4,label="Second Order Slope")

plt.legend()
plt.xlabel("1/h")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log')
plt.show()

# %%
