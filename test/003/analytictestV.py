#%%
import numpy as np
import scipy.linalg as sp
import os 
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import convectiveoperator,solutiontools,meshgen

N = 3

m = meshgen.meshGenClass(N)
cv = convectiveoperator.VconvectiveOperator(N,1/N)
cu = convectiveoperator.UconvectiveOperator(N,1/N)
t = solutiontools.utilsClass(N)

#print(cu.Uydiff)
#print(cv.Vydiff)

def Vconvergencestudy(N):
    mesh = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    Vconv = convectiveoperator.VconvectiveOperator(N,1/N)



    u_vec = np.zeros((1,(N+1)*(N)))
    v_vec = np.zeros((1,(N+1)*(N)))

    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vec:
            u_vec[0,i] = x**2+y**2
            i+=1
    i=0
    for y in utils.y_vec:
        for x in utils.x_vecdual:
            v_vec[0,i] = x**2-y**2
            i+=1

    #exact solution 
    exact = -((mesh.x_meshV**2+mesh.y_meshV**2)*2*mesh.x_meshV - (mesh.x_meshV**2-mesh.y_meshV**2)*2*mesh.y_meshV)[1:-1,1:-1]
    
    #exact = ((mesh.x_meshU**2-mesh.y_meshU**2)*2*mesh.y_meshU)[1:-1,1:-1]

    #utils.plotgivenmesh(mesh.x_meshU[1:-1,1:-1],mesh.y_meshU[1:-1,1:-1],exact[1:-1,1:-1])

    #using convective operator
    #print(np.shape(utils.uinterp(u_vec)))
    conv_vec = Vconv.apply(utils.uinterp(u_vec),v_vec)

    numeric = conv_vec.reshape(((N-1),(N-2)))

    #print(numeric)
    #print(exact)
    return np.mean((exact-numeric))**2

def Uconvergencestudy(N):
    mesh = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    Uconv = convectiveoperator.UconvectiveOperator(N,1/N)



    u_vec = np.zeros((1,(N+1)*(N)))
    v_vec = np.zeros((1,(N+1)*(N)))

    i=0
    for y in utils.y_vecdual:
        for x in utils.x_vec:
            u_vec[0,i] = x**2-y**2
            i+=1
    i=0
    for y in utils.y_vec:
        for x in utils.x_vecdual:
            v_vec[0,i] = x**2+y**2
            i+=1

    #exact solution 
    exact = -((mesh.x_meshU**2-mesh.y_meshU**2)*2*mesh.x_meshU - (mesh.x_meshU**2+mesh.y_meshU**2)*2*mesh.y_meshU)[1:-1,1:-1]
    #exact = exact/(2*1/N)
    #exact = ((mesh.x_meshU**2-mesh.y_meshU**2)*2*mesh.y_meshU)[1:-1,1:-1]

    #utils.plotgivenmesh(mesh.x_meshU[1:-1,1:-1],mesh.y_meshU[1:-1,1:-1],exact[1:-1,1:-1])

    #using convective operator
    #print(np.shape(utils.uinterp(u_vec)))
    conv_vec = Uconv.apply(u_vec,utils.vinterp(v_vec))

    numeric = conv_vec.reshape(((N-2),(N-1)))

    #print(numeric)
    #print(exact)
    return np.mean((exact-numeric))**2

N_list = np.logspace(2,7,5,endpoint=False,base=2,dtype=int)
#print(np.shape(cv.Uyinterp))



#N_list = [3]
errorlist1=[]
errorlist2=[]
for N in N_list:
    errorlist1.append(Vconvergencestudy(N))
    errorlist2.append(Uconvergencestudy(N))
    print(N)


plt.plot(N_list,errorlist2,'.',label="U Convective")
plt.plot(N_list,errorlist1,'x',label="V Convective")
plt.plot(N_list,1e-2/N_list**4,label="Second Order Slope")


plt.legend()
plt.xlabel("1/h")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log')
plt.show()


# %%
