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

lu = laplacianoperator.Ulaplacian(N)
lv = laplacianoperator.Vlaplacian(N)

#print(lu.ULap) #coherent
#print(lv.VLap) #coherent

#analytic solution

def convergencestudy(N):
    m = meshgen.meshGenClass(N)
    utils = solutiontools.utilsClass(N)
    lu = laplacianoperator.Ulaplacian(N)
    lv = laplacianoperator.Vlaplacian(N)

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

    exactU = (-2*np.sin(m.x_meshU)*np.cos(m.y_meshU))[1:-1,1:-1]
    exactV = (-2*np.cos(m.x_meshV)*np.sin(m.y_meshV))[1:-1,1:-1]

    

    Ulap = lu.applyUlap(u_vec)
    Vlap = lv.applyVlap(v_vec)

    numericU = Ulap.reshape(((N-2),(N-1))) #shape of valid points
    numericV = Vlap.reshape(((N-1),(N-2))) #shape of valid points

    return np.mean((exactU-numericU))**2,np.mean((exactV-numericV))**2

N_list = np.logspace(2,7,5,endpoint=False,base=2,dtype=int)

#N_list = [3]
errorlistU = []
errorlistV = []
for N in N_list:
    uerr,verr = convergencestudy(N)
    errorlistU.append(uerr)
    errorlistV.append(verr)
    print(N)



plt.plot(-np.log(1/N_list),errorlistU,'.')
plt.plot(-np.log(1/N_list),errorlistV,'x')
plt.xlabel("-log(h)")
plt.ylabel("MSE")
plt.yscale('log')
plt.show()
