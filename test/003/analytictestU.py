import numpy as np
import scipy.linalg as sp
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import convectiveoperator,solutiontools,meshgen

N = 3

m = meshgen.meshGenClass(N)
c = convectiveoperator.UconvectiveOperator(N,1/N)
t = solutiontools.utilsClass(N)


def calcsol(x,y):
    return -(x**2-y**2)*2*x+(x**2+y**2)*2*y
def usol(x,y):
    return x**2-y**2
def vsol(x,y):
    return x**2+y**2

u_vec = np.zeros((1,(N+1)*(N)))
v_vec = np.zeros((1,(N+1)*(N)))
i=0
for y in m.y_vecdual:
    for x in m.x_vec:
        u_vec[0,i]=usol(x,y)
        i+=1
i=0
for y in m.y_vec:
    for x in m.x_vecdual:
        v_vec[0,i]=vsol(x,y)
        i+=1
#print(c.apply(u_vec,t.vinterp(v_vec)))
#print(vsol(1/6,1/3))
#print(v_vec)
#print(t.vinterp(v_vec))
#print(calcsol(m.x_meshU,m.y_meshU))
#print(vsol(1/6,1/3))

