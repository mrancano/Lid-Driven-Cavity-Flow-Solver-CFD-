#%%
#testing the mesh generator
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import meshgen

N = 4

meshgenerator = meshgen.meshGenClass(N)
##h = 1/N
#print(meshgenerator.h)
##genXYvec
x_vec,y_vec = meshgenerator.genXYvec()
#print(x_vec)
#print(y_vec)
##genXYmesh
x_mesh,y_mesh = meshgenerator.genXYmesh()
z_mesh = np.sin(5*np.sqrt(x_mesh**2+y_mesh**2))
#h = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(x_mesh,y_mesh,z_mesh,50)
#plt.show()
##genXYvecdual
x_vecdual,y_vecdual = meshgenerator.genXYvecdual()
#print(x_vecdual,y_vecdual)
##genXYmeshdual
x_meshdual,y_meshdual = meshgenerator.genXYmeshdual()
#print(y_meshdual)
#z_meshdual = np.sin(5*np.sqrt(x_meshdual**2+y_meshdual**2))
#hdual = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(x_meshdual,y_meshdual,z_meshdual,50)
#plt.show()
##genUmesh
x_meshU , y_meshU = meshgenerator.genUmesh()
#print(y_meshU)
##genVmesh
x_meshV , y_meshV = meshgenerator.genVmesh()
#print(x_meshV)
#%%
plt.plot(x_mesh,y_mesh,'b.')
plt.plot(x_meshdual,y_meshdual,'rx')
plt.plot(x_meshU,y_meshU,'gs')
plt.plot(x_meshV,y_meshV,'yo')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

# %%
