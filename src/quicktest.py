#%%
import numpy as np
import scipy as sp

import momentumstep
import pressurestep
import divergenceoperator
import solutiontools
from plotsolution import plotx,ploty,plotp
import copy

N=3
test = solutiontools.utilsClass(N)
u_int,v_int = test.genPearSol(0)
phi_int = test.genPearPhi(0)
test.applyPHIBCs(phi_int)

# %%
