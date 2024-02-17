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

lphi = laplacianoperator.Philaplacian(N)

print(lphi.PhiLap)
