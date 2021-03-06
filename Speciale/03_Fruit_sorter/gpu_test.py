"""
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# normal function to run on cpu
def func(a):							
	for i in range(10000000):
		a[i]+= 1	

# function optimized to run on gpu
@cuda.jit						
def func2(a):
	for i in range(10000000):
		a[i]+= 1
"""

import torch


if __name__=="__main__":

    #print(torch.cuda.current_device())
    print(torch.cuda.is_available())


    """
    n = 10000000
    a = np.ones(n, dtype = np.float64)
    b = np.ones(n, dtype = np.float32)
    start = timer()
    func(a)
    print("without GPU:", timer()-start)

    start = timer()
    func2(a)
    print("with GPU:", timer()-start)
    """
