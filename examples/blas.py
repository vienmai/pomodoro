import numpy as np
import scipy 
from scipy.linalg import blas
import time
from timeit import timeit
# np.__config__.show()

def test_blas():
    m = 20000
    n = 20000
    x = np.random.randn(n)
    y = np.random.randn(m)
    A = np.random.randn(m, n)
    def rundot():
        np.dot(A, x)
    def rungemv():
        blas.dgemv(alpha=1.0, a=A.T, x=x, trans=True)
    
    tdot = timeit(rundot, number=100)
    print(1000*tdot/100)
    tblas = timeit(rungemv, number=100)
    print(1000*tblas/100) #ms
    
if __name__ == "__main__":
   test_blas() 
