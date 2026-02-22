import numpy as np
import matplotlib.pyplot as plt
"""
A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(A.ndim)
print(b.ndim)

x = np.linalg.solve(A,b)
print(x)

d = np.linalg.det(A)

print(d)

c = np.array(([1,1,1],[0,1,0],[0,0,1]))
det_c = np.linalg.det(c)
print(det_c)

A_system = np.hstack((A, b.reshape(2,1)))
print(A_system[:,1])


"""
"""
arr = np.array([[1,2],[2,1]])
b = np.array([3,0])
det_A = np.linalg.det(arr)
soln = np.linalg.solve(arr,b.reshape(2,1))

print(soln)
"""

"""
A = np.array([[4,-3,1],[2,1,3],[-1,2,-5]])
b = np.array([-10,0,17])

x = np.linalg.solve(A, b.reshape(3,1))
print(x)
print(f"{np.linalg.det(A):.5f}")
"""



A_2 = np.array([[1,1,1],[0,0,0],[2,2,2]])

b_2 = np.array([2,1,0])

print(np.linalg.det(A_2))

print(np.linalg.solve(A_2,b_2))
