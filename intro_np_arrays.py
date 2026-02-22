import numpy as np
"""
one_dim_arr = np.arange(1,20,3, dtype = float)
print(one_dim_arr)

lin_spaced_arr = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr)
"""
"""
char_arr = np.array((["Welcome to AI"],["Welcome to AI"]))
print(char_arr)
print(char_arr.dtype)

ones = np.ones(3)
print(ones)
zeros = np.zeros(3)
print(zeros)

new_arr = (3*ones)**2 + zeros

print(new_arr**2)
"""
"""
rand_arr = np.random.rand(3)
print(rand_arr)
"""
"""
two_dim = np.array(([1,2,3],[4,5,6],[7,8,9]))
print(two_dim)


arr = np.linspace(0,99,10)
arr = np.reshape(arr, (2,5))
print(arr)
ones_arr = np.ones(5)
print(ones_arr.ndim)
print(arr.shape)
print(arr.size)
"""
"""
arr_1 = np.linspace(1,10,4,dtype = int)
arr_2 = np.linspace(1,22,4, dtype= int )

arr_3 = arr_1 + arr_2
arr_4 = arr_2 - arr_1
arr_5 = arr_1 * arr_2

print(arr_3, arr_4 , arr_5)
"""
"""
arr = np.arange(1,19,3)
print(arr)

slicing_arr = np.arange(5,34,4)

print(slicing_arr[-2])

reshaped_arr = np.reshape(slicing_arr, (4,2))

print(reshaped_arr)

print(reshaped_arr[:,0])
"""

"""
arr_1 = np.reshape((np.linspace(10,100,10)), (2,5))
arr_2 = np.reshape(np.linspace(5,50,10), (2,5))

arr_3 = np.vstack((arr_1,arr_2))

print(arr_3)

x_1 = np.array([1,2,3,4,5])
y_1 = np.array([15,20,33,46,95])
"""

size = np.array([50, 60, 70])
bedrooms = np.array([1, 2, 2])
age = np.array([20, 15, 10])

X = np.c_[size, bedrooms, age]
print(X)


a = np.array([[1,2,5],[6,9,7]])
b = np.array([[1,2,3],[6,7,7]])

c = np.hstack((a,b))
print(c)