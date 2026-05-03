import numpy as np
import matplotlib.pyplot as plt

x_1 = np.array([0.1,0.2,0.3])
x_2 = np.array([1,2,3])

x_train = np.c_[x_1,x_2]

def g(z):
    return 1/(1+np.exp(-z))

def dense(a_in , w ,b):
    units = w.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(a_in , w[:,j]) + b[j]
        a_out[j] = g(z)
    return a_out


def sequential(x):
    a1 = dense(x , w1 ,b1)
    a2 = dense(a1 , w2 ,b2)
    a3 = dense(a2 , w3 ,b3)
    f_x = a3
    return f_x


w1 = np.array([[1,2,3],[4,5,6]]) / 10000000
w2 = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]]) / 1000000
w3 = np.array([[1],[2],[3],[4]]) / 10000000
b1 = np.array([1,2,3]) 
b2 = np.array([1,2,3,4]) 
b3 = np.array([1]) 

for i, x in enumerate(x_train):
    print(f"Sample {i}: {sequential(x)}")