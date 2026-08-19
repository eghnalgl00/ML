import numpy as np

def dense(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        out = 1/(1+np.exp(-z))
        a_out[j] = out
    return a_out


def sequential(x):
    a_1 = dense(x,W_1,b_1)
    a_2 = dense(a_1,W_2,b_2)
    a_3 = dense(a_2,W_3,b_3)
    a_4 = dense(a_3,W_4,b_4)
    f_x = a_4
    return f_x


W_1 = np.array([[1,1,1],[2,2,2]])
W_2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
W_3 = np.array([[1,1,1],[2,2,2],[3,3,3]])
W_4 = np.array([[1,1,1],[2,2,2],[3,3,3]])
b_1 = np.array([1,1,1])
b_2 = np.array([1,1,1])
b_3 = np.array([1,1,1])
b_4 = np.array([1,1,1])
x = np.array([5,10])
print(sequential(x))





