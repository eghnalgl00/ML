import numpy as np
import matplotlib.pyplot as plt 
import math

#training set
x_train = np.array([100000,200,300000,4])
y_train = np.array([500000,1000,1500000,20])
mean = np.mean(x_train)
min_x = np.min(x_train)
max_x = np.max(x_train)
std_x = np.std(x_train)
norm_x = (x_train - mean) / std_x
print(norm_x)

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    f_wb = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost += (y[i]-f_wb)**2
    total_cost = 1/(2*m)*cost

    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    f_wb = 0
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw += (f_wb-y[i])*x[i]
        dj_db += (f_wb-y[i])
    dj_dw = 1/(m)*dj_dw
    dj_db = 1/(m)*dj_db

    return dj_dw,dj_db


def gradient_descent(x, y, w, b, alpha, num_iters):
    cost_hist = []
    w_b_hist = []
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        cost_hist.append(compute_cost(x, y, w, b))
        w_b_hist.append((w,b))     
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if dj_dw  == 0 and dj_db == 0 :
            break
    n = cost_hist.index(min(cost_hist))
    return w_b_hist[n] , min(cost_hist) , len(cost_hist)

w = 34
b = 456
iters = 1000000
alpha = 0.01

best_wb, best_cost, steps= gradient_descent(norm_x, y_train, w, b, alpha, iters)
w_1, b_1 = best_wb


def predict(x):
    out= w_1 * x + b_1
    return out

print(predict(-0.1))
