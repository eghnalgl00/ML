import numpy as np
import matplotlib.pyplot as plt
import math
"""
#training set
x_train = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
y_train = np.array([15,40,65])
mean = np.mean(x_train, axis=0) 
std  = np.std(x_train, axis=0)  
x_min = np.min(x_train, axis=0)
x_max = np.max(x_train, axis=0)

norm_x = (x_train - mean) / std
norm_x_2 = (x_train - mean) / (x_max - x_min)
"""

size = np.array([50, 60, 70, 80, 90])
bedrooms = np.array([1, 2, 2, 3, 3])
age = np.array([20, 15, 10, 5, 2])

x_train = np.c_[size, bedrooms, age]
y_train = np.array([55, 100, 135, 180, 209], dtype=float)
mean = np.mean(x_train, axis=0) 
std  = np.std(x_train, axis=0)  
x_min = np.min(x_train, axis=0)
x_max = np.max(x_train, axis=0)

norm_x = (x_train - mean) / std
norm_x_2 = (x_train - mean) / (x_max - x_min)



def compute_cost(x,y,w,b):
    m = x.shape[0]
    f_wb = 0
    total_cost = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        total_cost += (f_wb-y[i])**2
    total_cost = total_cost/(2*m)
    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = np.zeros(len(x[0]))
    dj_db = 0
    f_wb = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        for j in range(len(dj_dw)):
            dj_dw[j] += (f_wb-y[i])*x[i][j]
        dj_db += (f_wb-y[i])
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw , dj_db

def gradient_descent(x,y,w,b,alpha,iters):
    cost_hist = []
    w_b_hist = []
    dw_db_hist = []
    for i in range(iters):
        dj_dw , dj_db = compute_gradient(x,y,w,b)
        cost_hist.append(compute_cost(x, y, w, b))
        w_b_hist.append((w,b))
        dw_db_hist.append([dj_dw,dj_db])
        w = w - alpha*dj_dw
        b = b - alpha*dj_db 
        if np.linalg.norm(dj_dw) < 1e-20 and abs(dj_db) < 1e-20:
            break
        if cost_hist[i] > cost_hist[i-1]:
            alpha *= 0.5 
    n = cost_hist.index(min(cost_hist))
    return w_b_hist[n] , min(cost_hist) , len(cost_hist) , alpha , cost_hist , dw_db_hist

w = np.array([600,600,600])
b = 19
alpha = 3e-1
iters = 10000

best_wb, best_cost, steps , alpha_final ,cost_hist , dw_db_hist= gradient_descent(norm_x, y_train, w, b, alpha, iters)
w_1, b_1 = best_wb

print(best_wb, best_cost, steps , alpha_final)
def predict(x):
    out= np.dot(w_1,x) + b_1
    return out

print(predict([75, 2, 12]))

plt.plot(cost_hist)
plt.grid(True)
plt.show()
