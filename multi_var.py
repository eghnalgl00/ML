import numpy as np
import matplotlib.pyplot as plt
import math

size = np.array([
    40, 45, 50, 55, 60,
    65, 70, 75, 80, 85,
    90, 95, 100, 105, 110,
    115, 120, 125, 130, 135
])

bedrooms = np.array([
    1, 1, 2, 2, 2,
    2, 3, 3, 3, 3,
    3, 4, 4, 4, 4,
    4, 5, 5, 5, 5
])

age = np.array([
    30, 28, 25, 23, 20,
    18, 15, 13, 12, 10,
    9, 8, 7, 6, 5,
    4, 3, 2, 2, 1
])

x_train = np.c_[size, bedrooms, age]

y_train = (
    2.5 * size
    + 15 * bedrooms
    - 1.2 * age
    + 10
).astype(float)
mean = np.mean(x_train, axis=0) 
std  = np.std(x_train, axis=0)  
x_min = np.min(x_train, axis=0)
x_max = np.max(x_train, axis=0)

norm_x = (x_train - mean) / std
norm_x_2 = (x_train - mean) / (x_max - x_min)



def compute_cost(x,y,w,b,lmd):
    m = x.shape[0]
    f_wb = 0
    total_cost = 0
    reg_term = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        total_cost += (f_wb-y[i])**2
    for i in range(len(x[0])):
        reg_term += w[i]**2
    total_cost += lmd*reg_term
    total_cost = total_cost/(2*m)
    return total_cost

def compute_gradient(x,y,w,b,lmd):
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
    for i in range(len(dj_dw)):
        dj_dw[i] += (lmd/m)*w[i]
    return dj_dw , dj_db

def gradient_descent(x,y,w,b,alpha,iters):
    cost_hist = []
    w_b_hist = []
    dw_db_hist = []
    for i in range(iters):
        dj_dw , dj_db = compute_gradient(x,y,w,b,lmd)
        cost_hist.append(compute_cost(x, y, w, b,lmd))
        w_b_hist.append((w.copy(),b))
        dw_db_hist.append([dj_dw,dj_db])
        w = w - alpha*dj_dw
        b = b - alpha*dj_db 
        if np.linalg.norm(dj_dw) < 1e-20 and abs(dj_db) < 1e-20:
            break
        if cost_hist[i] > cost_hist[i-1]:
            alpha *= 0.5 
    n = cost_hist.index(min(cost_hist))
    return w_b_hist[n] , min(cost_hist) , len(cost_hist) , alpha , cost_hist , dw_db_hist

w = np.array([6,6,6])
b = 19
alpha = 3e-1
iters = 10000
lmd = 0.6

best_wb, best_cost, steps , alpha_final ,cost_hist , dw_db_hist= gradient_descent(norm_x, y_train, w, b, alpha, iters)
w_1, b_1 = best_wb

print(best_wb, best_cost, steps , alpha_final)

def predict_raw(x_raw):
    x_raw = np.array(x_raw, dtype=float)
    x_norm = (x_raw - mean) / std
    return np.dot(w_1, x_norm) + b_1


print(predict_raw([40,1,30]))