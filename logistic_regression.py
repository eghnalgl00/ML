import numpy as np
import matplotlib.pyplot as plt


x_1 = np.array([1,1,1,3,3,3])
x_2 = np.array([1,2,3,1,2,3])
y = np.array([0,1,0,1,1,0])

x_train = np.c_[x_1,x_2]

def z(x,w,b):
    wx_b = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        wx_b[i] = np.dot(w,x[i]) + b
    
    return wx_b

def g(x):
    return 1/(1+np.exp(-x))

w = np.array([0.3,-0.3])
b = 1


def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = g(z(x, w, b))
    total_cost = 0.0
    for i in range(m):
        L = -y[i] * np.log(f_wb[i]) - (1 - y[i]) * np.log(1 - f_wb[i])
        total_cost += L
    return total_cost / m


def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = np.zeros(len(x[0]))
    dj_db = 0
    f_wb = g(z(x,w,b))
    for i in range(m):
        dj_db += (f_wb[i]-y[i])
        for j in range(len(x[0])):
            dj_dw[j] += (f_wb[i]-y[i])*x[i][j]
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
        w_b_hist.append((w.copy(), float(b)))
        dw_db_hist.append((dj_dw.copy(), float(dj_db)))
        w = w - alpha*dj_dw
        b = b - alpha*dj_db 
        if np.linalg.norm(dj_dw) < 1e-20 and abs(dj_db) < 1e-20:
            break
        if i > 0 and cost_hist[i] > cost_hist[i-1]:
            alpha *= 0.5 
    n = cost_hist.index(min(cost_hist))
    return w_b_hist[n] , min(cost_hist) , len(cost_hist) , alpha , cost_hist , dw_db_hist

alpha = 1e-3
iters = 10000


best_wb, best_cost, steps , alpha_final ,cost_hist , dw_db_hist= gradient_descent(x_train, y, w, b, alpha, iters)
w_1, b_1 = best_wb

print(best_wb, best_cost, steps , alpha_final)