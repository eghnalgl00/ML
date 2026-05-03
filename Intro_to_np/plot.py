import numpy as np
import matplotlib.pyplot as plt 
import math 

def f(x):
    return x **3 + 5

x = np.arange(-700,700,100)
print(x)
y = f(x)
print(y)
plt.scatter(x,y)
plt.plot(x,y)
plt.grid()
plt.axhline()
plt.axvline()
plt.show()