import numpy as np

def dot(a,b):
    sum_ = 0
    for ai , bi in zip(a,b):
        sum_ += (ai*bi)
    
    return sum_

a = np.array([4,5,6])
b = np.array([7,8,9])

print(dot(a, b))

print(np.dot(a,b))

print(a@b)


x = np.arange(0,101,10)
y = np.arange(0,51,5)
z = np.column_stack((x,y))
print(z)

d = np.stack((x,y), axis = 1)
print(d)