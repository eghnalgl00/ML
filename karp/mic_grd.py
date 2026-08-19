import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
"""
def f(x):
    return x**3 + 5*(x**2) + 3

xs = np.arange(-5,5,0.25)
ys = f(xs)

plt.plot(xs,ys)
"""
"""
a = 2.0
b = -3.0
c = 10.0
d = a*b + c


d1 = a*b+c
h = 0.00001
a += h
d2 = a*b + c

(d2-d1)/h #Slope
"""

def trace(root):
    # build the set of all nodes and edges in the graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        uid = str(id(n))
        # a record-shaped node showing label, data, and grad
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape='record'
        )
        if n._op:
            # separate node for the operation that produced n
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect child n1 to the op-node of its parent n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Value:
    def __init__(self,data,_children = () , _op = "" , label = ""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        out = Value(self.data + other.data , (self,other) , _op = "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out    

    def __mul__(self,other):
        out = Value(self.data * other.data , (self,other) , _op = "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1)/(math.exp(2*x) + 1)
        out = Value(t , (self, ) , "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

"""
def lol():

    h = 0.0001

    a = Value(2.0 , label = "a")
    #print(a.data) -> 2.0    
    b = Value(-3.0 , label = "b")
    c = a + b ; c.label = "c"  #Value(-1.0)
    d = a*b + c ; d.label = "d" #Value(-7.0)
    d._prev #(Value(-6.0) . Value(-1.0))
    d._op # "+"
    e = d * c ; e.label = "e"
    L = e * d ; L.label = "L"
    L.grad = 1.0
    e.grad = d.data ; d.grad = e.data
"""    

x1 = Value(2.0 , label = "x1")
x2 = Value(0.0 , label = "x2")
#Weights w1, w2
w1 = Value(-3.0 , label = "w1")
w2 = Value(1.0 , label = "w2")
#bias
b = Value(6.88134787584357857 , label = "b")

x1w1 = x1*w1 ; x1w1.label = "x1w1"
x2w2= x2*w2 ; x2w2.label = "x2w2"

x1w1x2w2 = x1w1 + x2w2 ; x1w1x2w2.label = "x1w1x2w2"
n = x1w1x2w2 + b ; label = "n"
o = n.tanh() ; o.label = "o"
"""
o.grad = 1.0 #do/do = 1
# tanh grad : do/dn = 1 - tanh(n)**2
n.grad = 0.5
x1w1x2w2 = 0.5 ; b.grad = 0.5 #addition -> grad = 1
x1w1.grad = 0.5 ; x2w2.grad = 0.5
x2.grad = w2.data * x2w2.grad ; w2.grad = x2.data * x2w2.grad
x1.grad = w1.data * x1w1.grad ;w1.grad = x1.data * x1w1.grad
"""
"""
o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
"""

o.backward()
