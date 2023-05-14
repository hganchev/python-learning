# Description:
# givven w1,w2,b,x1,x2 - compute the output y with sigmoid activation function
import numpy as np
w1 = 0
w2 = 1
b = 2
x1 = 1
x2 = 2

y=1/(1+np.exp(-(w1*x1 + w2*x2 + b)))
print(round(y,4))