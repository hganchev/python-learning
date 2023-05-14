import numpy as np
# what is the ouput
x = np.arange(3,9)
z = x.reshape(2,3)
print(z[1][1])

x = np.arange(1,5)
x = x*2
print(x[:3].sum())


