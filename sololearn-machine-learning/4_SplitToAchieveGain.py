import numpy as np
# description:
# calculate the information gain using gini impurity
S = [1, 0, 1, 0, 1, 0]
A = [1, 1, 1]
B = [0, 0, 0]

# information gain = H(S)-((|A|/|S|)*H(A))-((|B|/|S|)*H(B))
# H - impurity measure(eather gini or entropy)
# S - original dataset
# A,B - two sets spliting S into
gini_S = (2*np.mean(S)*(1-np.mean(S)))
gini_A = (2*np.mean(A)*(1-np.mean(A)))
gini_B = (2*np.mean(B)*(1-np.mean(B)))
AS = len(A)/len(S)  #|A|/|S|
BS = len(B)/len(S)  #|B|/|S|
inf_gain = gini_S - AS * gini_A - BS * gini_B
print(round(inf_gain,5))
#  second case
S = [1, 0, 1, 0, 1, 0, 1, 0]
A = [1, 0, 1, 0]
B = [1, 0, 1, 0]

gini_S = (2*np.mean(S)*(1-np.mean(S)))
gini_A = (2*np.mean(A)*(1-np.mean(A)))
gini_B = (2*np.mean(B)*(1-np.mean(B)))
AS = len(A)/len(S)
BS = len(B)/len(S)
inf_gain = gini_S - AS * gini_A - BS * gini_B
print(round(inf_gain,5))

