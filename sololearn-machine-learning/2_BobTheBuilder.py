import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

X = [[1,3],[3,5],[5,7],[3,1],[5,3],[7,5]]
y = [1, 1, 1, 0, 0, 0]

model = LogisticRegression()
model.fit(X,y)
y_pred = model.predict([[2,4]])
print(y_pred[0])
# print(model.score(X,y))