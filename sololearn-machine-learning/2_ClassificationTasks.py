# Classification is a stage of supervised learning.
# in Classification often the target value is true or false
# the thing we want to predict is called target and the other columns a feature

# Linear model for classification
# equation for the line
# ax+by+c = 0 - a,b,c are coefficients that decides where the line is
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(-5,5,100)
y = -2*x + 5
plt.plot(x, y, '-r', label='y')
plt.title('Graph of y')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
# plt.show()

# probability 
# the function is called sigmoid - e = 2.71828

# likelihood - how we score and compare possible choices

# prepare model
df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouces', 'Parents/Children',
'Fare']].values
y = df['Survived'].values
print(X)
print(y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,y)
print(model.coef_, model.intercept_) # 0 = 0.0161594x - 0.015490y - 0.51037

# prediction
# model.predict(X)
print(model.predict([[3,True,22.0,1,0,7.25]]))
print(model.predict(X[:5]))
print(y[:5])

# score the model
y_pred = model.predict(X)
y = y_pred
print((y==y_pred).sum())

# get total number of passenagers
passengers = y.shape[0]
print((y==y_pred).sum()/passengers) #accuracy
print(model.score(X,y)) #score = accuracy

# ================================================
# Brest cancer data
# ===============================================
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
# print(cancer_data.keys())
# print(cancer_data['DESCR'])

df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())

X = df[cancer_data.feature_names].values
y = df['target'].values

model = LogisticRegression(solver='liblinear')
model.fit(X,y)
print('prediction for datapoint 0 :',model.predict([X[0]]))
print(model.score(X,y))