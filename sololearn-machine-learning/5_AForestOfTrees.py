# description:
# Feature matrix X and target array y-
# 1. Split the data into trainin, test sets
# 2. build a Random Forest with the training set, and make prediction for the test set
# 3. Give the random forest 5 trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rnd_state = 1
n_datapoints = 10

X = [[-1.53,-2.86],[-4.42,0.71],[-1.55,1.04],[-0.6,-2.01],[-3.43,1.5],[1.45,-1.15],
[-1.6,-1.52],[0.79,0.55],[1.37,-0.23],[1.23,1.72]]
# print(X)
y = [0,1,1,0,1,0,0,1,0,1]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=rnd_state)
print(X_test)
rf = RandomForestClassifier(n_estimators=5, random_state=rnd_state)
rf.fit(X_train,y_train)
print(rf.predict(X_test))