# nonparametric alghoritm
# root node, internal nodes, leaf nodes
# gini inpurity - calculates how much the set is pure( from 0 to 0.5 where 0 is 100% pure 0.5 is 50-50)
# gini = 2*p*(1-p) where p is passengers who survived and (1-p) is percent didn't survived
# split A
A_left = 2*(10/50)*(40/50)
print('A left:', A_left)
A_right = 2*(40/50)*(10/50)
print('A_right:', A_right)
# Split B
B_left = 2*(5/10)*(5/10)
print('B_left:', B_left)
B_right = 2*(45/90)*(45/90)
print('B right:', B_right)

# Entrophy - another measurement of purity - 0 -complete pure, 1- complete impure
# entrophy = -[p*log(p)+(1-p)*log(1-p)]

# information gain = H(S)-((|A|/|S|)*H(A))-((|B|/|S|)*H(B))
# H - impurity measure(eather gini or entropy)
# S - original dataset
# A,B - two sets spliting S into
# 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix
import numpy as np

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouces', 'Parents/Children',
'Fare']].values
y = df['Survived'].values

model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=22)

model.fit(X_train, y_train)

print(model.predict([[3, True, 22, 1, 0, 7.25]]))

kf = KFold(n_splits = 5, shuffle=True)
for criterion in ['gini', 'entropy']:
    print('Decision tree - {}'.format(criterion))
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print('accuracy:', np.mean(accuracy))
    print('precision:', np.mean(precision))
    print('recall:', np.mean(recall))

from sklearn.tree import export_graphviz
import graphviz 
from IPython.display import Image
feature_name = ['Pclass', 'male']
X = df[feature_name].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_name)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)

# to reduce overfitting we do pruning the tree - two types pre-pruning, post-pruning
# pre-pruning
# we are looking for parameters - max_depth, min_sample_leaf, max_leaf_nodes
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_name)
graph = graphviz.Source(dot_file)
graph.render(filename='tree_prune', format='png', cleanup=True)

from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':[5, 15, 25],
                'min_sample_leaf':[1, 3],
                'max_leaf_nodes':[10, 20, 35,  50]}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
dt.fit(X, y)

print('best params:', gs.best_params_)
print('best score:', gs.best_score_)