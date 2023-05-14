# Bootstraping - creating multiple samples
# bagging (Bootstrap aggregation) - to reduce the variance of the set
# Decorrelate the trees - 
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] =cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

first_row = X[0]
print('prediction:', rf.predict([first_row]))
print('true value:', y_test[0])
print('random forest accuracy:',rf.score(X_test, y_test))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("decision tree accurcy:", dt.score(X_test, y_test))

# tuning - n_estimators - count of trees, max_features - 
from sklearn.model_selection import GridSearchCV
# n_estimators = list(range(1,101))
# param_grid = {'n_estimators': n_estimators}
# rf = RandomForestClassifier()
# gs = GridSearchCV(rf, param_grid, cv=5)

# gs.fit(X,y)
# print("best params", gs.best_params_)

# scores = gs.cv_results_['mean_test_score']

import matplotlib.pyplot as plt

# plt.plot(n_estimators, scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xlim(0,100)
plt.ylim(0.9,1)
# plt.show()

# best model
rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)

# feature importance
ft_imp = pd.Series(rf.feature_importances_,
index=cancer_data.feature_names).sort_values(ascending=False)
print(ft_imp.head(10))

worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)

X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=101)

rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))