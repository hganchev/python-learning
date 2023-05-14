# Confusion Matrix - has 4 squares 
# - true positive(TP), true negative(TN), false positive(FP), false negative(FN)
# predicted/total = score
# precision = TP/(TP+FP)
prec = 30/(30+20)
print(prec)
# recall = TP/(TP+FN)
recall = 30/(30+10)
print(recall)
# F1 - average between precision and recall
# = 2*(precision*recall/(precision+recall))
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouces', 'Parents/Children',
'Fare']].values
y = df['Survived'].values

# ===================================================================
# model = LogisticRegression()
# model.fit(X,y)

# y_pred = model.predict(X)
# print('accuracy:',accuracy_score(y,y_pred))
# print('precission:',precision_score(y,y_pred))
# print('recall:',recall_score(y,y_pred))
# print('F1 score:',f1_score(y,y_pred))

# print('confusion matrix:',confusion_matrix(y,y_pred))

# #training and testing
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# print('whole dataset:', X.shape, y.shape)
# print('training set:', X_train.shape, y_train.shape)
# print('test set:', X_test.shape, y_test.shape)

# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print('accuracy:',accuracy_score(y_test,y_pred))
# print('precission:',precision_score(y_test,y_pred))
# print('recall:',recall_score(y_test,y_pred))
# print('F1 score:',f1_score(y_test,y_pred))

# # giving a random state - to ensure we get the same split
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 27)
# print("X train:", X_train)
# print("X test:", X_test)

# # sensitivity = recall = TP/(TP+FN)
# # specificity = TN/(TN+FP)
# sens = 30/(30+10)
# spec = 40/(40+20)
# print(sens)
# print(spec)
# from sklearn.metrics import precision_recall_fscore_support

# sensitivity_score = recall_score
# def specificity_score(y_true, y_pred):
#     p, r, f, s = precision_recall_fscore_support(y_true,y_pred)
#     return r[0]

# print('sensitivity:', sensitivity_score(y_test, y_pred))
# print('specificity:', specificity_score(y_test, y_pred))

# # predict probability
# print('predict proba: ', model.predict_proba(X_test))

# y_pred = model.predict_proba(X_test)[:,1]>0.75

# print('precision:', precision_score(y_test,y_pred))
# print('recall:', recall_score(y_test,y_pred))
# ===================================================================
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# # roc curve
# model = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 27)
# model.fit(X_train, y_train)
# y_pred_proba = model.predict_proba(X_test)
# fpr, tpr, tresholds = roc_curve(y_test,y_pred_proba[:,1])

# plt.plot(fpr, tpr)
# plt.plot([0,1], [0,1], linestyle = '--')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xlabel('1-specificity')
# plt.ylabel('sensitivity')
# plt.show()
# # ROC curve is showing the performance on many models
# from sklearn.metrics import roc_auc_score
# # X_train, X_test, y_train, y_test = train_test_split(X,y)

# model1 = LogisticRegression()
# model1.fit(X_train, y_train)
# y_pred_proba1 = model1.predict_proba(X_test)
# print('model 1 AUC score', roc_auc_score(y_test, y_pred_proba1[:,1]))

# model2 = LogisticRegression()
# model2.fit(X_train[:,0.2],y_train)
# y_pred_proba2 = model2.predict_proba(X_test[:,0.2])
# print('model 2 AUC score', roc_auc_score(y_test, y_pred_proba2[:,1]))
# ===================================================================
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# # building the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Evaluating the model
# y_pred = model.predict(X_test)
# print('accuracy: {0:.5f}'.format(accuracy_score(y_test, y_pred)))
# print('precision: {0:.5f}'.format(precision_score(y_test, y_pred)))
# print('recall: {0:.5f}'.format(recall_score(y_test, y_pred)))
# print('f1 score: {0:.5f}'.format(f1_score(y_test, y_pred)))

# ===================================================================
# k-fold
from sklearn.model_selection import KFold
import numpy as np
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

print('X')
print(X)

kf = KFold(n_splits=3, shuffle=True)
# for train, test in kf.split(X):
#     print(train,test)

splits = list(kf.split(X))
first_split = splits[0]
print('first_split:', first_split)
train_indices, test_indices = first_split
print('training set indices:', train_indices)
print('test set indices:', test_indices)
scores = []
for train_indices, test_indices in kf.split(X):
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    print('X_train')
    print(X_train)
    print('y_train',y_train)
    print('X_test')
    print(X_test)
    print('y_test', y_test)    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    scores.append(model.score(X_test,y_test))

print(scores)
print(np.mean(scores))

final_model = LogisticRegression()
final_model.fit(X, y)