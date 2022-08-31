from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'

kf = KFold(n_splits=5, shuffle=True)

X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

def score_model(X, y, kf):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    #print("accuracy:", np.mean(accuracy_scores))
    #print("precision:", np.mean(precision_scores))
    #print("recall:", np.mean(recall_scores))
    #print("f1 score:", np.mean(f1_scores))

#print("Logistic Regression with all features")
'''
Logistic Regression with all features
accuracy: 0.795937281787596
precision: 0.7594807509618959
recall: 0.6943938138635736
f1 score: 0.7224972933238585
'''
#score_model(X1, y, kf)
#print()


#print("Logistic Regression with Pclass, Sex & Age features")
'''
Logistic Regression with Pclass, Sex & Age features
accuracy: 0.7982669967625214
precision: 0.7528839672604719
recall: 0.7066721799978196
f1 score: 0.7287454387923968
'''

#score_model(X2, y, kf)
#print()


#print("Logistic Regression with Fare & Age features")
'''
Logistic Regression with Fare & Age features
accuracy: 0.6515647813114962
precision: 0.6254838709677419
recall: 0.23225409876135528
f1 score: 0.33770214004073396
'''
#score_model(X3, y, kf)


#using model with all the features
model = LogisticRegression()
model.fit(X1, y)
#print(model.predict([[3, False, 25, 0, 1, 2]]))

#load pickle file
pickle.dump(model,open('model.pkl', 'wb'))