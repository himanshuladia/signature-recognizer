import os
import numpy as np
import scipy.io
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import xgboost as xgb
import pickle


## Training
reals1 = np.load('realChinesesignetf95_features.npy')
forges1 = np.load('forgeChinesesignetf95_features.npy')

genuines1 = []
fakes1 = []

for i in range(len(reals1)):
    genuines1.append(reals1[i].flatten())

for i in range(len(forges1)):
    fakes1.append(forges1[i].flatten())

genuines1 = np.array(genuines1)
fakes1 = np.array(fakes1)

X = np.vstack([genuines1, fakes1])

y = np.hstack([np.ones((genuines1.shape[0],)), np.zeros((fakes1.shape[0],))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = xgb.XGBClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

np.save('Dutchtrainy67.npy', y_train)
np.save('Dutchtesty33.npy', y_test)

trainprobabilities = clf.predict_proba(X_train)
testprobabilities = clf.predict_proba(X_test)
np.save('signetf95Dutchprobstrain67.npy', trainprobabilities)
np.save('signetf95Dutchprobstest33.npy', testprobabilities)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
