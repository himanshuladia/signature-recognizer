import os
import numpy as np
import scipy.io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pickle

train = np.load('signetDutchprobstrain67.npy')
trainf = np.load('signetf95Dutchprobstrain67.npy')

test = np.load('signetDutchprobstest33.npy')
testf = np.load('signetf95Dutchprobstest33.npy')

X_train = np.hstack([train, trainf])
X_test = np.hstack([test, testf])

y_train = np.load('Dutchtrainy67.npy')
y_test = np.load('Dutchtesty33.npy')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
