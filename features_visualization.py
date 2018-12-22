import os
import numpy as np
import scipy.io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import xgboost as xgb
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

reals = np.load('real1_features.npy')
forges = np.load('forge1_features.npy')

genuines = []
fakes = []

for i in range(len(reals)):
    genuines.append(reals[i].flatten())

for i in range(len(forges)):
    fakes.append(forges[i].flatten())

genuines = np.array(genuines)
fakes = np.array(fakes)

X_test = np.vstack([genuines, fakes])

y_test = np.zeros((720,))
for i in range(len(y_test)/2):
	y_test[i] = 1


reals1 = np.load('real2_features.npy')
forges1 = np.load('forge2_features.npy')

genuines1 = []
fakes1 = []

for i in range(len(reals1)):
    genuines1.append(reals1[i].flatten())

for i in range(len(forges1)):
    fakes1.append(forges1[i].flatten())

genuines1 = np.array(genuines1)
fakes1 = np.array(fakes1)

X_train = np.vstack([genuines1, fakes1])

y_train = np.zeros((300,))
for i in range(int(len(y_train)/2)):
	y_train[i] = 1

# tsne1 = TSNE(n_components=2, random_state=0)
# genuineReduced = tsne1.fit_transform(genuines)
#
# tsne2 = TSNE(n_components=2, random_state=0)
# forgedReduced = tsne2.fit_transform(fakes)
#
# X = np.vstack([genuineReduced, forgedReduced])
#
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier()
# clf = SVC()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# print(predictions)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# plt.scatter(genuineReduced[:,0], genuineReduced[:,1], color='black', label='genuine')
# plt.scatter(forgedReduced[:,0], forgedReduced[:,1], color='red', label='forged')
# plt.legend()
# plt.show()
