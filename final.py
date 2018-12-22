import os
import numpy as np
import scipy.io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import pickle
from scipy.misc import imread
from preprocess.normalize import remove_background
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import os
import numpy as np


def probability(destination):

    img = imread(destination, flatten=1)

    # Load the model
    model_weight_path = 'models/signet.pkl'
    model = CNNModel(signet, model_weight_path)

    canvas_size = (952, 1360)
    processedImg = preprocess_signature(img, canvas_size)

    featuresImg = model.get_feature_vector(processedImg)

    features = featuresImg
    filename = 'xgboostmodelstage2train.sav'
    clf = pickle.load(open(filename, 'rb'))

    prediction = clf.predict_proba(features)

    return prediction


# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))

# plt.scatter(genuineReduced[:,0], genuineReduced[:,1], color='black', label='genuine')
# plt.scatter(forgedReduced[:,0], forgedReduced[:,1], color='red', label='forged')
# plt.legend()
# plt.show()
