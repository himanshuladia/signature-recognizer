from scipy.misc import imread
from preprocess.normalize import remove_background
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import os
import numpy as np
import

canvas_size = (952, 1360)

# Use the CNN to extract features
datasets = ['dataset1','dataset2','dataset3','dataset4']
# datasets = ['dataset1']

reals = []
forges = []

for dataset in datasets:
    realpath = 'Dataset/'+dataset+'/real/'
    # realpath = 'DatasetInitial/'+dataset+'/genuine/'
    items = os.listdir(realpath)
    # items.sort(key = lambda x: x[9:12])
    items.sort(key = lambda x: x[5:8])
    for item in items:
        img = os.path.join(realpath, item)
        try:
            original = imread(img, flatten=1)
        except:
            pass
        processed = preprocess_signature(original, canvas_size)
        reals.append(processed)

for dataset in datasets:
    forgepath = 'Dataset/'+dataset+'/forge/'
    # forgepath = 'DatasetInitial/'+dataset+'/forged/'
    items = os.listdir(forgepath)
    # items.sort(key = lambda x: x[9:12])
    items.sort(key = lambda x: x[5:8])
    for item in items:
        img = os.path.join(forgepath, item)
        try:
            original = imread(img, flatten=1)
        except:
            pass
        processed = preprocess_signature(original, canvas_size)
        forges.append(processed)

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

real_features = []
forge_features = []

for real in reals:
    feature_vector = model.get_feature_vector(real)
    real_features.append(feature_vector)

for forge in forges:
    feature_vector = model.get_feature_vector(forge)
    forge_features.append(feature_vector)

np.save('realtest_features.npy', real_features)
np.save('forgetest_features.npy', forge_features)
