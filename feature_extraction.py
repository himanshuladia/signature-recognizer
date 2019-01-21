from scipy.misc import imread
from preprocess.normalize import remove_background
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import os
import numpy as np
from tqdm import tqdm

canvas_size = (952, 1360)

# Use the CNN to extract features
datasets = ['Chinese']

reals = []
forges = []

for dataset in datasets:
    realpath = 'sigcomp/testingSet/'+dataset+'/Ref/'
    items = os.listdir(realpath)
    for item in tqdm(items):
        person = os.path.join(realpath, item)
        signs = os.listdir(person)
        for sign in signs:
            img = os.path.join(person, sign)
            try:
                original = imread(img, flatten=1)
            except:
                pass
            processed = preprocess_signature(original, canvas_size)
            reals.append(processed)

for dataset in datasets:
    path = 'sigcomp/testingSet/'+dataset+'/Questioned/'
    items = os.listdir(path)
    for item in tqdm(items):
        person = os.path.join(path, item)
        signs = os.listdir(person)
        for sign in signs:
            img = os.path.join(person, sign)
            try:
                original = imread(img, flatten=1)
            except:
                pass
            processed = preprocess_signature(original, canvas_size)
            if len(sign)==10:
                reals.append(processed)
            else:
                forges.append(processed)

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

real_features = []
forge_features = []

for real in tqdm(reals):
    feature_vector = model.get_feature_vector(real)
    real_features.append(feature_vector)

for forge in tqdm(forges):
    feature_vector = model.get_feature_vector(forge)
    forge_features.append(feature_vector)

np.save('realChineseTestsignet_features.npy', real_features)
np.save('forgeChineseTestsignet_features.npy', forge_features)
