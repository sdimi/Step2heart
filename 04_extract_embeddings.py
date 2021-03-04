#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
from keras import *
from keras import backend as K
from keras.models import  model_from_json, Model
import glob
from tqdm import tqdm
from utils import *

K.set_learning_phase(0)


#folder of the best pre-trained model (or replace with own)
folder = '20200115-105719' 

X_train_timeseries_normalized = np.load('data/X_train_timeseries_normalized.npy')
X_train_timeseries_normalized = X_train_timeseries_normalized[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)

X_train_demographics_normalized = np.load('data/X_train_demographics_normalized.npy')

X_train_demographics_normalized_resting = X_train_demographics_normalized[:,0]
X_train_demographics_normalized_temporal = X_train_demographics_normalized[:,-4:]


X_test_timeseries_normalized = np.load('data/X_test_timeseries_normalized.npy')  
X_test_timeseries_normalized = X_test_timeseries_normalized[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)

X_test_demographics_normalized = np.load('data/X_test_demographics_normalized.npy')
X_test_demographics_normalized_resting = X_test_demographics_normalized[:,0]
X_test_demographics_normalized_temporal = X_test_demographics_normalized[:,-4:]


model, loss = modular_model(X_train_timeseries_normalized.shape[2], 4, "quantile") #for a quantile model
#model, loss = modular_model(X_train_timeseries_normalized.shape[2], 5, "mse") #for vanilla mse

#load best model from the folder
model = model_from_json(open('./models/'+ folder +'/model_architecture.json').read()) #custom_objects={'exp': Activation(exponential)})
files = glob.glob('./models/'+folder+'/*.hdf5')

#WARNING! DEPENDS ON THE NAMING CONVENTION!
#we parse the filename ('./models/20191029-125927/weights-regression-improvement-51.03.hdf5')
#and we convert the MSE to a float in order to sort, the first is the lowest (lowest val_loss)
weights = sorted(files, key=lambda name: float(name[56:-5]))[0]
print ("=============Best model loaded:", weights)

model.load_weights(weights)   
model.compile(loss=[loss], optimizer="adam")
model.summary()

print (model.layers)
print (model.input) #input layers (if multimodal, >1)
print (model.layers[-2:][0].output) #embeddings/penultimate layer (architecture-dependent)

#step2heart (A/R/T)
layer_activations = K.function([*model.input, K.learning_phase()],
                                  [model.layers[-2:][0].output])

#a unimodal model (e.g. the autoencoder) can use a simpler input (no *)
#layer_activations = K.function([model.input, K.learning_phase()],
#                                 [model.layers[5].output])

len(X_test_timeseries_normalized), len(X_train_timeseries_normalized)


activations_test=[]
for i in tqdm(range(0,len(X_test_timeseries_normalized), 197)):  #197 (step) is a proper divisor of 22458 (test set size)
    #we need to pass each input tensor individually
    activations_test.append(layer_activations([X_test_timeseries_normalized[i:i+197],
                                                X_test_demographics_normalized_temporal[i:i+197],
                                          np.expand_dims(X_test_demographics_normalized_resting[i:i+197], axis=1),
                                         ]))

activations_train=[]
for i in tqdm(range(0,len(X_train_timeseries_normalized), 192)):  #192 (step) is a proper divisor of 88320 (train set size)
    #we need to pass each input tensor individually
    activations_train.append(layer_activations([X_train_timeseries_normalized[i:i+192],
                                                X_train_demographics_normalized_temporal[i:i+192],
                                          np.expand_dims(X_train_demographics_normalized_resting[i:i+192], axis=1),
                                         ]))

print (np.squeeze(np.array(activations_train)).shape, np.squeeze(np.array(activations_test)).shape)

#[loops, steps, activation_dim] -> [loops*steps, activation_dim]
reshaped_arr_train = np.squeeze(np.array(activations_train)).reshape(len(X_train_timeseries_normalized), np.array(activations_train).shape[3]) #the dim of the output layer
reshaped_arr_train.shape

#[loops, steps, activation_dim] -> [loops*steps, activation_dim]
reshaped_arr_test = np.squeeze(np.array(activations_test)).reshape(len(X_test_timeseries_normalized), np.array(activations_test).shape[3]) #the dim of the output layer
reshaped_arr_test.shape


np.save('data/X_train_activations'+ folder +'.npy', reshaped_arr_train)  
np.save('data/X_test_activations'+ folder +'.npy', reshaped_arr_test)

