#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(14, 7) #set all figures size
import scipy
import seaborn as sns
from tqdm import tqdm
import time
import argparse
import glob
import os
import keras
from keras.utils import plot_model
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model, model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import CuDNNGRU,GlobalAveragePooling1D, Dense, Embedding, Masking, TimeDistributed, Dropout, Flatten, LSTM, GRU, Bidirectional, Activation, RepeatVector,InputLayer
from keras.optimizers import Adam, Nadam

from utils import *

parser = argparse.ArgumentParser(description='NN training.')
parser.add_argument('-m', '--modality', metavar='number', default=1, type=int) #1=acc, 2=acc+resting, 3= acc+time, 4=all
parser.add_argument('-l', '--loss', metavar='string', help='folder', type=str) #MSE or quantile

args = parser.parse_args()

X_train_timeseries_normalized = np.load('data/X_train_timeseries_normalized.npy')
X_train_timeseries_normalized = X_train_timeseries_normalized[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)

X_train_demographics_normalized = np.load('data/X_train_demographics_normalized.npy')


X_train_demographics_normalized_resting = X_train_demographics_normalized[:,0]
X_train_demographics_normalized_temporal = X_train_demographics_normalized[:,-4:]

y_train = np.load('data/y_train.npy',allow_pickle=True) 

print (X_train_timeseries_normalized.shape, X_train_demographics_normalized.shape, y_train.shape)

batch_size = 256

model, loss = modular_model(X_train_timeseries_normalized.shape[2], args.modality, args.loss)

# Create a folder and store the model
model_time = time.strftime("%Y%m%d-%H%M%S") #use timestamp as folder name...
path = 'models/%s/'%model_time
os.makedirs(os.path.dirname('./models/%s/'%model_time))
# Save architecture for this model
open('models/%s/model_architecture.json'%model_time, 'w').write(model.to_json())

plot_losses = PlotLosses(model_time) #live updating plot of learning curves
plot_model(model,show_shapes=True, to_file='models/%s/architecture.png'%model_time) #architecture schematic

#DON'T CHANGE THE FILEPATH NAMING CONVENTION, THE EVALUATION SCRIPT PARSING DEPENDS ON IT
filepath="models/%s/weights-regression-improvement-{val_loss:.2f}.hdf5"%model_time

checkpointer = ModelCheckpoint(monitor='val_loss', filepath=filepath, verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

csv_logger = CSVLogger('models/%s/log.csv'%model_time, append=True, separator=';')

if args.modality==1:
    model.fit([X_train_timeseries_normalized],
              [y_train], 
              batch_size=batch_size,
              epochs=300, 
              callbacks=[checkpointer,early_stop, csv_logger,plot_losses],
              validation_split=0.1, 
              shuffle=True) 
    
if args.modality==2:
    model.fit([X_train_timeseries_normalized, X_train_demographics_normalized_resting],
              [y_train], 
              batch_size=batch_size,
              epochs=300, 
              callbacks=[checkpointer,early_stop, csv_logger,plot_losses],
              validation_split=0.1, 
              shuffle=True)   

if args.modality==3:
    model.fit([X_train_timeseries_normalized, X_train_demographics_normalized_temporal],
              [y_train], 
              batch_size=batch_size,
              epochs=300, 
              callbacks=[checkpointer,early_stop, csv_logger,plot_losses],
              validation_split=0.1, 
              shuffle=True)  
    
if args.modality==4:
    model.fit([X_train_timeseries_normalized, X_train_demographics_normalized_temporal, X_train_demographics_normalized_resting],
              [y_train], 
              batch_size=batch_size,
              epochs=300, 
              callbacks=[checkpointer,early_stop, csv_logger,plot_losses],
              validation_split=0.1, 
              shuffle=True)            
    
if args.modality==5: #autoencoder
    model.fit([X_train_timeseries_normalized],
              [X_train_timeseries_normalized], 
              batch_size=batch_size,
              epochs=300, 
              callbacks=[checkpointer,early_stop, csv_logger,plot_losses],
              validation_split=0.1, 
              shuffle=True)                  

print ("Training finished. Folder of saved model:", model_time)

#Evaluation
K.clear_session()
del model

print ("Evaluation..")
X_test_timeseries_normalized = np.load('data/X_test_timeseries_normalized.npy')  
X_test_timeseries_normalized = X_test_timeseries_normalized[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)

X_test_demographics_normalized = np.load('data/X_test_demographics_normalized.npy')
X_test_demographics_normalized_resting = X_test_demographics_normalized[:,0]
X_test_demographics_normalized_temporal = X_test_demographics_normalized[:,-4:]

y_test = np.load('data/y_test.npy', allow_pickle=True) 

model = model_from_json(open('./models/'+ model_time +'/model_architecture.json').read())
files = glob.glob('./models/'+model_time+'/*.hdf5')

#WARNING! DEPENDS ON THE NAMING CONVENTION!
#we parse the filename ('./models/20191029-125927/weights-regression-improvement-51.03.hdf5')
#and we convert the MSE to a float in order to sort, the first is the lowest (lowest val_loss)
weights = sorted(files, key=lambda name: float(name[56:-5]))[0]
print ("=============Best model loaded:", weights)

model.load_weights(weights)   

model.compile(loss=[loss], optimizer="adam")
model.summary()

if args.modality==1:
    predicted = model.predict([X_test_timeseries_normalized])
if args.modality==2:
    predicted = model.predict([X_test_timeseries_normalized, X_test_demographics_normalized_resting])
if args.modality==3:
    predicted = model.predict([X_test_timeseries_normalized, X_test_demographics_normalized_temporal])
if args.modality==4:
    predicted = model.predict([X_test_timeseries_normalized, X_test_demographics_normalized_temporal, X_test_demographics_normalized_resting])  
if args.modality==5:
    import sys
    sys.exit("done! We don't predict with the autoencoder since there is not target to evaluate")
   
predicted = np.squeeze((predicted))

mse, rmse, mae =  error_metrics(y_test.astype('float'),np.squeeze(predicted))

#Save results 
import csv
import os.path
file_exists = os.path.isfile("results_file.csv") #create a csv if not exists, or open and append results
with open ("results_file.csv", 'a') as csvfile:
    headers = ['Datasize','Model_checkpoint', 'Modality', 'Loss', 'MSE', 'RMSE', 'MAE']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header
    writer.writerow({'Datasize': X_train_timeseries_normalized.shape[0],
                     'Model_checkpoint': model_time,
                     'Modality': args.modality, 
                     'Loss': args.loss,
                     'MSE': mse,
                     'RMSE': rmse,
                     'MAE': mae,
                    })    
    