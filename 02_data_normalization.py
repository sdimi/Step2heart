#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd

X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')

print ("Original data shapes:", X_train.shape, X_test.shape)

features_names = pd.read_csv('/data/feature_names/_features_used_deep_learning.csv') 

indices_timeseries = [0,1,2,3,4,7,8,9,10,11,12,13,14,15,16,17]
indices_dem_features = [5,18,19,20,21,22,23,24,27,28,29,32,33]
indices_userid = [6]
indices_temporal = [25,26,30,31]

features_names.iloc[indices_timeseries].to_csv("data/feature_names/timeseries_feature_names.csv") 
features_names.iloc[indices_dem_features+indices_temporal].to_csv("data/feature_names/metadata_PLUS_temporal_feature_names.csv") 

#normalize timeseries by row so that we don't have to apply the scaler to trainset
#and testset individually, thus doing window-level scaling

def min_max_scaling(array, indices): #transform every timeseries to [0,1]
    #min-max scaling row-wise (also split per feature)
    all_arrays = []
    for i in range(0,len(indices)):
        all_arrays.append(preprocessing.minmax_scale(array[:,:,i].T).T)
    #hacky but it's the only way to append here
    features_tensors =np.dstack([all_arrays[0],
                                 all_arrays[1],
                                 all_arrays[2],
                                 all_arrays[3],
                                 all_arrays[4],
                                 all_arrays[5],
                                 all_arrays[6],
                                 all_arrays[7],
                                 all_arrays[8],
                                 all_arrays[9],
                                 all_arrays[10],
                                 all_arrays[11],
                                 all_arrays[12],
                                 all_arrays[13],
                                 all_arrays[14],
                                 all_arrays[15]
                                ]) #stack them again (2D to 3D)
    return features_tensors

X_test_timeseries_normalized = min_max_scaling(X_test[:,:,indices_timeseries], indices_timeseries)
X_train_timeseries_normalized = min_max_scaling(X_train[:,:,indices_timeseries], indices_timeseries)

print("Normalized timeseries plus temporal:", X_train_timeseries_normalized.shape, X_test_timeseries_normalized.shape)

#this could be also standard scaler, but we do [0,1] for consistency 
scaler = MinMaxScaler()

X_train_demographics_normalized = scaler.fit_transform(X_train[:,0,indices_dem_features+indices_temporal]) #this should be the train set

scaler_filename = "data/scaler_demographics.save"
joblib.dump(scaler, scaler_filename) 

scaler = joblib.load(scaler_filename) 
X_test_demographics_normalized = scaler.transform(X_test[:,0,indices_dem_features+indices_temporal])

print("Normalized metadata:",X_train_demographics_normalized.shape, X_test_demographics_normalized.shape)

X_test_userid = np.squeeze(X_test[:,0,indices_userid])
X_train_userid = np.squeeze(X_train[:,0,indices_userid])

#timeseries
np.save('data/X_train_timeseries_normalized.npy', X_train_timeseries_normalized)  
np.save('data/X_test_timeseries_normalized.npy', X_test_timeseries_normalized) 

#metadata
np.save('data/X_train_demographics_normalized.npy', X_train_demographics_normalized)  
np.save('data/X_test_demographics_normalized.npy', X_test_demographics_normalized) 

#userid
np.save('data/X_train_userid.npy', X_train_userid)  
np.save('data/X_test_userid.npy', X_test_userid)

print("Vectors saved!") 