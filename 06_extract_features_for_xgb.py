#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

#taken from /02_data_normalization.py
indices_timeseries = [0,1,2,3,4,7,8,9,10,11,12,13,14,15,16,17]
indices_dem_features = [5,18,19,20,21,22,23,24,25,26,27,28]

#timeseries
print ("Loading test set...")
X_test = np.load('data/X_test.npy')  #we want NON-NORMALIZED data here, therefore we load the original X_

X_test_timeseries = X_test[:,:,indices_timeseries].astype(float)
X_test_timeseries = X_test_timeseries[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)
print (X_test_timeseries.shape)
del X_test #release memory 

#timeseries
print ("Loading train set...")
X_train = np.load('data/X_train.npy')  #we want NON-NORMALIZED data here, therefore we load the original X_

X_train_timeseries = X_train[:,:,indices_timeseries].astype(float)
X_train_timeseries = X_train_timeseries[:,:,6:] #select only ACC features (index here: data/feature_names/timeseries_feature_names.csv)
print (X_train_timeseries.shape)
del X_train #release memory 

def feature_extraction(array, feature_no): #3D tensor -> 2D features extraction from every [:,:,d] dimension
    print ("Extracting features for modality",feature_no)
    array = pd.DataFrame(array)

    #Pandas .describe() features (cound mean	std	min	25%	50%	75%	max)
    array_features = array.T.describe().T.add_suffix('_%s'%feature_no)
    
    #we use the slope of Linear Regression as a feature
    transp_array = array.T
    array_features["slope_%s"%feature_no] = transp_array.apply(lambda x: np.polyfit(transp_array.index, x, 1)[0])
    
    #remove Count feature since it's repeated in every row
    array_features.drop(list(array_features.filter(regex = 'count')), axis = 1, inplace = True) #drop counts, only 1 value
    return array_features

def concat_feature_modalities(array): #extract statistical features for each timeseries and concatenate into one array
    all_df = []
    for i in range(0,X_test_timeseries.shape[2]): #loop through all feature dimensions (10)
        assert X_test_timeseries.shape[2]==10 
        all_df.append(feature_extraction(array[:,:,i],i))
    
    print ("Concatenating all features in one array..")
    ts_features = pd.concat(all_df, axis=1)    
    return ts_features    

#extract features from timeseries and concat
print ("Extracting TRAIN set features...")
ts_train = concat_feature_modalities(X_train_timeseries[:,:,:])
print ("Extracting TEST set features...")
ts_test = concat_feature_modalities(X_test_timeseries[:,:,:])

print ("Column names:", ts_train.columns.values)

scaler = StandardScaler()

ts_train_scaled = scaler.fit_transform(ts_train) #fit on train test

scaler_filename = "data/scaler_timeseries_features.save"
joblib.dump(scaler, scaler_filename) 

scaler = joblib.load(scaler_filename) 
ts_test_scaled = scaler.transform(ts_test) #transform on test set


print("Normalized timeseries:",ts_train_scaled.shape, ts_test_scaled.shape)

#timeseries
np.save('data/X_train_timeseries_features_normalized.npy', ts_train_scaled)  
np.save('data/X_test_timeseries_features_normalized.npy', ts_test_scaled) 
print("Vectors saved!") 