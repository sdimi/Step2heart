#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context("poster")
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
from utils import *

X_train = np.load('data/X_train_timeseries_features_normalized.npy')
X_test = np.load('data/X_test_timeseries_features_normalized.npy')
y_train = np.load('data/y_train.npy') 
y_test = np.load('data/y_test.npy') 

print (X_train.shape,X_test.shape, y_train.shape, y_test.shape )


param_grid_random = {
        
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
}
model = xgb.XGBRegressor(objective='reg:squarederror')

gs = RandomizedSearchCV(model, param_grid_random, n_jobs=-1, n_iter=4, cv = 5,verbose=True)

gs = gs.fit(X_train, y_train)
print ("finished Gridsearch",model)
print (gs.best_params_)
print (gs.best_score_)
print ("saving model")
joblib.dump(gs.best_estimator_, 'models/best_model_xgboost.pkl') 
print ("---------")

clf = joblib.load('models/best_model_xgboost.pkl') 

predicted = clf.predict(X_test)

mse, rmse, mae =  error_metrics(y_test.astype('float'),predicted)

#Save results 
import csv
import os.path
file_exists = os.path.isfile("results_xgb.csv") #create a csv if not exists, or open and append results
with open ("results_xgb.csv", 'a') as csvfile:
    headers = ['Datasize', 'MSE', 'RMSE', 'MAE']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header
    writer.writerow({'Datasize': X_train.shape[0],
                     'MSE': mse,
                     'RMSE': rmse,
                     'MAE': mae,
                    })    


plt.figure(figsize=(15,10))
pd.DataFrame([(predicted),y_test.astype('float')]).T[0].hist(alpha=0.5,bins=100, label='predicted') 
pd.DataFrame([(predicted),y_test.astype('float')]).T[1].hist(alpha=0.5,bins=100, label='ground truth') 
plt.ylabel('frequency'); plt.xlabel('heart rate')
plt.legend(loc='upper right')

plt.figure(figsize=(10,30))
plt.barh(np.arange((X_test.shape[1])), clf.feature_importances_)
plt.yticks(np.arange((X_test.shape[1])), np.arange((X_test.shape[1])))