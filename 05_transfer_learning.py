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
from sklearn.decomposition import PCA
from pylab import *

#folder of the best pre-trained model (or replace with own)
folder = '20200115-105719' 

X_train_activations = np.load('data/X_train_activations'+ folder +'.npy')  
X_test_activations = np.load('data/X_test_activations'+ folder +'.npy') 

#Group window-level activations to user-level
def window2user (activations, trainset=True):
    if trainset: 
        userid = np.load('data/X_train_userid.npy', allow_pickle=True)
    else:
        userid = np.load('data/X_test_userid.npy',allow_pickle=True)
    print (np.unique(userid).shape)    
    df = pd.DataFrame(activations).astype(float)#transform datapoints to floats 
    df.index= userid #use the userID as index (helps with groupbys)
    user_level_activations = df.groupby(df.index).mean() #can be max, min, mean, median
    print (user_level_activations.shape)
    return user_level_activations
    
X_train_user_level = window2user(X_train_activations, trainset=True)
X_test_user_level = window2user(X_test_activations, trainset=False)

#Extract demographic labels for classification (per user)
def extract_labels (trainset=True):
    if trainset:
        X = np.load('data/X_train.npy', allow_pickle=True)[:,:,:]
    else:
        X = np.load('data/X_test.npy', allow_pickle=True)
    features_names = pd.read_csv('/home/'+name+'/rds/rds-rjh234-deeplearning/_features_used_deep_learning.csv') 
    label_names = features_names[(features_names.features_used=='id')
               | (features_names.features_used=='height')
              | (features_names.features_used=='weight')
              | (features_names.features_used=='bmi')
              | (features_names.features_used=='sex')
              | (features_names.features_used=='age')
              | (features_names.features_used=='resting_HR')]
    display(label_names)
    label_names_df = pd.DataFrame(X[:,0,label_names.index.values])
    label_names_df.index = label_names_df[0] #move the user_id to index
    del label_names_df[0] #delete the user_id col
    label_names_df_user = label_names_df.groupby(label_names_df.index).first() 
    return label_names_df_user, label_names
    
y_train_metadata, label_names = extract_labels(trainset=True) #takes time since we load the trainset
y_test_metadata, label_names = extract_labels(trainset=False)

y_test_metadata.columns = np.squeeze(label_names.values[1:]) #replace column names, we removed the first column (userID) earlier
y_train_metadata.columns = np.squeeze(label_names.values[1:]) #replace column names, we removed the first column (userID) earlier

fitness_df = pd.read_csv("/data/fitness_test/outcomes.csv") #see ../data_dictionary.csv
fitness_df.index = fitness_df['SerNo'] #move the user_id to index
del fitness_df['SerNo']
fitness_df = fitness_df[['P_TR_FITNESS_trunc15_est', 'P_PAEE_Branch2', 'P_PAEE_Branch6']]

#merge fitness with demographic labels
y_test_medatadata_fitness =  pd.merge(fitness_df, y_test_metadata, left_index=True, right_index=True).dropna(how='any')
#merge activations with new labels
X_y_test_combined = pd.merge(y_test_medatadata_fitness, X_test_user_level, left_index=True, right_index=True )  

#merge fitness with demographic labels
y_train_medatadata_fitness =  pd.merge(fitness_df, y_train_metadata, left_index=True, right_index=True).dropna(how='any')
#merge activations with new labels
X_y_train_combined = pd.merge(y_train_medatadata_fitness, X_train_user_level, left_index=True, right_index=True )  

for i in X_y_train_combined.columns[:9]: #loop through the outcomes #X_y_train_combined.columns[:9] 
    print ("Outcome:",i)
    outcome = i
    if outcome == 'sex': #the only binary outcome (we don't use quantiles here)
        X_y_train_combined["class"] = np.where(X_y_train_combined[outcome] < 0.5, 0, 1) #for binary outcomes
        X_y_test_combined["class"] = np.where(X_y_test_combined[outcome] < 0.5, 0, 1) #for binary outcomes
    else: #here we use the quantile of the TRAIN SET so that there is no bias on the test set
        X_y_train_combined["class"] = np.where(X_y_train_combined[outcome] < X_y_train_combined[outcome].quantile(1/2), 0, 1)
        X_y_test_combined["class"] = np.where(X_y_test_combined[outcome] < X_y_train_combined[outcome].quantile(1/2), 0, 1)
    
    X_train = X_y_train_combined.iloc[:,9:-1].values
    X_test = X_y_test_combined.iloc[:,9:-1].values
    y_train = X_y_train_combined["class"].values
    y_test = X_y_test_combined["class"].values
    print (X_train.shape, X_test.shape)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    pca_explained = 0.999 #parameter to evaluate dimension size 
    pca = PCA(pca_explained) #TSNE
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print ("Explained variance:", pca.explained_variance_ratio_.cumsum().shape) 
    #plt.plot(pca.explained_variance_ratio_.cumsum())
    print (X_train.shape, X_test.shape)
    
    auc_folds = [] 
    
    for j in range(0,5): #repeat training 5 times
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.externals import joblib
        from sklearn.metrics import make_scorer, roc_auc_score
        from scipy import stats
        from scipy.stats import uniform, truncnorm, randint
        
        distributions = dict(C=uniform(loc=0, scale=10),penalty=['l2', 'l1'])
        model = LogisticRegression(class_weight='balanced')
        
        auc = make_scorer(roc_auc_score) #comment out if scoring = 'accuracy'
        print ("starting Randomsearch")
        gs = RandomizedSearchCV(model, distributions,scoring=auc, n_iter=20,n_jobs=-1, cv = 5,verbose=1)
        gs = gs.fit(X_train, y_train)
        print ("finished Gridsearch",model)
        print (gs.best_params_)
        print (gs.best_score_)
        print (gs.cv_results_['mean_test_score'].mean())
        print ("saving model")
        joblib.dump(gs.best_estimator_, 'best_model_LR.pkl') 
        print ("---------")
        clf = joblib.load('best_model_LR.pkl') 

        predicted = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        from sklearn import metrics
        current_auc = metrics.roc_auc_score(y_test, probs[:, 1])
        auc_folds.append(current_auc*100)
        print (i, "AUC:", auc_folds)
    print ("Aggregate results:", i, np.mean(auc_folds), np.std(auc_folds))
    
    #Save results 
    import csv
    import os.path
    file_exists = os.path.isfile("results_transfer_learning.csv") #create a csv if not exists, or open and append results
    with open ("results_transfer_learning.csv", 'a') as csvfile:
        headers = ['Datasize','Model_checkpoint', 'Outcome', 
                   'Original Dimension', 'PCA_components', 'PCA_pcnt', 'mean_AUC', 'std_AUC']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({'Datasize': X_train.shape[0],
                         'Model_checkpoint': folder,
                         'Outcome': i, 
                         'Original Dimension': X_y_train_combined.shape[1],
                         'PCA_components': X_train.shape[1],
                         'PCA_pcnt': pca_explained,
                         'mean_AUC': np.mean(auc_folds),   
                         'std_AUC': np.std(auc_folds)
                        })  
