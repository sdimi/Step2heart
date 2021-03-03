#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split , StratifiedShuffleSplit

X = np.load("data/X_windows_one_user_sample.npy")
Y = np.load("data/Y_windows_one_user_sample.npy")
 
from sklearn.model_selection import GroupShuffleSplit
#split in two disjoint groups of users
#https://stackoverflow.com/questions/44007496/random-sampling-with-pandas-data-frame-disjoint-groups
# Initialize the GroupShuffleSplit.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indexers for the split.
idx1, idx2 = next(gss.split(X, groups=X[:,0,6]))  #the 6th feature is the userID that we split our sets with (check _features_used_deep_learning.csv for variable order)

print(idx1.shape, idx2.shape)

# Get the split.
X_train, y_train = X[idx1], Y[idx1]
X_test, y_test = X[idx2], Y[idx2]

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

mask = np.isin(X_test[:,0,6], X_train[:,0,6]) #the 6th feature is the userID that we split our sets with
assert mask.all() == False #make sure test users do not appear in train set

np.save('data/X_train.npy', X_train)  
np.save('data/y_train.npy', y_train) 
np.save('data/X_test.npy', X_test)  
np.save('data/y_test.npy', y_test) 