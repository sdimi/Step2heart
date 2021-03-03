#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, Masking, TimeDistributed, Dropout, Flatten, LSTM, GRU, Bidirectional, Activation, RepeatVector, InputLayer, Conv1D, Input, Lambda, MaxPooling1D, CuDNNGRU,GlobalAveragePooling1D, multiply, Reshape, Lambda, Permute, GlobalMaxPooling1D, UpSampling1D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import keras.backend as K
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns

def modular_model (dim_shape, modality, loss):
    if loss=="quantile":
        quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
        loss_avg_p = lambda y,f: multi_tilted_loss(quantiles,y,f)
    
    if modality==1: #acc
        tensors_input = Input(shape=(512, dim_shape), name='tensors')
        x = Conv1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu')(tensors_input)
        x = Conv1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        output = GlobalAveragePooling1D()(x)
        final = Dense(1, activation='linear')(output)
        model = Model(inputs=[tensors_input], outputs=[final])
        
    if modality==2: #acc+resting
        tensors_input = Input(shape=(512, dim_shape), name='tensors')
        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(tensors_input)
        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        
        resting_input = Input(shape=(1,), name='resting')
        z = BatchNormalization()(resting_input)
        z = Dense(128, activation='relu')(z)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.33)(z)
        
        output = keras.layers.concatenate([x, z])
        final = Dense(1, activation='linear')(output)
        model = Model(inputs=[tensors_input, resting_input], outputs=[final])
        
    if modality==3: #acc+time
        tensors_input = Input(shape=(512, dim_shape), name='tensors')
        

        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(tensors_input)
        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
        
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
       
        x = GlobalMaxPooling1D()(x)
        
        temporal_input = Input(shape=(4,), name='temporal')
        y = BatchNormalization()(temporal_input)
        y = Dense(128, activation='relu')(y)
        y = Dense(128, activation='relu')(y)
        y = Dropout(0.33)(y)
        
        output = keras.layers.concatenate([x, y])
        final = Dense(1, activation='linear')(output)
        model = Model(inputs=[tensors_input, temporal_input], outputs=[final])
        
    if modality==4: #acc+time+resting
        tensors_input = Input(shape=(512, dim_shape), name='tensors')
        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(tensors_input)
        x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        
        temporal_input = Input(shape=(4,), name='temporal')
        y = BatchNormalization()(temporal_input)
        y = Dense(128, activation='relu')(y)
        y = Dense(128, activation='relu')(y)
        y = Dropout(0.33)(y)
        
        resting_input = Input(shape=(1,), name='resting')
        z = BatchNormalization()(resting_input)
        z = Dense(128, activation='relu')(z)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.33)(z)
        
        output = keras.layers.concatenate([x, y, z])
        final = Dense(1, activation='linear')(output)
        model = Model(inputs=[tensors_input, temporal_input, resting_input], outputs=[final])
        
    if modality==5: #autoencoder
        tensors_input = Input(shape=(512, dim_shape), name='tensors')
        x = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(tensors_input)
        x = MaxPooling1D()(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        encoded = Dense(128)(x)
        x = Reshape((128,1))(encoded)
        x = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(x)
        x = UpSampling1D(2)(x)
        final = Conv1D(filters=10, kernel_size=3, padding='same', strides=1, activation='sigmoid')(x)
        model = Model(inputs=[tensors_input], outputs=[final])

    if loss == "quantile":
        model.compile(loss=[loss_avg_p], optimizer="adam")
        loss = loss_avg_p 
    else:
        model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model, loss

def tilted_loss(q,y,f): #quantile loss function (q=quantile, y,f = ground-truth, predicted)
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss_numpy(y,f): #quantile loss function (q=quantile, y,f = ground-truth, predicted)
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    sum_q = 0 
    e = (y-f)
    for k in range(0,len(quantiles)):
        q = quantiles[k]
        q_individual = np.mean(np.maximum(q*e, (q-1)*e), axis=-1)
        sum_q += q_individual
    return sum_q

def multi_tilted_loss(quantiles,y,f):
    #a traditional MSE loss 
    loss = K.mean(K.square(y-f), axis=-1)#*0 or 0.5 to evaluate impact
    #print (K.shape(loss))    
    for k in range(0,len(quantiles)):
        q = quantiles[k]
        #print (q)
        e = (y-f)
        q_individual = K.mean(K.maximum(q*e, (q-1)*e), axis=-1) #calculate individual quantile
        loss += q_individual #add it to global loss
        #print (K.get_value(q_individual))    
    #print (K.shape(loss))        
    return loss #final loss is [MSE + q1 + q2 + etc.]

def error_metrics(test, predicted):
    mse =  mean_squared_error(test, predicted) #MSE
    rmse =  sqrt(mean_squared_error(test, predicted)) #RMSE
    mae = mean_absolute_error(test, predicted) #MAE
    return mse, rmse, mae

class PlotLosses(keras.callbacks.Callback): #live updating plot with loss and validation loss
    
    def __init__(self, model_time):
        self.model_time = model_time #do this function in order to pass the model folder for the saved png
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        #clear_output(wait=True)
        plt.clf() #new addition, important if not in jupyter, equivalent to clear_output(wait=True)
        
        plt.plot(self.x, self.losses, label="train")
        plt.plot(self.x, self.val_losses, label="val")
        plt.ylabel('Loss')
        plt.legend()        
        plt.xlabel('Epoch')
        plt.savefig("models/%s/training_curves.png"%self.model_time, bbox_inches="tight")       
        #plt.show();