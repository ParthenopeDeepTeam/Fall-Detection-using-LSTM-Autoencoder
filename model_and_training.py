# define model
#from numpy.random import seed
#seed(42071)
#from tensorflow import random
#random.set_seed(42071)
#Seed buono = 42070

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.regularizers import l2
from preprocessing import *
from datetime import datetime
from math import floor

def get_preprocessed_data(path):
    start_time = datetime.utcnow()
    skeletons_keypoints = load_data(path=path)
    skeletons_keypoints = remove_low_scoring_keypoints(skeletons_keypoints)
    skeletons_keypoints = remove_null_skeleton(skeletons_keypoints)
    # skeletons_keypoints = downsample_dataset(skeletons_keypoints) #don't need data downsample cause they are already at 25 fps
    # the dataset returned by the transformation has not confidence scores for each keypoint
    # we don't need them anymore
    skeletons_keypoints = normalization(skeletons_keypoints)
    batched_scheletons = data_windowing(skeletons_keypoints, overlap=1.5)
    time_delta = datetime.utcnow() - start_time
    print(len(batched_scheletons), "total windows")
    print("Average execution time for preprocessing:", time_delta.total_seconds() / np.array(batched_scheletons).shape[0])

    return np.array(batched_scheletons)


def load_model(timesteps, n_features, lr = 0.001, architecture='shallow'):
    model = Sequential()
    if architecture == 'shallow': # Shallow architecture
        # ENCODER
        model.add(LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        # -------
        # BRIDGE FROM ENCODER TO DECODER
        model.add(RepeatVector(timesteps))
        # -------
        # DECODER
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        # -------
    elif architecture == 'deep': # Deep architecture

    # provare 128, 64, 32 togliendo la regolarizzazione (da confermare che nn va bene)
    # provare anche shallow con 256, 128  (da provare)
        # ENCODER
        model.add(LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=False))
        # -------
        # BRIDGE FROM ENCODER TO DECODER
        model.add(RepeatVector(timesteps))
        # -------
        # DECODER
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        # -------
    
    # Adjusting data shape
    model.add(TimeDistributed(Dense(n_features)))
    # -------

    #optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    return model


def splitting_data(dataset):
    # Splitting data in train set and validation set (80% and 20%)

    total_batches = dataset.shape[0]
    validation_interval = int(20*total_batches/100)
    train_set = dataset
    val_set = np.empty([0,75,38])

    del_indices = range(0, len(train_set), floor(total_batches/validation_interval)) # Indices of elements which will be moved to the validation set

    for i in del_indices:    
        val_set = np.append(val_set, [train_set[i]], axis = 0)
        
    train_set = np.delete(train_set, del_indices, axis=0)
    
    print("Train-set: ", train_set.shape)
    print("Val-set: ", val_set.shape)    

    return train_set, val_set


if __name__ == "__main__":
    # Load data
    data = get_preprocessed_data(path="Train-Set")
    print(data.shape)  # The right shape is (Windows, Frames, Keypoints) = (Batch, Timesteps, N_features)
    data_size = data.shape[0]
    timesteps = data.shape[1]
    n_features = data.shape[2]
    
    train_set, val_set = splitting_data(data)
    
    # /home/spoleto/.miniconda3/envs/falldetection/lib/python3.7/site-packages/keras/callbacks/callbacks.py:95: 
    # RuntimeWarning: Method (on_train_batch_end) is slow, compared to the batch update (1.648069). Check your 
    # callbacks.

    # loss essere tipo: 23742389759238769346592356937456 :D

    # Callbacks definition
    callbacks = []

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001,
                                                           verbose=1)
    callbacks.append(reduce_lr_callback)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=0.0001, restore_best_weights=True, verbose=1)
    callbacks.append(early_stopping_callback)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="models/backup_{epoch:03d}.h5", save_weights_only=True, period=20, verbose=1)
    callbacks.append(checkpoint_callback)

    # Training
    model = load_model(timesteps=timesteps, n_features=n_features, architecture='shallow')

    #for layer in model.layers: print(layer.get_config(), layer.get_weights())

    history = model.fit(x=train_set,
                        y=train_set,
                        epochs=300,
                        batch_size=16,
                        shuffle=False,
                        verbose=2,
                        validation_data=(val_set, val_set),
                        callbacks=callbacks)

    model.save_weights("models/best.h5") # Save model's weights to use them later for prediction
