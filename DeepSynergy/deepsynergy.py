import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS')
parser.add_argument('--val', default=1)
parser.add_argument('--test', default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="0" #specify GPU 
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print(tf.config.list_physical_devices('GPU'))

# If there are GPUs available, set TensorFlow to use only the first GPU
if tf.config.list_physical_devices('GPU'):
    try:
        # Specify which GPU to use
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
        # Configure TensorFlow to use only a specific amount of GPU memory
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Catch runtime error if modification occurs after GPUs have been initialized
        print(e)


hyperparameter_file = '/cta/users/ebcandir/DeepSynergy/hyperparameters.txt' # textfile which contains the hyperparameters of the model
data_file = f'/cta/users/ebcandir/DeepSynergy/data_test_fold{args.test}_val_fold{args.val}_tanh.p' # pickle file which contains the data (produced with normalize.ipynb)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

exec(open(hyperparameter_file).read()) 


file = open(data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()


model = Sequential()
for i in range(len(layers)):
    if i==0:
        model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
                        kernel_initializer='he_normal'))
        model.add(Dropout(float(input_dropout)))
    elif i==len(layers)-1:
        model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
    else:
        model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
        model.add(Dropout(float(dropout)))
    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))


hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
val_loss = hist.history['val_loss']
model.reset_states()

#smooth validation loss for early stopping parameter determination

average_over = 15
mov_av = moving_average(np.array(val_loss), average_over)
smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
epo = np.argmin(smooth_val_loss)

#determine model performance for methods comparison
hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
test_loss = hist.history['val_loss']

# Evaluate on test set
pred = model.predict(X_test)

def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err




np.savetxt(f'/cta/users/ebcandir/DeepSynergy/pred_test{args.test}_val{args.val}.txt', np.asarray(pred), delimiter=",")
np.savetxt(f'/cta/users/ebcandir/DeepSynergy/y_test{args.test}_val{args.val}.txt', np.asarray(y_test), delimiter=",")

mse_value = mse(y_test, pred)
spearman_value = spearman(y_test, pred)
pearson_value = pearson(y_test, pred)

def squared_error(y,pred):
    errs = []
    for i in range(y.shape[0]):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)

def calculate_se(y, pred):

    squared_errs = squared_error(y, pred)
    # standard error of mse
    mse_se = np.std(squared_errs, ddof=1) / np.sqrt(len(squared_errs))   
    return mse_se


# Now call the function with your test data and predictions
calculate_se(y_test, pred)