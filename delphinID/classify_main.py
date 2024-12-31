# -*- coding: utf-8 -*-
###classify_main.ipynb
#########################
## IMPORT MODULES
#########################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import shutil
import random
from colorama import Fore
from tqdm import tqdm
import itertools
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras import Sequential,Input,Model,regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models 
from tensorflow.keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

from classify_functions.py import *

#######################################
##SETTINGS FOR CLASSIFIER DESIGN
#######################################

##LOCATE DATA
folder = '.../delphinID/data'                     #INPUT FULL PATH TO FOLDER CONTAINING SPECTRA DATA
voctype = '...'                                   #INPUT 'whistle' or 'click'
if voctype == 'whistle':
  d = pd.read_csv(f'{folder}//whistlespectra.csv')
else:
  d = pd.read_csv(f'{folder}//clickspectra.csv')

selectspecies = np.unique(d.species)
testencs = np.unique(d.enc_id)

###HYPERPARAMETERS
nmin = 1                              #minimum number of clicks for detection frames
dd = (0.1, 100)                       #range of whistle detection density for detection frames
nmax = 30                             #maximum examples per encounter used for training (default = 30)       
resize = 1                            #factor to compress input arrays by (i.e. factor of 2 halves array length) (default = 1)
batch_size = 1                        #number of examples used for training before retraining internal model parameters (default = 1)
epochs = 20                           #number of training epochs for each bootstrap of training and validation data (default = 20)
partitions = 5                        #number of different partitions/bootstraps of training and validation data to train model on (default = 5)
nfiltersconv=16                       #number of filters used in 1D convolutional layers in CNN model (default = 16)
kernelconv=3                          #number of kernels used in 1D convolutional layers in CNN model (default = 3)
padding='same'                        #zero-pad input features to match output size of 1D convolutional layer in CNN model (default = 'same')
maxpool=2                             #size of sliding window for max pooling layer in CNN model (default = 2)
leaky=0.1                             #constant value for negative inputs into leaky ReLU activation function in CNN model (default = 0.1)
densesize=10                          #size of dense layer in CNN model (default = 10)
dropout=0.2                           #proportion of neurons randomly discarded by dropout in each training step (default = 0.2)
patience=20                           #number of epochs before early stopping callback during model training (default = 20)
seed = 10                             #initial random seed for training
use_selectencs = False                #use list of select encounters for training
split = 0.33                          #proportion of training data used for validation in each training epoch (default = 0.33)
omit_encs = []                        #omit select encounters
model_format = 'saved_model'          #'saved_model' or '.keras' format for saving CNN models

##INITIALISE MODEL AND RESULTS FOLDER
history = {'2s':{}, '4s':{}, '8s':{}}
count_framecorrect = {'2s':0, '4s':0, '8s':0}
count_frametotal = {'2s':0, '4s':0, '8s':0}
count_encscorrect = {'2s':0, '4s':0, '8s':0}
count_encstotal = {'2s':0, '4s':0, '8s':0}
ccc = 0
max_ss = None
ratio = 0.5
seedcount = 0
df_preds_tot = pd.DataFrame()

runs = [int(item[3:]) for item in os.listdir(f'{folder}/models') if 'run' in item]
runid = 0
while runid in runs:
  runid += 1

if f'run{runid}' not in os.listdir(f'{folder}/models'):
  os.mkdir(f'{folder}/models/run{runid}')

##TRAIN AND TEST CLASSIFIER
print(f'Classifying {len(testencs)} encounters...\n')
for test in testencs:
  seedcount += 1
  for w in [4]:
    if voctype in ['click', 'clicks', 'c']:
      dd = None
      startcol = 0
      path = f'{folder}/clickspectra.csv'
      img_shape = (81,1)

    if voctype in ['whistle', 'whistles', 'w']:
      startcol = 0
      path = f'{folder}/whistlespectra.csv'
      img_shape = (90,1)

    model_2s, model_4s, model_8s, es, lr_schedule = initialise_model(img_shape=img_shape, classes=len(selectspecies), batch_size=batch_size, epochs=epochs,
                                                                     patience=patience, nfiltersconv=nfiltersconv, kernelconv=kernelconv, padding=padding, maxpool=maxpool, leaky=leaky, densesize=densesize, dropout=dropout)

    models = {'2s':model_2s, '4s':model_4s, '8s': model_8s}

    print(f'Testing encounter {test} ({seedcount}/{len(testencs)})')

    for i in range(0, partitions):
      xtrain, xval, xtest, ytrain, yval, ytest, filenames, encs, info_train, info_val, info_test = compile_data_from_csvpath(path,
                                                                                                          seed=seed+i)
      if i == 0:
        print(f'Train: {xtrain.shape}')
        print(f'Validation: {xval.shape}')
        print(f'Test: {xtest.shape}')

      if len(xtest) > 0:
        print(f'Fitting model... (partition {i+1}) ({epochs} epochs)')

      if len(xtest) > 0:
        testsp = ytest[0]
        sub = info_train[info_train.species==testsp]
        sub['site'].replace(np.nan, 'other', inplace=True)

        for s in np.unique(sub.site):
          tot = len(sub)
          count = np.sum(sub.site == s)

        history[f'{w}s'] = models[f'{w}s'].fit(xtrain, ytrain, batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(xval, yval), callbacks=[es, lr_schedule])
      else:
        if i == 0:
          print(f'No test examples found. Skipping encounter {test}')
          print('')

    if len(xtest) > 0:
      df_preds, cum, predicted_sp, y0, conf_matrix = predict_test_singlemodel(model=models[f'{w}s'], xtest=xtest, ytest=ytest, testenc=test, selectspecies=selectspecies, w=w)
      extractor = keras.Model(inputs=models[f'{w}s'].inputs, outputs=models[f'{w}s'].get_layer(models[f'{w}s'].layers[-3].name).output)
      embs = pd.DataFrame(extractor.predict(xtest, verbose=0))

      count_framecorrect[f'{w}s'] += sum(df_preds.pred==df_preds.label)
      count_frametotal[f'{w}s'] += len(df_preds)

      if y0 == predicted_sp:
        count_encscorrect[f'{w}s'] += 1
      count_encstotal[f'{w}s'] += 1

      conf = np.max(list(cum.values()))
      frameacc = round(100*count_framecorrect[f'{w}s']/count_frametotal[f'{w}s'], 1)
      encacc = round(100*count_encscorrect[f'{w}s']/count_encstotal[f'{w}s'], 1)
      print(f'Prediction: {np.array(list(cum.values()))}')
      print(f'Running frame accuracy ({w} sec res.): {frameacc}%')
      print(f'Running encounter accuracy ({w} sec res.): {encacc}% \n')

      cumdf = pd.DataFrame([np.array(list(cum.values()))], columns=list(cum.keys()))
      cumdf.insert(0, 'enc_id', test)

      if ccc == 0:
        header = True
        mode = 'w'
        initial = True
      else:
        header = False
        mode = 'a'
        initial = False

      ind = int(np.where(df_preds.columns=='pred')[0][0])
      if 'site' in info_test.columns:
        info_test = info_test.drop(columns=['site'])
      if 'filename' in info_test.columns:
        info_test = info_test.drop(columns=['filename'])

      a = np.array(df_preds.x)
      input_test = pd.DataFrame()
      for item in a:
        input_test = input_test._append(np.transpose(pd.DataFrame(item)))

      info_test = pd.concat([info_test, df_preds.iloc[:,ind:ind+2+len(selectspecies)]], axis=1)
      info_test = pd.concat([info_test, input_test.reset_index().drop(columns=['index'])], axis=1)
      info_test = info_test.sort_values(by=['starttime'], ascending=True).reset_index().drop(columns=['index'])

      #SAVE MODEL AND TEST RESULTS
      if f'{voctype}_{w}s_{test}' not in os.listdir(f'{folder}/models/run{runid}'):
        os.mkdir(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}')

      if model_format == 'saved_model':
        tf.saved_model.save(model_4s, f'{folder}/models/run{runid}/{voctype}_{w}s_{test}')
      else:
        model_4s.save(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/model.keras')
        
      info_train.to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/infotrain.csv', index=False, header=True, mode='w')
      info_val.to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/infoval.csv', index=False, header=True, mode='w')
      info_test.to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/infotest.csv', index=False, header=True, mode='w')
      embs.to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/{voctype[0]}{test}_{nmin}_all_{nmax}_emb.csv', index=False, header=header, mode=mode)
      cumdf.to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/{voctype[0]}{test}_{nmin}_all_{nmax}_cumulative.csv', index=False, header=header, mode=mode)
      df_preds['time_res'] = [w]*len(df_preds)
      df_preds.iloc[:,2:].to_csv(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}/c{test}_{nmin}_all_{nmax}_log.csv', index=False, header=header, mode=mode)
      ccc += 1

      acc = []
      valacc = []
      loss = []
      valloss = []
      hist = history[f'{w}s'].history
      acc.extend(hist['accuracy'])
      valacc.extend(hist['val_accuracy'])
      loss.extend(hist['loss'])
      valloss.extend(hist['val_loss'])

      df_preds_tot = df_preds_tot._append(df_preds).reset_index().drop(columns=['index'])
      filename = f'models/run{runid}/{voctype}_{w}s_{test}/c{test}_{nmin}_all_{nmax}_meta.csv'
      savetolog(folder=folder, filename=filename, selectspecies=selectspecies, info_train=info_train, info_val=info_val,
                testsp=testsp, testenc=test, w=w, epochs=epochs, batch_size=batch_size, resize=resize, patience=patience, model=models[f'{w}s'], nfiltersconv=nfiltersconv, kernelconv=kernelconv,
                padding=padding, maxpool=maxpool, leaky=leaky, densesize=densesize, dropout=dropout, trainencs=None, valencs=None, THR=dd, nmax=nmax, minval=None,
                xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest, encs_t=None, encs_v=None, conf_matrix=conf_matrix, cum=cum, acc=acc, loss=loss, valacc=valacc, valloss=valloss,
                initial=True)