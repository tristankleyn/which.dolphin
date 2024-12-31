# -*- coding: utf-8 -*-
###classify_functions.py
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


def compile_data_from_csvpath(path, testenc, w=4, startcol=1, nmax=50, selectspecies=['Dde', 'Ggr', 'Gme', 'Lal', 'Ttr'], omit_encs=[], dd=None, max_ss=None, ss_exc_species=[], split=0.33, verbose=False, seed=42):
  data = pd.read_csv(path).iloc[:,startcol:]
  data = data[data.enc_id != 'Unk']
  data['enc_id'] = data['enc_id'].astype(int)
  data = data[data.species.isin(selectspecies)].reset_index().drop(columns=['index'])
  data = data[~data.enc_id.isin(omit_encs)].reset_index().drop(columns=['index'])
  newsp = ['Dde' if list(data.enc_id)[i]==474 else list(data.species)[i] for i in range(0, len(data))]
  data['species'] = newsp

  if 'dd' in data.columns and dd is not None:
    data = data[(data.dd >= dd[0]) & (data.dd <= dd[1])].reset_index().drop(columns=['index'])

  test = data[data.enc_id == testenc].reset_index().drop(columns=['index'])

  if 'dd' in test:
    test = test[test.dd != 'aug'].reset_index().drop(columns=['index'])

  data = data[data.enc_id != testenc].reset_index().drop(columns=['index'])

  data_ = pd.DataFrame()
  for enc in np.unique(data.enc_id):
    sub = data[data.enc_id==enc]
    if len(sub) > nmax:
      sub = sub.sample(n=nmax, random_state=seed)
    data_ = data_._append(sub).reset_index().drop(columns=['index'])

  data = data_.sort_index()
  nmin = 1000000
  for sp in np.unique(data.species):
    L = len(data[data.species == sp])
    if L < nmin:
      nmin = L

  data_ = pd.DataFrame()
  for sp in np.unique(data.species):
    sub = data[data.species == sp].reset_index().drop(columns=['index'])
    while len(sub) > nmin:
      maxenc = sub.groupby('enc_id').count().iloc[:,:1].sort_values(by='species').index[-1]
      dropind = sub[sub.enc_id == maxenc].sample(frac=1, random_state=seed).index[0]
      sub = sub.drop(dropind)

    data_ = data_._append(sub).reset_index().drop(columns=['index'])

  data = data_.sort_index()

  if max_ss is not None:
    if 'n' in data.columns:
      ind = np.where(data.columns=='0')[0][0]
    else:
      ind = np.where(data.columns=='2100')[0][0]
    newdata = pd.DataFrame()
    for enc in np.unique(data.enc_id):
      sub = data[data.enc_id == enc].reset_index().drop(columns=['index'])
      sp = list(sub.species)[0]
      if sp not in ss_exc_species:
        subsp = datap[datap.species == sp]
        medspectrum = np.median(sub.iloc[:,ind:], axis=0)
        medspectrum_sp = np.median(subsp.iloc[:,ind:], axis=0)
        b = medspectrum_sp
        spectra = [sub.iloc[n, ind:].values for n in range(0, len(sub))]
        ss = [100*np.mean((a-b)**2/(np.mean(b)**2)) for a in spectra]
        sub['ss'] = ss
        newdata = newdata._append(sub).reset_index().drop(columns=['index'])

    for sp in ss_exc_species:
      subsp = datap[datap.species == sp]
      newdata = newdata._append(subsp).reset_index().drop(columns=['index'])

    data = newdata[newdata.ss < max_ss].iloc[:,:-1].reset_index().drop(columns=['index'])

  rng = 100
  while rng > 20:
    train = data[data.enc_id != testenc].sample(frac=1).reset_index().drop(columns=['index'])
    ind = int(len(train)*split)
    val = train.iloc[:ind, :].reset_index().drop(columns=['index'])
    train = train.iloc[ind:, :].reset_index().drop(columns=['index'])
    rng = np.max(train.groupby('species').count().iloc[:,:1]) - np.min(train.groupby('species').count().iloc[:,:1])

  train, val = (train.sample(frac=1), val.sample(frac=1))
  if 'sel_id' not in train.columns:
    ind = np.where(train.columns=='rec_id')[0][0]
    train.insert(int(ind), 'sel_id', np.linspace(1,len(train),len(train)))

  if 'sel_id' not in val.columns:
    ind = np.where(val.columns=='rec_id')[0][0]
    val.insert(int(ind), 'sel_id', np.linspace(1,len(val),len(val)))

  if 'sel_id' not in test.columns:
    ind = np.where(test.columns=='rec_id')[0][0]
    test.insert(int(ind), 'sel_id', np.linspace(1,len(test),len(test)))

  if 'n' in train.columns:
    info_train = pd.DataFrame({'species': train['species'], 'enc_id': train['enc_id'], 'rec_id': train['rec_id'], 'sel_id': train['sel_id'], 'starttime':train['starttime'], 'n':train['n'], 'filename':[0]*len(train), 'site':['Other']*len(train)})
    info_val = pd.DataFrame({'species': val['species'], 'enc_id': val['enc_id'], 'rec_id': val['rec_id'], 'sel_id': val['sel_id'], 'starttime':val['starttime'], 'n':val['n'], 'filename':[0]*len(val), 'site':['Other']*len(val)})
    info_test = pd.DataFrame({'species': test['species'], 'enc_id': test['enc_id'], 'rec_id': test['rec_id'], 'sel_id': test['sel_id'], 'starttime':test['starttime'], 'n':test['n'], 'filename':[0]*len(test), 'site':['Other']*len(test)})
  else:
    info_train = pd.DataFrame({'species': train['species'], 'enc_id': train['enc_id'], 'rec_id': train['rec_id'], 'sel_id': train['sel_id'], 'starttime':train['starttime'], 'dd':train['dd'], 'site':['Other']*len(train)})
    info_val = pd.DataFrame({'species': val['species'], 'enc_id': val['enc_id'], 'rec_id': val['rec_id'], 'sel_id': val['sel_id'], 'starttime':val['starttime'], 'dd':val['dd'], 'site':['Other']*len(val)})
    info_test = pd.DataFrame({'species': test['species'], 'enc_id': test['enc_id'], 'rec_id': test['rec_id'], 'sel_id': test['sel_id'], 'starttime':test['starttime'], 'dd':test['dd'], 'site':['Other']*len(test)})

  encs = list(test['enc_id'])
  filenames = list(test['rec_id'])

  ytrain = np.array(train['species'])
  yval = np.array(val['species'])
  ytest = np.array(test['species'])

  if 'SNR' in train.columns:
    ind = np.where(data.columns=='X1')[0][0]+1
  elif 'n' in train.columns:
    ind = np.where(data.columns=='0')[0][0]+1
  elif 'X1' in train.columns:
    ind = np.where(data.columns=='X1')[0][0]+1
  else:
    ind = np.where(data.columns=='2100')[0][0]

  xtrain = np.array(train.iloc[:, ind:])
  xtrain = np.expand_dims(xtrain, axis=2)
  xtrain/(xtrain.sum(axis=1)[:,None]) #make sure each example is sum normalised
  xval = np.array(val.iloc[:, ind:])
  xval = np.expand_dims(xval, axis=2)
  xval/(xval.sum(axis=1)[:,None]) #make sure each example is sum normalised
  xtest = np.array(test.iloc[:, ind:])
  xtest = np.expand_dims(xtest, axis=2)
  xtest/(xtest.sum(axis=1)[:,None]) #make sure each example is sum normalised

  from sklearn.preprocessing import OneHotEncoder
  onehotencoder = OneHotEncoder()
  ytrain = onehotencoder.fit_transform(ytrain.reshape(-1,1)).toarray()
  yval = onehotencoder.transform(yval.reshape(-1,1)).toarray()

  if verbose == True:
    print(f'X Training: {xtrain.shape}')
    print(f'X Validation: {xval.shape}')
    print(f'Y Training: {ytrain.shape}')
    print(f'Y Validation: {yval.shape}')

  return xtrain, xval, xtest, ytrain, yval, ytest, filenames, encs, info_train, info_val, info_test

def initialise_model(img_shape=(48,64,1),
                     resize=10,
                     batch_size=2,
                     epochs=100,
                     lr=0.01,
                     partitions=1,
                     classes=7,
                     nfiltersconv=16,
                     kernelconv=5,
                     padding='same',
                     maxpool=4,
                     leaky=0.1,
                     densesize=8,
                     l2=0.0001,
                     dropout=0.5,
                     patience=25):

  model_2s = Sequential()
  model_2s.add(Conv1D(nfiltersconv, kernel_size=kernelconv, activation='linear', input_shape=img_shape, padding=padding))
  model_2s.add(MaxPooling1D(maxpool,padding=padding))
  model_2s.add(LeakyReLU(alpha=leaky))
  model_2s.add(Flatten())
  model_2s.add(Dense(densesize, activation='linear', kernel_regularizer=regularizers.l2(l2)))
  model_2s.add(Dropout(dropout))
  model_2s.add(Dense(classes, activation='softmax'))

  model_4s = Sequential()
  model_4s.add(Conv1D(nfiltersconv, kernel_size=kernelconv, activation='linear', input_shape=img_shape, padding=padding))
  model_4s.add(MaxPooling1D(maxpool,padding=padding))
  model_4s.add(Conv1D(nfiltersconv, kernel_size=kernelconv+2, activation='linear', input_shape=img_shape, padding=padding))
  model_4s.add(MaxPooling1D(maxpool,padding=padding))
  model_4s.add(LeakyReLU(alpha=leaky))
  model_4s.add(Flatten())
  model_4s.add(Dense(densesize, activation='linear', kernel_regularizer=regularizers.l2(l2)))
  model_4s.add(Dropout(dropout))
  model_4s.add(Dense(classes, activation='softmax'))

  model_8s = Sequential()
  model_8s.add(Conv1D(nfiltersconv, kernel_size=kernelconv, activation='linear', input_shape=img_shape, padding=padding))
  model_8s.add(MaxPooling1D(maxpool,padding=padding))
  model_8s.add(LeakyReLU(alpha=leaky))
  model_8s.add(Flatten())
  model_8s.add(Dense(densesize, activation='linear', kernel_regularizer=regularizers.l2(l2)))
  model_8s.add(Dropout(dropout))
  model_8s.add(Dense(classes, activation='softmax'))

  import keras.backend as K

  model_2s.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
  model_4s.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
  model_8s.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
  es = EarlyStopping('val_loss', mode = 'min', verbose=1, patience = patience)

  def scheduler(epoch, lr):
    if epoch < 30:
      return 0.0005
    else:
      return 0.00025

  lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

  return model_2s, model_4s, model_8s, es, lr_schedule

def predict_test_singlemodel(model, xtest, ytest, testenc, selectspecies=['Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Oor', 'Ttr'], w=2, verbose=True):

  preds = model.predict(xtest, verbose=0)

  d = {'id':[], 'x':[], 'enc_id':[], 'filename':[], 'label':[], 'pred':[], 'conf':[]}
  for sp in selectspecies:
    d[sp] = []
  cum = {}
  for sp in selectspecies:
    cum[sp] = 0

  id = 1
  for i in range(0, len(preds)):
    d['id'].append(id)
    d['x'].append(xtest[i])
    d['enc_id'].append(encs[i])
    d['filename'].append(filenames[i])
    d['label'].append(ytest[i])
    d['pred'].append(np.unique(selectspecies)[np.argmax(preds[i])])
    d['conf'].append(np.max(preds[i]))
    for j in range(0, len(selectspecies)):
      sp = selectspecies[j]
      cum[selectspecies[j]] += preds[i][j]
      d[sp].append(preds[i][j])

  cum_total = np.sum(list(cum.values()))
  for sp in selectspecies:
    cum[sp] = cum[sp]/cum_total

  df_preds = pd.DataFrame(d)

  predicted_species = selectspecies[np.argmax(list(cum.values()))]

  conf_matrix = pd.crosstab(df_preds['label'], df_preds['pred'], rownames=['True'], colnames=['Predicted'])

  if verbose == True:
    if ytest[0] == predicted_species:
      result = 'correct'
      print(f'Encounter {testenc} ({w} sec res.) - - - ' + f'Prediction {result} (True: {ytest[0]}, Predicted: {predicted_species})')
    else:
      result = 'incorrect'
      print(f'Encounter {testenc} ({w} sec res.) - - - ' + f'Prediction {result} (True: {ytest[0]}, Predicted: {predicted_species})')

  return df_preds, cum, predicted_species, ytest[0], conf_matrix

def savetolog(folder, filename, info_train, info_val, selectspecies=[],
              testsp=None, testenc=None, w=None, epochs=None, batch_size=None, resize=None, patience=None, model=None, nfiltersconv=None, kernelconv=None,
              padding=None, maxpool=None, leaky=None, densesize=None, dropout=None, trainencs=None, valencs=None, THR=None, nmax=None, minval=None,
              xtrain=None, xtest=None, ytrain=None, ytest=None, encs_t=None, encs_v=None, conf_matrix=None, cum=None, acc=None, loss=None, valacc=None, valloss=None,
              initial=False, verbose=False):
  d = {}
  allspecies = ['Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Oor', 'Ttr']

  for species in sorted(allspecies):
    d[species] = [species in selectspecies]
  d['testspecies'] = [testsp]
  d['testenc'] = [testenc]
  d['time_res'] = [w]
  d['epochs_max'] = [epochs]
  d['epochs_used'] = [len(valloss)]
  d['batch_size'] = batch_size
  d['compress_factor'] = resize
  d['lrs'] = [[0.001, 0.0005]]
  d['lr_epochchange'] = [30]
  d['patience'] = [patience]
  d['n_layers'] = [len(model.layers)]
  d['layers'] = [model.layers]
  d['nfiltersConv'] = [nfiltersconv]
  d['kernelSize'] = [kernelconv]
  d['padding'] = [padding]
  d['maxPool'] = [maxpool]
  d['leakyReLu'] = [leaky]
  d['denseSize'] = [densesize]
  d['dropout'] = [dropout]

  encs_t = []
  if trainencs is not None:
    for sp in list(trainencs.keys()):
      for item in trainencs[sp]:
        encs_t.append(item)

  encs_v = []
  if valencs is not None:
    for sp in list(valencs.keys()):
      for item in valencs[sp]:
        encs_v.append(item)

  d['min_enc'] = [THR]
  d['max_trainenc'] = [nmax]
  d['min_valset'] = [minval]
  d['ntrain'] = [len(xtrain)]
  d['nval'] = [len(xval)]
  d['ntest'] = [len(xtest)]
  d['n_encstrain'] = [len(encs_t)]
  d['n_encsval'] = [len(encs_v)]
  d['encstrain'] = [encs_t]
  d['encsval'] = [encs_v]

  info_train_sub = info_train[info_train.species == testsp]
  info_val_sub = info_val[info_val.species == testsp]

  tot = 0
  for species in sorted(allspecies):
    if species in conf_matrix.columns:
      d[f'pred{species}'] = [conf_matrix[species][0]]
      tot += conf_matrix[species][0]
    else:
      d[f'pred{species}'] = [0]
      tot += 0

  d['predTOTAL'] = [tot]

  for species in sorted(allspecies):
    if species in selectspecies:
      d[f'aggpred{species}'] = [round(cum[species], 3)]
    else:
      d[f'aggpred{species}'] = [0]

  acc10_end = np.mean(acc[-10:])
  acc10_start = np.mean(acc[:10])
  valacc10_end = np.mean(valacc[-10:])
  valacc10_start = np.mean(valacc[:10])
  loss10_end = np.mean(loss[-10:])
  loss10_start = np.mean(loss[:10])
  valloss10_end = np.mean(valloss[-10:])
  valloss10_start = np.mean(valloss[:10])

  d['acctrain_10'] = [round(acc10_end,3)]
  d['acctrainstd_10'] = [round(np.std(acc[-10:]),3)]
  d['accval_10'] = [round(valacc10_end,3)]
  d['accvalstd_10'] = [round(np.std(valacc[-10:]),3)]
  d['losstrain_10'] = [round(loss10_end,3)]
  d['losstrainstd_10'] = [round(np.std(loss[-10:]),3)]
  d['lossval_10'] = [round(valloss10_end,3)]
  d['lossvalstd_10'] = [round(np.std(valloss[-10:]),3)]
  d['acctrain_trend'] = [round((acc10_end-acc10_start)/(len(acc)-10),3)]
  d['accval_trend'] = [round((valacc10_end-valacc10_start)/(len(valacc)-10),3)]
  d['losstrain_trend'] = [round((loss10_end-loss10_start)/(len(loss)-10),3)]
  d['valloss_trend'] = [round((valloss10_end-valloss10_start)/(len(valloss)-10),3)]

  if initial == True:
    mode = 'w'
    header = True
  else:
    mode = 'a'
    header = False

  pd.DataFrame(d).to_csv(f'{folder}/{filename}', index=True, header=header, mode=mode)
  if verbose == True:
    print(f'Saved results to {folder}/{filename}')