# -*- coding: utf-8 -*-
"""classify_main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MYcrc_FhYQZjqvuqXGhb-DFeRPTHPh6O
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import shutil
import random

from colorama import Fore
import shap

from tqdm import tqdm
from PIL import Image, ImageOps
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
from tensorflow.keras import layers, models # For specifying the type of layer (Dense)
from tensorflow.keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

folder = 'drive/MyDrive/I'

enclocs = {'NE Atlantic':[6, 10, 21, 25, 33, 37, 112, 121, 125, 129, 144, 150, 173, 148, 439, 453, 459, 461, 463, 465, 467, 474, 488, 496, 497, 498, 499, 524, 530, 539, 542, 544, 547, 565, 569, 575, 577, 579, 600, 615, 619, 622],
           'North Sea':[55, 558, 561, 585, 596, 436, 437],
           'Scotland East':[65, 67, 68, 84, 95],
           'Scotland West':[77, 80, 81, 83, 181, 182, 186, 192, 199, 202, 218, 229, 242, 250, 258, 282, 296, 303, 309, 318, 321, 335, 342, 352, 355, 369, 375, 376, 381, 411, 413, 414, 415, 428, 443, 415, 417, 427, 435, 477, 481, 483, 484, 485],
           'Shetland':[105, 107, 109],
           'Wales':[494, 495, 625, 626],
           'Iceland':[550, 574],
           'Canary Islands':[634, 646, 656]}

locs = {}
for key in list(enclocs.keys()):
  x = enclocs[key]
  locs[key] = []
  for item in x:
    lab = str(item)
    if len(lab) < 2:
      lab = f'00{lab}'
    elif len(lab) < 3:
      lab = f'0{lab}'
    locs[key].append(lab)

#load click data
d = pd.read_csv(f'{folder}/clickspectra_all/spectra_4s_pruned_smooth5.csv')
selectspecies = ['Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Ttr']
d = d[d.species.isin(selectspecies)].reset_index().drop(columns=['index'])

omit_encs = [148, 181, 33, 37, 144, 112, 129, 150, 173, 342, 547, 601, 619, 622, 696]
testencs = [e for e in np.unique(d.enc_id) if e not in omit_encs]
testencs[:15]
random.shuffle(testencs)

#load whistle data
d = pd.read_csv(f'{folder}/whistlespectra_minMAS1/whistlespectra_minMAS1_pruned.csv')
selectspecies = ['Dde', 'Ggr', 'Gme', 'Lal', 'Ttr']
d = d[d.species.isin(selectspecies)].reset_index().drop(columns=['index'])

omit_encs = [148, 181, 33, 37, 144, 112, 129, 150, 173]
testencs = [e for e in np.unique(d.enc_id) if e not in omit_encs]
testencs[:15]
random.shuffle(testencs)

#classify
voctype = 'whistle'
binary = False
selectspecies = ['Dde', 'Ggr', 'Gme', 'Lal', 'Ttr']
#selectspecies = ['Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Oor', 'Ttr']

nmin = 1 #min clicks per second
THR = 0.1
dd = (0.15,2)

img_shape = (160,1)
#img_shape = (90,1)
nmax = 30
resize = 1
batch_size = 1
epochs = 20
partitions = 5
nfiltersconv=16
kernelconv=3
padding='same'
maxpool=2
leaky=0.1
densesize=10
dropout=0.3
patience=20
seed = 10
use_selectencs = False


split = 0.33
omit_encs = [112, 148, 181, 129, 150, 173, 342, 453, 459, 461, 463, 465]
omit_encs = [494, 625, 6, 713, 112, 6, 148, 181, 173]

history = {'2s':{}, '4s':{}, '8s':{}}
count_framecorrect = {'2s':0, '4s':0, '8s':0}
count_frametotal = {'2s':0, '4s':0, '8s':0}
count_encscorrect = {'2s':0, '4s':0, '8s':0}
count_encstotal = {'2s':0, '4s':0, '8s':0}

ccc = 0
max_ss = None
ratio = 0.5
seedcount = 0
testencs = np.unique(d.enc_id)
np.random.shuffle(testencs)
df_preds_tot = pd.DataFrame()

runs = [int(item[3:]) for item in os.listdir(f'{folder}/models') if 'run' in item]
runid = 0
while runid in runs:
  runid += 1


if f'run{runid}' not in os.listdir(f'{folder}/models'):
  os.mkdir(f'{folder}/models/run{runid}')


print(f'Classifying {len(testencs)} unseen encounters...\n')
for test in testencs:
  seedcount += 1
  for w in [4]:
#    items = os.listdir(f'{folder}/CDsegments_{w}s_ALL')
#    if len(items) < THR:
#      print(f'/nSkipping encounter: enc{test} / temp res. {w}s')
#      continue

 #   imgs = compile_examples(folder=folder, w=w, nmin=nmin, THR=THR, create_test=True, specify_enc=test, verbose=True)
 #   imgs = imgs[~imgs.enc_id.isin(omit_encs)].reset_index().drop(columns=['index'])
 #   if use_selectencs == True:
 #     imgs = imgs[imgs.enc_id.isin(se)].reset_index().drop(columns=['index'])
 #   for key in list(selectencs.keys()):
 #     if test in selectencs[key]:
 #       testsp = key
 #   create_trainval_new(basefolder=folder, subf=f'CDsegments_{w}s_ALL', imgs=imgs, selectspecies=selectspecies, split=split, nmax=nmax, seed=seed, verbose=True)
#    create_trainval_exp(basefolder=folder, locs=locs, subf=f'CDsegments_{w}s_ALL', imgs=imgs, selectspecies=selectspecies, split=split, n=20, seed=seedcount, siteratio=ratio, testsp = testsp, verbose=True)
#    ratio += 0.1

    if voctype in ['click', 'clicks', 'c']:
      dd = None
      selectspecies = ['Dde', 'Ggr', 'Gme', 'Lal', 'Ttr']
      if binary == True:
        startcol = 0
        path = f'{folder}/spectraBinary/allclicks_ds2_pruned.csv'
        img_shape = (81,1)
      else:
        startcol = 1
        path = f'{folder}/clickspectra_all/spectra_4s_pruned_smooth5.csv'
        img_shape = (156,1)

    if voctype in ['whistle', 'whistles', 'w']:
      selectspecies = ['Dde', 'Ggr', 'Gme', 'Lal', 'Ttr']
      if binary == True:
        startcol = 0
        path = f'{folder}/spectraBinary/allwhistles_ds2_pruned_edit.csv'
        img_shape = (90,1)
      else:
        startcol = 1
        path = f'{folder}/whistlespectra_minMAS1/whistlespectra_minMAS1_pruned.csv'
        img_shape = (90,1)

    model_2s, model_4s, model_8s, es, lr_schedule = initialise_model(img_shape=img_shape, classes=len(selectspecies), batch_size=batch_size, epochs=epochs,
                                                                     patience=patience, nfiltersconv=nfiltersconv, kernelconv=kernelconv, padding=padding, maxpool=maxpool, leaky=leaky, densesize=densesize, dropout=dropout)

    models = {'2s':model_2s, '4s':model_4s, '8s': model_8s}

    print(f'Testing encounter {test} ({seedcount}/{len(testencs)})')

    for i in range(0, partitions):
      xtrain, xval, xtest, ytrain, yval, ytest, filenames, encs, info_train, info_val, info_test = compile_data_from_csvpath(path,
                                                                                                                             testenc=int(test),
                                                                                                                             w=w,
                                                                                                                             startcol=startcol,
                                                                                                                             nmax=nmax,
                                                                                                                             selectspecies=selectspecies,
                                                                                                                             omit_encs=omit_encs,
                                                                                                                             dd=dd,
                                                                                                                             split=split,
                                                                                                                             max_ss=max_ss,
                                                                                                                             verbose=False,
                                                                                                                             seed=seed+i)
      if i == 0:
        print(f'Train: {xtrain.shape}')
        print(f'Validation: {xval.shape}')
        print(f'Test: {xtest.shape}')

      if len(xtest) > 0:
        print(f'Fitting model... (partition {i+1}) ({epochs} epochs)')

      if len(xtest) > 0:
        testsp = ytest[0]
    #    xtrain, xval, xtest, ytrain, yval, ytest, filenames, encs, info_train, info_val = compile_data(folder, locs=locs, verbose=False, resize_factor=resize)
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

      if f'{voctype}_{w}s_{test}' not in os.listdir(f'{folder}/models/run{runid}'):
        os.mkdir(f'{folder}/models/run{runid}/{voctype}_{w}s_{test}')

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

      #save model

      filename = f'models/run{runid}/{voctype}_{w}s_{test}/c{test}_{nmin}_all_{nmax}_meta.csv'
      savetolog(folder=folder, filename=filename, locs=locs, info_train=info_train, info_val=info_val,
                testsp=testsp, testenc=test, w=w, epochs=epochs, batch_size=batch_size, resize=resize, patience=patience, model=models[f'{w}s'], nfiltersconv=nfiltersconv, kernelconv=kernelconv,
                padding=padding, maxpool=maxpool, leaky=leaky, densesize=densesize, dropout=dropout, trainencs=None, valencs=None, THR=THR, nmax=nmax, minval=None,
                xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest, encs_t=None, encs_v=None, conf_matrix=conf_matrix, cum=cum, acc=acc, loss=loss, valacc=valacc, valloss=valloss,
                initial=True)