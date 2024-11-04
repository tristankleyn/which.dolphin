# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:48:43 2023

@author: tk81
"""
### Import required modules
print('Importing required modules... ')
import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.lines
from matplotlib.collections import QuadMesh

import seaborn as sns
import pandas as pd
import sqlite3
import random
import copy
from collections import Counter

import librosa
import librosa.display
from scipy.signal import argrelextrema
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks
from scipy.signal import peak_widths

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn import tree

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import warnings

from PIL import Image
import time
import joblib

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

### 
import sys
sys.path.append('C:\\Users\\tk81\\OneDrive - University of St Andrews\\Python\\scripts')
sys.path

from classificationfunctions2_edit import *

### turn off warnings
warnings.filterwarnings('ignore')
verbose = False
rseed = int(input('Enter integer to set as random seed for analysis: '))

resultspath = input("\nEnter destination folder for saving results to: ")
if resultspath in ['', 'new', 'New', 'NEW', 'N', 'n']:
    name = os.listdir('I:/temp')[-1]
    while name in os.listdir('I:/temp'):
        ind = np.random.randint(0,999)
        name = f'classifierresults_{ind}'
    resultspath = f'I:/temp/{name}'

os.mkdir(resultspath)

preset_path = input("Preset path: ")
if len(preset_path) > 0:
    preset = pd.read_csv(preset_path, sep=',')

use_whistles = input("Whistles? (y/n): ")
use_clicks = input("Clicks? (y/n): ")

### load annotations
if len(preset_path) > 0:
    print('Loading annotations from preset...')
    annots = pd.read_csv(str(preset['annots'][0]), sep=',')
else:
 #   annots = input("\nEnter annotations csv path: ")
    print('Loading annotations from preset...')
    annots = 'I:/Downloads/audioannotation.csv'
    annots = pd.read_csv(annots, sep=',')

omit_encs = []
omit = input("\nOmit encounter(s) from file? (y/n)")
if omit in ['y', 'Y', 'yes', 'Yes']:
    x = pd.read_csv('I:/temp/omitencs.csv', header=None).iloc[:,0].values
    for val in x:
        val = str(val)
        if len(val) < 2:
            val = f'00{val}'
        elif len(val) < 3:
            val = f'0{val}'
        omit_encs.append(val)

else:
    omit = '0'
    while len(omit) > 0:
        omit = input('\nOmit encounter: ')
        if len(omit) > 0:
            if len(omit) < 2:
                omit = f'00{omit}'
            elif len(omit) < 3:
                omit = f'0{omit}'

            omit_encs.append(omit)

### load whistles
if use_whistles in ['y', 'Y']:
    selectvars = ['DCMEAN','DCSTDDEV','DCQUARTER1MEAN','DCQUARTER2MEAN','DCQUARTER3MEAN','DCQUARTER4MEAN', 'OVERLAP']
    N_min = 0
    if len(preset_path) > 0:
        print('Loading whistle settings from preset...')
        folder = str(preset['folder_whistles'][0])
        Nmax = int(str(preset['nmax_whistles'][0]))
        min_dur = int(str(preset['mindur_whistles'][0]))
        min_dur = min_dur/1000
        min_freq = int(str(preset['minfreq_whistles'][0]))
        cat_ = str(preset['cat_whistles'][0])
        cat = ''
        for x in ['Aw', 'Au', 'Af', 'Bw', 'Bu', 'Bf']:
            if x in cat_:
                if len(cat) < 1:
                    cat = x
                else:
                    cat = cat + str(f'|{x}')
    else:
        folder = input("\nEnter whistle contours folder: ")
        Nmax = int(input("Enter maximum whistles per recording: "))
        selectvars = ['DCMEAN','DCSTDDEV','DCQUARTER1MEAN','DCQUARTER2MEAN','DCQUARTER3MEAN','DCQUARTER4MEAN', 'OVERLAP']
        min_dur = int(input('\nMinimum whistle duration in ms: '))
        min_dur = min_dur/1000
        min_freq = int(input('Minimum whistle frequency in Hz: '))
        min_MAS = int(input('Minimum whistle mean absolute slope in Hz/s: '))
        cat_ = input('Select annotation categories (Aw, Au, Bw, Bu): ')
        cat = ''
        for x in ['Aw', 'Au', 'Af', 'Bw', 'Bu', 'Bf']:
            if x in cat_:
                if len(cat) < 1:
                    cat = x
                else:
                    cat = cat + str(f'|{x}')

        print(f'Cat: {cat}')


    print('')
    data_w = compile_whistles(folder, Nmax=Nmax, N_min=N_min, cat=cat, rseed=rseed, min_dur=min_dur, min_freq=min_freq, min_MAS = min_MAS)

    nencs = len(np.unique(data_w.enc_id))
    for enc in omit_encs:
        data_w = data_w[data_w.enc_id != enc].reset_index().drop(columns=['index'])

    print(f'{nencs - len(np.unique(data_w.enc_id))} encounters omitted from whistle dataset')

    survey = []
    data_source = []
    for i in range(0, len(data_w)):
        enc = list(data_w.enc_id)[i]
        s = annots[annots.enc_id==int(enc)].reset_index().drop(columns=['index']).survey_id[0]
        ds = annots[annots.enc_id==int(enc)].reset_index().drop(columns=['index']).data_source[0]
        survey.append(s)
        data_source.append(ds)

    data_w = data_w.drop(columns=selectvars)
    data_w.insert(1, 'data_source', data_source)
    data_w.insert(1,'survey', survey)

    data474 = data_w[data_w.enc_id.isin(['474', 474])]
    data474['species'] = ['Dde']*len(data474)
    data_w = data_w[~data_w.enc_id.isin(['474', 474])]
    data_w = data_w._append(data474).reset_index().drop(columns=['index'])

    omitcols_w = [6,None]

    savewhistles = input("\nSave whistle dataset .csv? (y/n) ")
    if savewhistles == 'y':
        data_w.to_csv(f'{resultspath}/data_whistles.csv')


### load clicks
if use_clicks in ['y', 'Y']:
    os.chdir('I:/')
    if len(preset_path) > 0:
        print('Loading whistle settings from preset...')
        folder = str(preset['folder_clicks'][0])
        selectvars = ['DCMEAN','DCSTDDEV','DCQUARTER1MEAN','DCQUARTER2MEAN','DCQUARTER3MEAN','DCQUARTER4MEAN', 'OVERLAP', 'SNR']
        vartype = str(preset['vartype_clicks'][0])
        maxsegs = int(str(preset['maxsegs_clicks'][0]))
        min_snr = float(str(preset['minsnr_clicks'][0]))
        max_snr = float(str(preset['maxsnr_clicks'][0]))
    else:
        folder = input("\nEnter click selections folder: ")
        selectvars = ['DCMEAN','DCSTDDEV','DCQUARTER1MEAN','DCQUARTER2MEAN','DCQUARTER3MEAN','DCQUARTER4MEAN', 'OVERLAP', 'SNR']
        vartype = input('Click variables: (mfcc, band, or rocca) ')
        maxsegs = int(input("Enter maximum clicks per recording: "))
        min_snr = float(input("Enter minimum SNR threshold: "))
        max_snr = float(input("Enter maximum SNR threshold: "))

    if vartype in ['ROCCA', 'Rocca', 'rocca', 'r']:
        data_c = compile_clicksROCCA(folder, Nmax=maxsegs, N_min=0, min_SNR=min_snr, max_SNR=max_snr)

    else:
        data_c, df_spectra, df_clips, df_avgbg = compile_clicks(folder_clicks=folder, annotations=annots,
                                                            sr=96000, nfft=512, seg = 1, maxsegs = maxsegs, length = 0.005333, min_snr = min_snr, max_snr = max_snr,
                                                            smoothing = 2, remove_noise=True, cutoff = 5000, order = 10, rseed = rseed, verbose=False)

        L1 = len(data_c)
        data_c = data_c[data_c.snr > min_snr].reset_index().drop(columns=['index'])
        data_c = data_c[data_c.snr < max_snr].reset_index().drop(columns=['index'])
        L2 = len(data_c)

        print(f'\n{L1-L2} clicks outside SNR interval removed')

    print('')

    ### select click parameters to be used
    if vartype in ['ROCCA', 'Rocca', 'rocca', 'r']: 
        for sv in selectvars:
            if sv in data_c.columns:
                data_c = data_c.drop(columns=[sv])

        nencs = len(np.unique(data_c.enc_id))
        for enc in omit_encs:
            data_c = data_c[data_c.enc_id != enc].reset_index().drop(columns=['index'])

        print(f'{nencs - len(np.unique(data_c.enc_id))} encounters omitted from click dataset')

        survey = []
        data_source = []
        for i in range(0, len(data_c)):
            enc = list(data_c.enc_id)[i]
            s = annots[annots.enc_id==int(enc)].reset_index().drop(columns=['index']).survey_id[0]
            ds = annots[annots.enc_id==int(enc)].reset_index().drop(columns=['index']).data_source[0]
            survey.append(s)
            data_source.append(ds)

        data_c.insert(1, 'data_source', data_source)
        data_c.insert(1,'survey', survey)

        omitcols_c = [6,None]
    else:
        data_c = data_c.rename(columns={'recording':'rec_id'})
        selectvars = ['id', 'startclick', 'starttime', 'n', 'snr', 'survey', 'rec_id', 'enc_id', 'species']

        new = []
        for i in range(0, len(data_c)):
            enc = list(data_c.enc_id)[i]
            new.append(enc[3:])
            
        data_c['enc_id'] = new

        omit = []
        for c in data_c.columns:
            if c not in selectvars:
                if vartype not in c or 'energy' in c or 'std' in c:
                    omit.append(c)

        nencs = len(np.unique(data_c.enc_id))
        for enc in omit_encs:
            data_c = data_c[data_c.enc_id != enc].reset_index().drop(columns=['index'])

        print(f'{nencs - len(np.unique(data_c.enc_id))} encounters omitted from click dataset')

        data_c = data_c.drop(columns=omit)

        data_source = []
        for i in range(0, len(data_c)):
            enc = list(data_c.enc_id)[i]
            ds = annots[annots.enc_id==int(enc)].reset_index().drop(columns=['index']).data_source[0]
            data_source.append(ds)
            
        data_c.insert(1, 'data_source', data_source)

        omitcols_c = [8,8]

    data474 = data_c[data_c.enc_id.isin(['474', 474])]
    data474['species'] = ['Dde']*len(data474)
    data_c = data_c[~data_c.enc_id.isin(['474', 474])]
    data_c = data_c._append(data474).reset_index().drop(columns=['index'])

    saveclicks = input("\nSave click dataset .csv? (y/n) ")
    if saveclicks == 'y':
        data_c.to_csv(f'{resultspath}/data_clicks.csv')

### density clustering
dc = input("\nTrim dataset with density clustering? (y/n) ")
if dc == 'y':
    red = int(input("Percentage reduction: "))/100
    if use_whistles in ['y', 'Y']:
        data_w = densitycluster(data_w, omitcols=omitcols_w, red=red, rseed=rseed)
    if use_clicks in ['y', 'Y']:
        data_c = densitycluster(data_c, omitcols=omitcols_c, red=red, rseed=rseed)

### parameters for classification 
print('\nDefine classification parameters:')
n_estimators = int(input("\n Number of estimator trees: "))

max_depth = input("Maximum tree depth: ")
if max_depth == 'none' or max_depth == 'None':
    max_depth = None
else:
    max_depth = int(max_depth)
    
max_features= input("Maximum number of features per tree: ")
if max_features != 'sqrt':
    max_features = int(max_features)
    
cf_thr = float(input("Strong classification threshold: "))

downsample = input("Downsample to species with least data? (y/n) ")
if downsample == 'y':
    downsample = True
else:
    downsample = False
    
enc_cf_thr = 0.0

info = {'whistles':[], 'clicks':[], 'n_estimators':[], 'max_depth':[], 'max_features':[],
        'cf_thr': [], 'downsample':[], 'nmax':[], 'nmin':[], 'binsize': [], 'whistle_cat':[], 'whistle_minfreq':[], 
        'whistle_mindur': [], 'click_minsnr':[], 'control':[], 'seed':[]}

### bin whistles
if use_whistles in ['y', 'Y']:
    bins = input("\nBin whistle data? (y/n): ")
    if bins == 'y':
        binsize = int(input("Enter bin size: "))
        data_w = bindata(data=data_w, dtype='whistles', N=binsize, rseed=rseed)
        omitcols_w = [7,None]
    else:
        binsize = 1

    ### train whistle classifier
    print('')
    nmax = int(input("\nMaximum number of whistles per training encounter: "))
    nmin = int(input('Minimum number of whistles per testing encounter: '))
    control = input('Specify level of test data independence for whistle classification: (encounter, survey, or source): ')

    print('\nRunning whistle classification...')
    if binsize > 1:
        data_w.to_csv(f'{resultspath}/testbindata.csv', index=False)

    results_w, acc_w, std_w, fi = classify(data_w, omitcols=omitcols_w, target='species', control=control, nmax=nmax, nmin=nmin, n_estimators=n_estimators, 
                                        max_depth=max_depth, max_features=max_features, cf_thr=cf_thr, verbose=verbose, rseed=rseed)


    results_w.to_csv(f'{resultspath}/whistleclassifier_testing.csv')
    print(f'\nSaved whistle classification test results to {resultspath}')
    print(f'Correct whistle classification rate: {round(acc_w,2)} (s = {round(std_w, 2)})')

    #whistle encounter results
    enc_results_w = {'enc_id':[], 'n':[], 'y_test':[], 'pred':[], 'conf':[]}
    for sp in np.unique(results_w.y_test):
        enc_results_w[sp] = []

    for enc in (np.unique(results_w.enc_id)):
        sub = results_w[results_w.enc_id==enc].reset_index().drop(columns=['index'])
        enc_results_w['enc_id'].append(enc)
        enc_results_w['n'].append(len(sub))
        enc_results_w['y_test'].append(sub.y_test[0])
        best = 0
        for sp in np.unique(results_w.y_test):
            avg = np.mean(sub[sp])
            enc_results_w[sp].append(avg)
            if avg > best:
                best = avg
                pred = sp
        enc_results_w['pred'].append(pred)
        enc_results_w['conf'].append(best)
        
    enc_results_w = pd.DataFrame(enc_results_w)
    enc_results_w = enc_results_w[enc_results_w.conf>enc_cf_thr].reset_index().drop(columns=['index'])
    acc = sum(enc_results_w.y_test==enc_results_w.pred)/len(enc_results_w)
    print(f'Correct encounter classification rate: {round(acc,2)}')

    confmatrix = input('\nShow whistle classification results? (y/n) ')
    if confmatrix == 'y':
        confusionmatrix(results_w, tag='Whistle')
        confusionmatrix(enc_results_w, tag='Whistle encounter')

    ### save whistle classifier
    counts = {}
    enccounts = {}
    for sp in np.unique(data_w.species):
        temp = data_w[data_w.species==sp]
        counts[sp] = len(temp)
        enccounts[sp] = len(np.unique(temp.enc_id))

    for sp in np.unique(data_w.species):
        info[f'n_{sp}'] = []
        info[f'N_{sp}'] = []

    save = input("\n Save whistle classifier? (y/n) ")
    if save == 'y':
        destpath = f'{resultspath}/whistleclassifier.joblib'
        save_model(data=data_w, omitcols=omitcols_w, destpath=destpath, target='species', n_estimators=n_estimators, 
                max_depth=max_depth, max_features=max_features, downsample=downsample, cf_thr=cf_thr, 
                verbose=False, rseed=rseed, compress=0)
        
        info['whistles'].append(1)
        info['clicks'].append(0)
        info['n_estimators'].append(n_estimators)
        info['max_depth'].append(max_depth)
        info['max_features'].append(max_features)
        info['cf_thr'].append(cf_thr)
        info['downsample'].append(downsample)
        info['nmax'].append(nmax)
        info['nmin'].append(nmin)
        info['binsize'].append(binsize)
        info['whistle_cat'].append(cat)
        info['whistle_minfreq'].append(min_freq)
        info['whistle_mindur'].append(min_dur)
        info['click_minsnr'].append(None)
        info['control'].append(control)
        info['seed'].append(rseed)
        
        for sp in np.unique(data_w.species):
            info[f'n_{sp}'].append(counts[sp])
            info[f'N_{sp}'].append(enccounts[sp])
    

### bin clicks
if use_clicks in ['y', 'Y']:
    bins = input("\nBin click data? (y/n): ")
    if bins == 'y':
        binsize = int(input("Enter bin size: "))
        data_c = bindata(data=data_c, dtype='clicks', vartype=vartype, N=binsize, rseed=rseed)
        omitcols_c = [8,None]
    else:
        binsize = 1

    ### train click classifier
    print('')
    nmax = int(input("\nMaximum number of clicks per training encounter: "))
    nmin = int(input('Minimum number of clicks per training/testing encounter: '))
    control = input('Specify level of test data independence for click classification: (encounter, survey, or source): ')

    print('\nRunning click classification...')
    if binsize > 1:
        data_c.to_csv(f'{resultspath}/testbindata.csv', index=False)

    results_c, acc_c, std_c, fi = classify(data_c, omitcols=omitcols_c, target='species', control=control, nmax=nmax, nmin=nmin, n_estimators=n_estimators, 
                                        max_depth=max_depth, max_features=max_features, cf_thr=cf_thr, verbose=verbose, rseed=rseed)

    results_c.to_csv(f'{resultspath}/clickclassifier_testing.csv')
    print(f'\nSaved click classification test results to {resultspath}')
    print(f'Correct click classification rate: {round(acc_c,2)} (s = {round(std_c, 2)})')

    #click encounter results
    enc_results_c = {'enc_id':[], 'n':[], 'y_test':[], 'pred':[], 'conf':[]}
    for sp in np.unique(results_c.y_test):
        enc_results_c[sp] = []

    for enc in (np.unique(results_c.enc_id)):
        sub = results_c[results_c.enc_id==enc].reset_index().drop(columns=['index'])
        enc_results_c['enc_id'].append(enc)
        enc_results_c['n'].append(len(sub))
        enc_results_c['y_test'].append(sub.y_test[0])
        best = 0
        for sp in np.unique(results_c.y_test):
            avg = np.mean(sub[sp])
            enc_results_c[sp].append(avg)
            if avg > best:
                best = avg
                pred = sp
        enc_results_c['pred'].append(pred)
        enc_results_c['conf'].append(best)
        
    enc_results_c = pd.DataFrame(enc_results_c)
    enc_results_c = enc_results_c[enc_results_c.conf>enc_cf_thr].reset_index().drop(columns=['index'])
    acc = sum(enc_results_c.y_test==enc_results_c.pred)/len(enc_results_c)
    print(f'Correct encounter classification rate: {round(acc,2)}')

    confmatrix = input('\n Show click classification results? (y/n) ')
    if confmatrix == 'y':
        confusionmatrix(results_c, tag='Click')
        confusionmatrix(enc_results_c, tag='Click encounter')
        
    ### save click classifier
    counts = {}
    enccounts = {}
    for sp in np.unique(data_c.species):
        temp = data_c[data_c.species==sp]
        counts[sp] = len(temp)
        enccounts[sp] = len(np.unique(temp.enc_id))

    for sp in np.unique(data_c.species):
        if f'n_{sp}' not in list(info.keys()):
            info[f'n_{sp}'] = []
            info[f'N_{sp}'] = []

    save = input("\nSave click classifier? (y/n) ")
    if save == 'y':
        destpath = f'{resultspath}/clickclassifier.joblib'
        save_model(data=data_c, omitcols=omitcols_c, destpath=destpath, target='species', n_estimators=n_estimators, 
                max_depth=max_depth, max_features=max_features, downsample=downsample, cf_thr=cf_thr, 
                verbose=False, rseed=rseed, compress=0)

        info['whistles'].append(0)
        info['clicks'].append(1)
        info['n_estimators'].append(n_estimators)
        info['max_depth'].append(max_depth)
        info['max_features'].append(max_features)
        info['cf_thr'].append(cf_thr)
        info['downsample'].append(downsample)
        info['nmax'].append(nmax)
        info['nmin'].append(nmin)
        info['binsize'].append(binsize)
        info['whistle_cat'].append(None)
        info['whistle_minfreq'].append(None)
        info['whistle_mindur'].append(None)
        info['click_minsnr'].append(min_snr)
        info['control'].append(control)
        info['seed'].append(rseed)
        
        for sp in np.unique(data_c.species):
            info[f'n_{sp}'].append(counts[sp])
            info[f'N_{sp}'].append(enccounts[sp])
        
if use_whistles in ['y', 'Y'] and use_clicks in ['y', 'Y']:
    #combine whistle and click predictions
    results_combined = {'enc_id':[], 'y_test':[], 'n_w':[], 'n_c':[], 'conf_w':[], 'conf_c':[]}
    null = 1/len(np.unique(enc_results_w.y_test))

    count = 0
    for col in enc_results_w.columns:
        if count > 4:
            results_combined[f'wc_{col}'] = []
            results_combined[f'w_{col}'] = []
            results_combined[f'c_{col}'] = []
        count += 1

    encs_w = list(np.unique(enc_results_w.enc_id))
    encs_c = list(np.unique(enc_results_c.enc_id))
    encs_w.extend(encs_c)
    enclist = np.unique(encs_w)

    common_encs = list(set(encs_c).intersection(encs_w))

    for enc in enclist:
        results_combined['enc_id'].append(enc)
        temp_w = enc_results_w[enc_results_w.enc_id==enc]
        temp_c = enc_results_c[enc_results_c.enc_id==enc]
        if len(temp_w) == 0:
            if len(temp_c) == 0:
                continue
            else:
                results_combined['y_test'].append(temp_c.y_test.values[0])
                results_combined['n_w'].append(0)
                results_combined['conf_w'].append(null)
                results_combined['n_c'].append(temp_c.n.values[0])
                results_combined['conf_c'].append(temp_c.conf.values[0])
                for sp in np.unique(data_c.species):
                    wprob = null
                    cprob = temp_c[sp].values[0]
                    results_combined[f'wc_{sp}'].append(cprob)
                    results_combined[f'w_{sp}'].append(wprob)
                    results_combined[f'c_{sp}'].append(cprob)
        elif len(temp_c) == 0:
            results_combined['y_test'].append(temp_w.y_test.values[0])
            results_combined['n_c'].append(0)
            results_combined['conf_c'].append(null)
            results_combined['n_w'].append(temp_w.n.values[0])
            results_combined['conf_w'].append(temp_w.conf.values[0])
            for sp in np.unique(data_w.species):
                wprob = temp_w[sp].values[0]
                cprob = null
                results_combined[f'wc_{sp}'].append(wprob)
                results_combined[f'w_{sp}'].append(wprob)
                results_combined[f'c_{sp}'].append(cprob)
        else:
            results_combined['y_test'].append(temp_w.y_test.values[0])
            results_combined['n_c'].append(temp_c.n.values[0])
            results_combined['conf_c'].append(temp_c.conf.values[0])
            results_combined['n_w'].append(temp_w.n.values[0])
            results_combined['conf_w'].append(temp_w.conf.values[0])
            for sp in np.unique(data_w.species):
                wprob = temp_w[sp].values[0]
                cprob = temp_c[sp].values[0]
                weight_wprob = wprob/(wprob+cprob)
                weight_cprob = cprob/(wprob+cprob)
                results_combined[f'wc_{sp}'].append(wprob*weight_wprob + cprob*weight_cprob)
                results_combined[f'w_{sp}'].append(wprob)
                results_combined[f'c_{sp}'].append(cprob)
        
    results_combined = pd.DataFrame(results_combined).sort_values(by='enc_id').reset_index().drop(columns=['index'])

    pred_wc = []
    pred_w = []
    pred_c = []

    for i in range(0, len(results_combined)):
        cols_w = []
        cols_c = []
        cols_wc = []
        for c in results_combined.columns:
            if 'w_' in c[:2]:
                cols_w.append(c)
            elif 'c_' in c[:2]:
                cols_c.append(c)
            elif 'wc_' in c[:3]:
                cols_wc.append(c)
        abr_w = results_combined[cols_w]
        abr_c = results_combined[cols_c]
        abr_wc = results_combined[cols_wc]
        
        if abr_w.iloc[i,0] == 'NA':
            pred_w.append('NA')
        else:
            pred_w.append(abr_w.columns[np.argmax(abr_w.iloc[i,:])][2:])

        if abr_c.iloc[i,0] == 'NA':
            pred_c.append('NA')
        else:
            pred_c.append(abr_c.columns[np.argmax(abr_c.iloc[i,:])][2:])
        
        pred_wc.append(abr_wc.columns[np.argmax(abr_wc.iloc[i,:])][3:])
        

    results_combined['pred_wc'] = pred_wc
    results_combined['pred_w'] = pred_w
    results_combined['pred_c'] = pred_c

    drop = []
    for c in results_combined.columns:
        if 'wc' in c or 'conf' in c:
            drop.append(c)
            
    df = results_combined.drop(columns=drop).reset_index().drop(columns=['index'])
    df = df.rename(columns={'y_test':'species'})

    ### combined whistle click classification
    cf_thr = 0
    combinedclassification = input("\nMethod for combined click/whistle classification (addition or randomforest): ")

    print("\nRunning combined whistle/click classification...")
    if combinedclassification == 'addition':
        results, acc = combined_predict(results_combined, verbose=False)

    elif combinedclassification == 'randomforest':
        results, acc, std, feat_imps = classify(df, omitcols=[4,2], n_estimators=n_estimators, max_depth=max_depth, 
                                    max_features=max_features, cf_thr=cf_thr, verbose=verbose, rseed=rseed)

    confmatrix = input('\nShow combined classification results? (y/n) ')
    if confmatrix == 'y':
        confusionmatrix(results, tag='Combined encounter')

    ### save results
    results.to_csv(f'{resultspath}/classificationresults_combined.csv', index=False)
    results_combined.to_csv(f'{resultspath}/classificationresults_combined1.csv', index=True)
    print('')
    print(f'\nCorrect encounter classification rate: {round(acc,2)}')
    print(f'Saved combined classification results to {resultspath}')

    if combinedclassification == 'randomforest':

        save = input("\nSave combined classifier? (y/n) ")
        if save == 'y':
            destpath = f'{resultspath}/combinedclassifier.joblib'
            save_model(data=df, omitcols=[4,2], destpath=destpath, target='species', n_estimators=n_estimators, 
                    max_depth=max_depth, max_features=max_features, downsample=downsample, cf_thr=cf_thr, 
                    verbose=False, rseed=42, compress=0)
        
            info['whistles'].append(1)
            info['clicks'].append(1)
            info['n_estimators'].append(n_estimators)
            info['max_depth'].append(max_depth)
            info['max_features'].append(max_features)
            info['cf_thr'].append(cf_thr)
            info['downsample'].append(downsample)
            info['nmax'].append(1)
            info['whistle_cat'].append(cat)
            info['whistle_minfreq'].append(min_freq)
            info['whistle_mindur'].append(min_dur)
            info['control'].append('encounter')
            info['seed'].append(rseed)
            
    if len(info['whistles'])>0:
        pd.DataFrame(info).to_csv(f'{resultspath}/modelinfo.csv')
        
print('\n End.')