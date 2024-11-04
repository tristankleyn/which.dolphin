# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:51:56 2023

@author: tk81
"""

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

### functions

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def generatespectrum(clip, sr, nfft = 256, window = 'hann', smoothing = 10, remove_noise=False, clip_bg=None,
                     plot_waveform=False, plot_spectrum=True, figsize=(12,6), verbose=True,
                     cutoff=2000, order=2):
    
    ny = sr/2
    cutoff1 = cutoff
    cutoff2 = 45000

    if plot_waveform == True:
        plt.figure(figsize=figsize)
        plt.plot(np.linspace(0, len(clip), len(clip)), clip)

    hop = int(nfft/2)

    S = np.abs(librosa.stft(y=clip, n_fft=nfft, hop_length=hop, window=window)) 
    P = librosa.power_to_db(S**2, ref= 300)
    
    if remove_noise == True:
        S_bg = np.abs(librosa.stft(y=clip_bg, n_fft=nfft, window=window))
        P_bg = librosa.power_to_db(S_bg**2)
        
    sumspectrum = 0
    w = smoothing
    count = 0
    
    for i in range(0, len(P[0])):
        x = P[:,i]
        x = moving_average(x, w)
        if len(np.unique(x)) > 1:
            count += 1
            if count == 1:
                sumspectrum = x
            else:
                sumspectrum = sumspectrum + x

    if remove_noise == True:
        sumspectrum_bg = 0
        w = smoothing
        count = 0

        for i in range(0, len(P_bg[0])):
            x = P_bg[:,i]
            x = moving_average(x, w)
            if len(np.unique(x)) > 1:
                count += 1
                if count == 1:
                    sumspectrum_bg = x
                else:
                    sumspectrum_bg = sumspectrum_bg + x
    
    x = sumspectrum

    if remove_noise == True:
        x = sumspectrum - sumspectrum_bg
        
    f = np.arange(0, ny, (ny)/len(x))
    cut1 = np.where(f > cutoff1)[0][0]
    cut2 = np.where(f < cutoff2)[0][-1]
    x = x[cut1:cut2]
    f = f[cut1:cut2]
        
    snr = np.mean(x)
    
    if w > 1:
        x = moving_average(x, w)
        f = np.arange(cutoff1, cutoff2, (cutoff2-cutoff1)/len(x))

    if plot_spectrum==True:
        plt.figure(figsize=figsize)
        plt.plot(f, x)
        xticks = plt.xticks(fontsize=15)
        yticks = plt.yticks(fontsize=15)

    
    return x, f, snr, sumspectrum, sumspectrum_bg

def peakwidth(f, x, peak, thr=10):
    p = peak
    dif = 0
    difs = []
    countR = 1
    while dif < thr:
        if p+countR == len(f):
            rw = np.max(f)-f[p]
            break
        else:
            dif = x[p]-x[p+countR]
            difs.append(dif)
            if dif > thr:
                if countR > 1:
                    a = thr-difs[-2]
                    b = difs[-1]-thr
                    leg = f[p+countR-1]-f[p]
                    add = (a/(a+b))*(f[p+countR]-f[p+countR-1])
                    rw = leg+add
                else:
                    a = thr
                    b = difs[-1]-thr
                    leg = 0
                    add = (a/(a+b))*(f[p+countR]-f[p+countR-1])
                    rw = leg+add

            countR += 1

    dif = 0
    difs = []
    countL = 1
    while dif < thr:
        if p-countL < 0:
            lw = f[p]-np.min(f)
            break
        else:
            dif = x[p]-x[p-countL]
            difs.append(dif)
            if dif > thr:
                if countL > 1:      
                    a = thr-difs[-2]
                    b = difs[-1]-thr
                    leg = f[p]-f[p-countL+1]
                    add = (a/(a+b))*(f[p-countL+1]-f[p-countL])
                    lw = leg+add
                else:
                    a = thr
                    b = difs[-1]-thr
                    leg = 0
                    add = (a/(a+b))*(f[p-countL+1]-f[p-countL])
                    lw = leg+add

            countL += 1

    start = f[peak]-lw
    end = f[peak]+rw
    width = lw+rw

    return start, end, width

def spectrumstats(f, means, stds, npw=2000, ndiv=20, order=5, bandmax=40000, bandmin=20000):
    d = {}
    maxima = argrelextrema(means, np.greater, order=order)[0]
    if len(maxima) == 0:
        maxima = np.array([np.argmax(means)])
    fstep = (f[-1]-f[0])/(len(f)-1)
    
    x = means
    x = np.min(x)-x
    x = -(x)
    
    div = (bandmax-bandmin)/ndiv
    bounds = list(np.arange(bandmin, bandmax, div))
    bounds.append(bandmax)

    i = 0
    props = []
    arr1 = f > bandmin
    arr2 = f < bandmax
    frame_tot = []
    for j in range(0, len(arr1)):
        frame_tot.append(arr1[j]*arr2[j])
    energy_tot = np.sum(x[frame_tot])
    
    while i < len(bounds)-1:
        arr1 = f > bounds[i]
        arr2 = f < bounds[i+1]
        frame = []
        for j in range(0, len(arr1)):
            frame.append(arr1[j]*arr2[j])
        prop = np.sum(x[frame])/energy_tot
        band = str(i+1)
        if len(band) < 2:
            band = f'0{band}'
        d[f'bandenergy{band}'] = prop
        props.append(prop)
        i += 1
        
    d['std_bandenergy'] = np.std(props)
    
    peaks = {}
    for m in list(maxima):
        peaks[means[m]] = m    
        
    pk1 = peaks[max(peaks.keys())]
    del peaks[max(peaks.keys())]
    if len(peaks.keys()) > 0:
        pk2 = peaks[max(peaks.keys())]
        del peaks[max(peaks.keys())]
    else:
        pk2 = pk1
    if len(peaks.keys()) > 0:
        pk3 = peaks[max(peaks.keys())]
    else:
        pk3 = pk2
    
    d['peak1freq'], d['peak2freq'], d['peak3freq'] = f[pk1], f[pk2], f[pk3]
    d['peak1relamp_mean'], d['peak2relamp_mean'], d['peak3relamp_mean'] = means[pk1], means[pk2], means[pk3]
    d['peak1relamp_std'], d['peak2relamp_std'], d['peak3relamp_std'] = stds[pk1], stds[pk2], stds[pk3]
    
    dist12 = np.abs(f[pk1]-f[pk2])
    dist13 = np.abs(f[pk1]-f[pk3])
    dist23 = np.abs(f[pk2]-f[pk3])
    d['distpk1pk2'], d['distpk1pk3'], d['distpk2pk3'] = dist12, dist13, dist23
    
    count = 0
    thr = 10
    for p in maxima:
        w = peakwidth(f=f, x=means, peak=p, thr=thr)[2]
        if w < npw:
            count += 1
    
    d['n_narrowpeaks'] = count
    
    thr = 3
    width1 = peakwidth(f=f, x=means, peak=pk1, thr=thr)[2]
    width2 = peakwidth(f=f, x=means, peak=pk2, thr=thr)[2]
    width3 = peakwidth(f=f, x=means, peak=pk3, thr=thr)[2]
    d[f'peak1width_{thr}dB'], d[f'peak2width_{thr}dB'], d[f'peak3width_{thr}dB'] = width1, width2, width3
    
    thr = 10
    width1 = peakwidth(f=f, x=means, peak=pk1, thr=thr)[2]
    width2 = peakwidth(f=f, x=means, peak=pk2, thr=thr)[2]
    width3 = peakwidth(f=f, x=means, peak=pk3, thr=thr)[2]
    d[f'peak1width_{thr}dB'], d[f'peak2width_{thr}dB'], d[f'peak3width_{thr}dB'] = width1, width2, width3
    
    return d

def clicksegments1(clips, sr, seg=10, maxsegs = 100, cutoff=5000, order=2, smoothing=2, flat=1000,
                  nfft = 128, remove_noise=True, clips_bg=None, rseed=42):
    df = {'id':[], 'startclick':[], 'n':[], 'snr':[], 'pftrend':[], 'pfstd':[], 'pf_nrelextrema':[],
         'prop_lf':[], 'prop_mf':[], 'prop_hf':[]}
    avgspectra = {'id':[], 'startclick':[], 'f':[], 'means':[], 'stds':[]}
    avgbg = {'id':[], 'startclick':[], 'f':[], 'means':[], 'stds':[]}
    startclick = 0
    count = 1
    ny = sr/2

    random.seed(rseed)
    if len(clips)/seg > maxsegs:
        sample = random.sample(list(range(0,len(clips))), maxsegs)
    else:
        sample = np.arange(0, len(clips), seg).tolist()
    
    for startclick in sample:
        spectra = []
        domband = []
        if remove_noise == True:
            background = []
            snrs = []       

        if len(clips)-startclick < seg:
            continue
        for j in range(startclick, startclick+seg):
            x, f, snr, ss, ssbg = generatespectrum(clips[j], sr, smoothing = smoothing, figsize=(8,4), cutoff=cutoff, 
                                              order=order, nfft=nfft, plot_spectrum=False, verbose=False, 
                                              remove_noise=remove_noise, clip_bg = clips_bg[j])
            
            m = librosa.feature.mfcc(y=clips[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            m_bg = librosa.feature.mfcc(y=clips_bg[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            
            m_mean = []
            m_mean_bg = []
            for i in range(0, len(m)):
                m_mean.append(np.mean(m[i]))
                m_mean_bg.append(np.mean(m_bg[i]))

            x = np.linspace(0, len(m_mean), len(m_mean))
            m_mean = np.array(m_mean)
            m_mean_bg = np.array(m_mean_bg)
            comb = m_mean-m_mean_bg
            plus = -(np.min(comb))
            comb = comb + plus
            comb = comb/np.sum(comb)
            
            x_ = x
#            x_ = np.min(x_)-x_
#            x_ = -(x_)
            f = np.arange(cutoff, ny, (ny-cutoff)/len(x_))
            
            X_ = -(-300-x_)
            div = (np.max(f)-np.min(f))/3
            bounds = np.arange(np.min(f), np.max(f)+div, div)
            maxprop = 0
            lab = {0: 'l', 1:'m', 2:'h'}
            energy = 0
            i = 0
            while i < len(bounds)-1:
                arr1 = f > bounds[i]
                arr2 = f < bounds[i+1]
                frame = arr1 * arr2
                prop = np.sum(x_[frame])/np.sum(x_)
                if prop > maxprop:
                    energy = lab[i]
                    maxprop = prop
                i += 1
            
            domband.append(energy)
            spectra.append(x)
            if remove_noise == True:
                background.append(ssbg)
                snrs.append(snr)

            
        pfcontour = []
        for s in spectra:
            f = np.arange(cutoff, ny, (ny-cutoff)/len(s))
            pf = f[np.argmax(s)]
            pfcontour.append(pf)
            
        pfcontour = np.array(pfcontour)
            
        relext = np.mean([len(argrelextrema(pfcontour, np.greater)[0]), len(argrelextrema(pfcontour, np.less)[0])])
        pfvar = np.std(pfcontour)
        if pfcontour[-1]-pfcontour[0] > flat:
            trend = 1
        elif pfcontour[-1]-pfcontour[0] < -flat:
            trend = -1
        else:
            trend = 0

        spectra = np.array(spectra)
        if remove_noise == True:
            background = np.array(background)
            
        if remove_noise == True:
            avgsnr = np.mean(snrs)
        else:
            avgsnr = None
        means = []
        stds = []
        means_bg = []
        stds_bg = []

        for i in range(0, len(spectra[0])):
            means.append(np.mean(spectra[:,i]))
            stds.append(np.std(spectra[:,i]))
        
        if remove_noise == True:
            for i in range(0, len(spectra[0])):
                means_bg.append(np.mean(background[:,i]))
                stds_bg.append(np.std(background[:,i]))

        means = np.array(means)
        stds = np.array(stds)
        if remove_noise == True:
            means_bg = np.array(means_bg)
            stds_bg = np.array(stds_bg)
        f = np.arange(cutoff, ny, (ny-cutoff)/len(x))
        
        avgspectra['id'].append(count)
        avgspectra['startclick'].append(int(startclick+1))
        avgspectra['f'].append(f)
        avgspectra['means'].append(means)
        avgspectra['stds'].append(stds)
        
        if remove_noise == True:
            avgbg['id'].append(count)
            avgbg['startclick'].append(int(startclick+1))
            avgbg['f'].append(f)
            avgbg['means'].append(means_bg)
            avgbg['stds'].append(stds_bg)
        
        domband = np.array(domband)

        stats = spectrumstats(f, means, stds)
        if len(df['id'])<1:
            for key in list(stats.keys()):
                df[key] = []
            for key in list(stats.keys()):
                df[key].append(stats[key])
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'] = [mfcc]
                count += 1
            
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)
            df['snr'].append(avgsnr)
            df['pftrend'].append(trend)
            df['pfstd'].append(pfvar)
            df['pf_nrelextrema'].append(relext)
            df['prop_lf'].append(np.sum(domband=='l')/len(domband))
            df['prop_mf'].append(np.sum(domband=='m')/len(domband))
            df['prop_hf'].append(np.sum(domband=='h')/len(domband))
                
                
        else:
            for key in list(stats.keys()):
                df[key].append(stats[key])
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'].append(mfcc)
                count += 1
                
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)
            df['snr'].append(avgsnr)
            df['pftrend'].append(trend)
            df['pfstd'].append(pfvar)
            df['pf_nrelextrema'].append(relext)
            df['prop_lf'].append(np.sum(domband=='l')/len(domband))
            df['prop_mf'].append(np.sum(domband=='m')/len(domband))
            df['prop_hf'].append(np.sum(domband=='h')/len(domband))

        startclick += seg
        count += 1

    df = pd.DataFrame(df).reset_index().drop(columns=['index'])
    avgspectra = pd.DataFrame(avgspectra).reset_index().drop(columns=['index'])
    avgbg = pd.DataFrame(avgbg).reset_index().drop(columns=['index'])
    
    return df, avgspectra, avgbg

def clicksegments2(clips, sr, seg=10, maxsegs = 100, cutoff=5000, order=2, smoothing=2, flat=1000,
                  nfft = 128, remove_noise=True, clips_bg=None, rseed=42):
    
    df = {'id':[], 'startclick':[], 'n':[], 'snr':[], 'pftrend':[], 'pfstd':[], 'pf_nrelextrema':[],
         'prop_lf':[], 'prop_mf':[], 'prop_hf':[]}
    avgspectra = {'id':[], 'startclick':[], 'f':[], 'means':[], 'stds':[]}
    avgbg = {'id':[], 'startclick':[], 'f':[], 'means':[], 'stds':[]}
    startclick = 0
    count = 1
    ny = sr/2
    
    random.seed(rseed)
    if len(clips)/seg > maxsegs:
        sample = random.sample(list(range(0,len(clips))), maxsegs)
    else:
        sample = np.arange(0, len(clips), seg).tolist()
    
    for startclick in sample:
        spectra = []
        domband = []
        if remove_noise == True:
            background = []
            snrs = []       
    
        if len(clips)-startclick < seg:
            continue
        for j in range(startclick, startclick+seg):
            x, f, snr, ss, ssbg = generatespectrum(clips[j], sr, smoothing = smoothing, figsize=(8,4), cutoff=cutoff, 
                                              order=order, nfft=nfft, plot_spectrum=False, verbose=False, 
                                              remove_noise=remove_noise, clip_bg = clips_bg[j])
            
            m = librosa.feature.mfcc(y=clips[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            m_bg = librosa.feature.mfcc(y=clips_bg[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            
            m_mean = []
            m_mean_bg = []
            for i in range(0, len(m)):
                m_mean.append(np.mean(m[i]))
                m_mean_bg.append(np.mean(m_bg[i]))
    
            x_ = np.linspace(0, len(m_mean), len(m_mean))
            m_mean = np.array(m_mean)
            m_mean_bg = np.array(m_mean_bg)
            comb = m_mean-m_mean_bg
            plus = -(np.min(comb))
            comb = comb + plus
            comb = comb/np.sum(comb)
            
    #            x_ = np.min(x_)-x_
    #            x_ = -(x_)
            step = (ny-cutoff)/len(x)
            v = cutoff
            f = []
            for i in range(0, len(x)):
                f.append(float(v))
                v += step
            f = np.array(f)
            
            x = -(-300-x)
            div = (np.max(f)-np.min(f))/3
            bounds = np.arange(np.min(f), np.max(f)+div, div)
            maxprop = 0
            lab = {0: 'l', 1:'m', 2:'h'}
            energy = 0
            i = 0
            while i < len(bounds)-1:
                arr1 = f > bounds[i]
                arr2 = f < bounds[i+1]
                frame = arr1 * arr2
                prop = np.sum(x[frame])/np.sum(x)
                if prop > maxprop:
                    energy = lab[i]
                    maxprop = prop
                i += 1
            
            domband.append(energy)
            spectra.append(x)
            if remove_noise == True:
                background.append(ssbg)
                snrs.append(snr)
    
            
        pfcontour = []
        for s in spectra:
            step = (ny-cutoff)/len(s)
            v = cutoff
            f = []
            for i in range(0, len(s)):
                f.append(float(v))
                v += step
            f = np.array(f)
            pf = f[np.argmax(s)]
            pfcontour.append(pf)
            
        pfcontour = np.array(pfcontour)
            
        relext = np.mean([len(argrelextrema(pfcontour, np.greater)[0]), len(argrelextrema(pfcontour, np.less)[0])])
        pfvar = np.std(pfcontour)
        if pfcontour[-1]-pfcontour[0] > flat:
            trend = 1
        elif pfcontour[-1]-pfcontour[0] < -flat:
            trend = -1
        else:
            trend = 0
    
        spectra = np.array(spectra)
        if remove_noise == True:
            background = np.array(background)
            
        if remove_noise == True:
            avgsnr = np.mean(snrs)
        else:
            avgsnr = None
        means = []
        stds = []
        means_bg = []
        stds_bg = []
    
        for i in range(0, len(spectra[0])):
            means.append(np.mean(spectra[:,i]))
            stds.append(np.std(spectra[:,i]))
        
        if remove_noise == True:
            for i in range(0, len(spectra[0])):
                means_bg.append(np.mean(background[:,i]))
                stds_bg.append(np.std(background[:,i]))
    
        means = np.array(means)
        stds = np.array(stds)
        if remove_noise == True:
            means_bg = np.array(means_bg)
            stds_bg = np.array(stds_bg)
    
    
        tot = np.sum(means)
        w = 5
        bands = {}
        indexmin = np.where(f > 15000)[0][0]
        indexmax = np.where(f > 40000)[0][0]
        f_abr = f[indexmin:indexmax]
        means_abr = means[indexmin:indexmax]
        for i in range(0, len(means_abr), w):
            frame = means[i:i+w]
            if len(frame) < w:
                continue
            prop = np.sum(frame)/tot
            lab = str(f_abr[i])
            if len(lab) < 2:
                lab = f'0{lab}'
            bands[f'band{lab}'] = prop
            
        step = (ny-cutoff)/len(x)
        v = cutoff
        f = []
        for i in range(0, len(x)):
            f.append(float(v))
            v += step
        f = np.array(f)
        
        avgspectra['id'].append(count)
        avgspectra['startclick'].append(int(startclick+1))
        avgspectra['f'].append(f)
        avgspectra['means'].append(means)
        avgspectra['stds'].append(stds)
        
        if remove_noise == True:
            avgbg['id'].append(count)
            avgbg['startclick'].append(int(startclick+1))
            avgbg['f'].append(f)
            avgbg['means'].append(means_bg)
            avgbg['stds'].append(stds_bg)
        
        domband = np.array(domband)
    
        if len(df['id'])<1:
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'] = [mfcc]
                count += 1
            for i in range(0, len(list(bands.keys()))):
                df[list(bands.keys())[i]] = [list(bands.values())[i]]
            
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)
            df['snr'].append(avgsnr)
            df['pftrend'].append(trend)
            df['pfstd'].append(pfvar)
            df['pf_nrelextrema'].append(relext)
            df['prop_lf'].append(np.sum(domband=='l')/len(domband))
            df['prop_mf'].append(np.sum(domband=='m')/len(domband))
            df['prop_hf'].append(np.sum(domband=='h')/len(domband))
                
                
        else:
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'].append(mfcc)
                count += 1
            for i in range(0, len(list(bands.keys()))):
                df[list(bands.keys())[i]].append(list(bands.values())[i])
                
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)
            df['snr'].append(avgsnr)
            df['pftrend'].append(trend)
            df['pfstd'].append(pfvar)
            df['pf_nrelextrema'].append(relext)
            df['prop_lf'].append(np.sum(domband=='l')/len(domband))
            df['prop_mf'].append(np.sum(domband=='m')/len(domband))
            df['prop_hf'].append(np.sum(domband=='h')/len(domband))
    
        startclick += seg
        count += 1
    
    df = pd.DataFrame(df).reset_index().drop(columns=['index'])
    avgspectra = pd.DataFrame(avgspectra).reset_index().drop(columns=['index'])
    avgbg = pd.DataFrame(avgbg).reset_index().drop(columns=['index'])
    
    return df, avgspectra, avgbg


def clicksegments(clips, sr, seg=10, maxsegs = 100, cutoff=5000, order=2, smoothing=2, flat=1000,
                  nfft = 128, remove_noise=True, clips_bg=None, rseed=42):
    df = {'id':[], 'startclick':[], 'n':[]}
    startclick = 0
    count = 1
    ny = sr/2

    random.seed(rseed)
    if len(clips)/seg > maxsegs:
        sample = random.sample(list(range(0,len(clips))), maxsegs)
    else:
        sample = np.arange(0, len(clips), seg).tolist()
    
    for startclick in sample:
        if len(clips)-startclick < seg:
            continue
        for j in range(startclick, startclick+seg):          
            m = librosa.feature.mfcc(y=clips[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            m_bg = librosa.feature.mfcc(y=clips_bg[j], sr=sr, n_fft=nfft, hop_length=int(nfft/2), fmax=sr/2)
            
            m_mean = []
            m_mean_bg = []
            for i in range(0, len(m)):
                m_mean.append(np.mean(m[i]))
                m_mean_bg.append(np.mean(m_bg[i]))

            m_mean = np.array(m_mean)
            m_mean_bg = np.array(m_mean_bg)
            comb = m_mean-m_mean_bg
            plus = -(np.min(comb))
            comb = comb + plus
            comb = comb/np.sum(comb)

        if len(df['id'])<1:
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'] = [mfcc]
                count += 1
            
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)
                
        else:
            count = 1
            for mfcc in comb:
                lab = str(count)
                if len(lab) < 2:
                    lab = f'0{lab}'
                df[f'mfcc{lab}'].append(mfcc)
                count += 1
                
            df['id'].append(count)
            df['startclick'].append(int(startclick+1))
            df['n'].append(seg)

        startclick += seg
        count += 1

    df = pd.DataFrame(df).reset_index().drop(columns=['index'])
    
    avgspectra = 1
    avgbg = 2
    
    return df, avgspectra, avgbg

def visualise_spectrum(data, n, color='darkred', fontsize=14, figsize=(10,6), save_plot=False,
                      figpath=None):

    f = data.f[n]
    x = data.means[n]
    s = data.stds[n]

    plt.figure(figsize=(figsize))
    plt.plot(f,x,c=color)
    plt.fill_between(f, x-s, x+s, alpha=0.2, color=color)
    plt.plot(f,x+s, alpha=0.5, linestyle='dotted', color=color)
    plt.plot(f,x-s, alpha=0.5, linestyle='dotted', color=color)

    maxima = argrelextrema(x, np.greater)[0]

    peaks = {}
    for m in list(maxima):
        peaks[x[m]] = m    

    pk1 = peaks[max(peaks.keys())]
    del peaks[max(peaks.keys())]
    if len(peaks.keys()) > 0:
        pk2 = peaks[max(peaks.keys())]
    else:
        pk2 = pk1

    thr = 10
    for p in [pk1, pk2]:
        start, end, width = peakwidth(f=f, x=x, peak=p, thr=10)
        plt.axvline(x=start, alpha = 0.75, c=color)
        plt.axvline(x=end, alpha = 0.75, c=color)
        plt.axvline(x=f[p], linestyle='dashed', alpha = 0.5, c='black')
        plt.axvspan(start, end, alpha=0.3, color=color)
        
    plt.xlabel('Frequency (Hz)', fontsize=fontsize-1)
    plt.ylabel('Relative dB', fontsize=fontsize-1)
    plt.title(f'Average Spectrum \n {data.species[n]} / {data.enc_id[n]} / {data.recording[n]} \n Segment no. {n}', fontsize=fontsize)
    
    if save_plot == True:
        plt.savefig(figpath)
        
def getROCCAstats(folder, N = 40, N_min = 1, cat = 'all', min_dur=0, min_freq=0, min_MAS=0, max_pf=40000, min_SNR=0, max_SNR=50, startcol=14, endcol=69, randseed = 42, clicks=False, unique=False):

    clickcols = ['DURATION', 'FREQCENTER', 'FREQPEAK', 'BW3DB', 'BW3DBLOW', 'BW3DBHIGH', 'BW10DB', 'BW10DBLOW', 'BW10DBHIGH', 'NCROSSINGS', 'SWEEPRATE', 'MEANTIMEZC', 'MEDIANTIMEZC', 'VARIANCETIMEZC', 'SNR']

    ctrstats = {}

    for species in os.listdir(folder):
        if 'x' in species:
            continue
        if unique == True:
            key = 'unq'
        else:
            key = 'enc'
        if 'old' in species:
            continue    
        ctrstats[species] = {}
        enc_count = 0
        for file in os.listdir(f'{folder}/{species}'):
            if '.csv' in file:
                continue
                
            lst = os.listdir(f'{folder}/{species}')
            switch = False
            for item in lst:
                if item[:3] == key:
                    switch = True
                
            if switch == False:
                key = 'enc'
                
            if key in file:
                reccount = 0
                oldencid = 0
                for rec in os.listdir(f'{folder}/{species}/{file}'):
                    nrecs = len(os.listdir(f'{folder}/{species}/{file}'))
                    for i in os.listdir(f'{folder}/{species}/{file}/{rec}'):
                        if 'ContourStats' in i:
                            encid = i[3:6]
                            
                            if encid == oldencid:
                                a = pd.read_csv(f'{folder}/{species}/{file}/{rec}/{i}')
                                if len(a) > N_min:
                                    reccount += 1
                                    if clicks==False:
                                        if cat not in ['', 'all', 'any']:
                                            a = a[a.Source.str.contains(cat)]
                                        a = a[(a.DURATION >= min_dur) & (a.FREQMAX >= min_freq) & (a.FREQABSSLOPEMEAN >= min_MAS)]
                                        a = a.iloc[:,startcol:endcol]
                                        if len(a) > N:
                                            a = a.sample(n=N, random_state=randseed)
                                        a.insert(0, 'sel_id', list(np.arange(1,len(a)+1,1)))
                                        a.insert(0, 'rec_id', rec)
                                        a.insert(0, 'enc_id', encid)
                                    else:
                                        a = a[(a.SNR >= min_SNR) & (a.SNR < max_SNR) & (a.FREQPEAK <= max_pf)]
                                        a = a.loc[:,clickcols]
                                        if len(a) > N:
                                            a = a.sample(n=N, random_state=randseed)
                                        a.insert(0, 'sel_id', list(np.arange(1,len(a)+1,1)))
                                        a.insert(0, 'rec_id', rec)
                                        a.insert(0, 'enc_id', encid)

                                    x = x.append(a, ignore_index=True)
                                    
                            else:
                                b = pd.read_csv(f'{folder}/{species}/{file}/{rec}/{i}')
                                if len(b) > N_min:
                                    x = b
                                    enc_count += 1
                                    reccount += 1
                                    if clicks ==  False:
                                        if cat not in ['', 'all', 'any']:
                                            x = x[x.Source.str.contains(cat)]
                                        x = x[(x.DURATION >= min_dur) & (x.FREQMAX >= min_freq) & (x.FREQABSSLOPEMEAN >= min_MAS)]
                                        x = x.iloc[:,startcol:endcol]
                                        if len(x) > N:
                                            x = x.sample(n=N, random_state=randseed)
                                        x.insert(0, 'sel_id', list(np.arange(1,len(x)+1,1)))
                                        x.insert(0, 'rec_id', rec)
                                        x.insert(0, 'enc_id', encid)
                                    else:
                                        x = x[(x.SNR >= min_SNR) & (x.SNR < max_SNR) & (x.FREQPEAK <= max_pf)]
                                        x = x.loc[:,clickcols]
                                        if len(x) > N:
                                            x = x.sample(n=N, random_state=randseed)
                                        x.insert(0, 'sel_id', list(np.arange(1,len(x)+1,1)))
                                        x.insert(0, 'rec_id', rec)
                                        x.insert(0, 'enc_id', encid)
                            if reccount == nrecs:
                                for c in x.columns:
                                    if c not in list(ctrstats[species].keys()):
                                        ctrstats[species][c] = []
                                        ctrstats[species][c].extend(x[c].values)
                                    else:
                                        ctrstats[species][c].extend(x[c].values)
                                #print(f'Loaded {len(x)} contours from {file}')
                            oldencid = encid  

        variables = x.columns
        ctrstats[species] = list(ctrstats[species].values())
        if clicks == False:
            print(f'{len(ctrstats[species][0])} whistle contours loaded from {enc_count} encounters for {species}.')
        else:
            print(f'{len(ctrstats[species][0])} clicks loaded from {enc_count} encounters for {species}.')

    return ctrstats, variables
    
def compile_whistles(folder, Nmax=50, N_min=0, cat='Aw|Bw|Au', rseed=42, min_dur=0, min_freq=0, min_MAS=0, groupings=None):

    ctrstats, variables = getROCCAstats(folder, N=Nmax, N_min=N_min, min_dur=min_dur, min_freq=min_freq, min_MAS=min_MAS, cat = cat, randseed = rseed)

    contour_means = {}
    contour_stdevs = {}

    index = variables.tolist()
    nvariables = len(index)
    contour_stats = ctrstats

    for species in list(contour_stats.keys()):
        contour_means[species] = []

        contour_stdevs[species] = []

        x = pd.DataFrame(contour_stats[species]).iloc[2:,:].astype(float)
        for i in range(0, len(x.iloc[:,0])):
            contour_means[species].append(np.nanmean(x.iloc[i,:], dtype=float))
            contour_stdevs[species].append(np.nanstd(x.iloc[i,:], dtype=float))

    count = 1
    for sp in list(ctrstats.keys()):
        temp = np.transpose(pd.DataFrame(ctrstats[sp], index=index)).reset_index().drop(columns=['index'])
        temp.insert(0, 'species', [sp]*len(temp))
        if count == 1:
            data = temp
        else:
            data = data.append(temp).reset_index().drop(columns=['index'])
        count += 1
    
    if groupings != None:
        new_species = []
        for i in range(0, len(data)):
            for group in list(groupings.values()):
                if data.species[i] in (group):
                    new_species.append(list(groupings.keys())[list(groupings.values()).index(group)])
                else:
                    new_species.append(data.species[i])

        data['species'] = new_species

    return data

def compile_clicksROCCA(folder, Nmax=50, N_min=0, min_SNR=0, max_SNR=50, max_pf=40000, cat='Aw|Bw|Au', rseed=42, min_dur=0, min_freq=0, groupings=None):

    ctrstats, variables = getROCCAstats(folder, N=Nmax, N_min=N_min, min_dur=min_dur, min_freq=min_freq, max_pf=max_pf, min_SNR=min_SNR, max_SNR=max_SNR, cat = cat, randseed = rseed, clicks=True)

    contour_means = {}
    contour_stdevs = {}

    index = variables.tolist()
    nvariables = len(index)
    contour_stats = ctrstats

    for species in list(contour_stats.keys()):
        contour_means[species] = []

        contour_stdevs[species] = []

        x = pd.DataFrame(contour_stats[species]).iloc[2:,:].astype(float)
        for i in range(0, len(x.iloc[:,0])):
            contour_means[species].append(np.nanmean(x.iloc[i,:], dtype=float))
            contour_stdevs[species].append(np.nanstd(x.iloc[i,:], dtype=float))

    count = 1
    for sp in list(ctrstats.keys()):
        temp = np.transpose(pd.DataFrame(ctrstats[sp], index=index)).reset_index().drop(columns=['index'])
        temp.insert(0, 'species', [sp]*len(temp))
        if count == 1:
            data = temp
        else:
            data = data.append(temp).reset_index().drop(columns=['index'])
        count += 1
    
    if groupings != None:
        new_species = []
        for i in range(0, len(data)):
            for group in list(groupings.values()):
                if data.species[i] in (group):
                    new_species.append(list(groupings.keys())[list(groupings.values()).index(group)])
                else:
                    new_species.append(data.species[i])

        data['species'] = new_species

    return data

def compile_clicks(folder_clicks, annotations, sr=96000, nfft=128, seg = 10, maxsegs = 1000, rseed = 42, length = 0.010, 
                   smoothing = 3, remove_noise=True, cutoff = 2000, order = 10, verbose=False, classify=False):

    folder = folder_clicks
    annots = annotations
    ny = sr/2

    count = 0
    spcount = 0
    for sp in os.listdir(folder):
        if 'x' in sp:
            continue
        if verbose == True:
            print(f'Loading clicks for {sp}')
        clipcount = 0
        enccount = 0
        for enc in os.listdir(f'{folder}/{sp}'):
            if 'enc' in enc:
                enccount += 1
                for rec in os.listdir(f'{folder}/{sp}/{enc}'):
                    for item in os.listdir(f'{folder}/{sp}/{enc}/{rec}'):
                        if 'ann' in item:
                            count += 1
                            filename = f'{folder}/{sp}/{enc}/{rec}/{item}'
                            clicks = pd.read_csv(filename, sep='\t')
                            clicks = clicks[clicks['Begin Time (s)'] > 1].reset_index().drop(columns=['index'])
                            if len(clicks) > maxsegs:
                                clicks = clicks.sample(n=maxsegs, random_state=rseed).reset_index().drop(columns=['index'])
                            clicks['End Time (s)'] = clicks['Begin Time (s)']
                            clicks['Selection'] = list(np.arange(1, len(clicks)+1, 1))
                            if classify == False:
                                survey = annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).survey_id[0]
                                cruise = annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).data_source[0]
                                ch = int(annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).channel_analysed[0])-1
                            else:
                                survey = annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).survey_id[0]
                                cruise = annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).data_source[0]
                                ch = int(annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).channel_analysed[0])-1       
                            wavpath = f'UKPAM/{sp}/{cruise}/{rec}.wav'

                            clip_id = []
                            clip_sp = []
                            clip_enc = []
                            clip_rec = []
                            clip_survey = []
                            clipstart = []
                            clips = []
                            clips_bg = []
                            
                            cutoff1 = cutoff
                            cutoff2 = 45000
                            
                            for i in range(0, len(clicks)):
                                print(f'click {i} loaded')
                                dur = clicks.iloc[i,4] - clicks.iloc[i,3]
                                buf = (length - dur)/2
                                if buf*sr < nfft/2 -1:
                                    continue

                                start = clicks.iloc[i,3] - buf
                                end = clicks.iloc[i,4] + buf
                                dur = end-start

                                if ch == 0:
                                    mono = True
                                else:
                                    mono = False

                                clip, sr = librosa.load(wavpath, sr=sr, offset=start, duration=dur, mono=mono)
                                if mono == False:
                                    clip = clip[ch]
                                
                                clips.append(clip)
                                clip_id.append(int(clicks.Selection[i]))
                                clip_sp.append(str(sp))
                                clip_enc.append(str(enc))
                                clip_rec.append(str(rec))
                                clip_survey.append(int(survey))
                                clipstart.append(float(start+buf))
                                

                                if remove_noise==True:
                                    bg, sr = librosa.load(wavpath, sr=sr, offset=start-buf, duration=buf, mono=mono)
                                    if mono == False:
                                        bg = bg[ch]
                                    
                                    clips_bg.append(bg)
                                else:
                                    clips_bg.append(0)
                                    
                            temp = pd.DataFrame([clip_id, clip_sp, clip_enc, clip_rec, clip_survey, clipstart, clips, clips_bg])
                            temp = np.transpose(temp)
                            temp.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']

                            if clipcount == 0:
                                sp_clips = temp

                            else:
                                sp_clips = sp_clips.append(temp)
                            
                            data, avgspectra, avgbg = clicksegments(clips=clips, sr=sr, seg=seg, maxsegs=maxsegs, cutoff=cutoff, rseed=rseed, 
                                                             smoothing=smoothing, nfft=nfft, remove_noise=remove_noise, clips_bg=clips_bg)

                            data['survey'], avgspectra['survey'], avgbg['survey'] = [survey]*len(data), [survey]*len(avgspectra), [survey]*len(avgbg)
                            data['recording'], avgspectra['recording'], avgbg['recording'] = [rec]*len(data), [rec]*len(avgspectra), [rec]*len(avgbg)
                            data['enc_id'], avgspectra['enc_id'], avgbg['enc_id'] = [enc]*len(data), [enc]*len(avgspectra), [enc]*len(avgbg)
                            data['species'], avgspectra['species'], avgbg['species'] = [sp]*len(data), [sp]*len(avgspectra), [sp]*len(avgbg)

                            if count == 1:
                                df = data
                                df_spectra = avgspectra
                                df_avgbg = avgbg

                            else:
                                df = df.append(data)
                                df_spectra = df_spectra.append(avgspectra)
                                df_avgbg = df_avgbg.append(avgbg)
                            
                            if verbose == True:
                                print(f'{sp} - {enc}/{rec}: {len(data)} segments loaded.')
                            clipcount += len(clips)


        if spcount == 0:
            df_clips = sp_clips
        else:
            df_clips = df_clips.append(sp_clips)
        
        print(f'{len(sp_clips)} click segments loaded from {enccount} encounters for {sp}.')
        spcount += 1
        

    data = df.reset_index().drop(columns=['index'])
    df_spectra = df_spectra.reset_index().drop(columns=['index'])
    df_clips.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']
    df_clips = df_clips.reset_index().drop(columns=['index'])
    df_avgbg = df_avgbg.reset_index().drop(columns=['index'])
    
    for enc in np.unique(data.enc_id):
        temp = data[data.enc_id==enc].reset_index().drop(columns=['index'])
        if len(temp) > maxsegs:
            frac = maxsegs/len(temp)
            temp = temp.sample(frac=frac, random_state=rseed)
        if enc == np.unique(data.enc_id)[0]:
            x = temp
        else:
            x = x.append(temp)
    
    data = x.reset_index().drop(columns=['index'])
    
    for rec in np.unique(data.recording):
        temp1 = df_clips[df_clips.rec_id==rec].reset_index().drop(columns=['index'])
        temp2 = data[data.recording==rec].reset_index().drop(columns=['index'])
        starttime = []
        for i in range(0, len(temp2)):
            st = temp1[temp1.id==temp2.startclick[i]].clipstart.values[0]
            starttime.append(st)
        temp2.insert(2, 'starttime', starttime)
        if rec == np.unique(data.recording)[0]:
            x = temp2
        else:
            x = x.append(temp2)
            
    data = x

    if verbose == True:
        print()
        print(f'{len(df)} total click segments loaded.')

    return data, df_spectra, df_clips, df_avgbg

def compile_clicks(folder_clicks, annotations, sr=96000, nfft=128, seg = 10, maxsegs = 1000, rseed = 42, length = 0.010, min_snr=0, max_snr=10,
                   smoothing = 3, remove_noise=True, cutoff = 2000, order = 10, verbose=False, classify=False):

    folder = folder_clicks
    annots = annotations
    ny = sr/2

    count = 0
    spcount = 0
    for sp in os.listdir(folder):
        if 'x' in sp:
            continue
        if verbose == True:
            print(f'Loading clicks for {sp}')
        clipcount = 0
        enccount = 0
        for enc in os.listdir(f'{folder}/{sp}'):
            if 'enc' in enc:
                enccount += 1
                for rec in os.listdir(f'{folder}/{sp}/{enc}'):
                    for item in os.listdir(f'{folder}/{sp}/{enc}/{rec}'):
                        if 'ann' in item:
                            count += 1
                            filename = f'{folder}/{sp}/{enc}/{rec}/{item}'
                            clicks = pd.read_csv(filename, sep='\t')
                            clicks = clicks[clicks['Begin Time (s)'] > 1].reset_index().drop(columns=['index'])
                            if len(clicks) > maxsegs:
                                clicks = clicks.sample(n=maxsegs, random_state=rseed).reset_index().drop(columns=['index'])
                            clicks['End Time (s)'] = clicks['Begin Time (s)']
                            clicks['Selection'] = list(np.arange(1, len(clicks)+1, 1))
                            if classify == False:
                                survey = annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).survey_id[0]
                                cruise = annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).data_source[0]
                                ch = int(annots[annots.enc_id==int(enc[3:])].reset_index().drop(columns=['index']).channel_analysed[0])-1
                            else:
                                survey = annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).survey_id[0]
                                cruise = annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).data_source[0]
                                ch = int(annots[annots.enc_id==enc[3:]].reset_index().drop(columns=['index']).channel_analysed[0])-1       
                            wavpath = f'UKPAM/{sp}/{cruise}/{rec}.wav'

                            clip_id = []
                            clip_sp = []
                            clip_enc = []
                            clip_rec = []
                            clip_survey = []
                            clipstart = []
                            clips = []
                            clips_bg = []
                            snr = []
                            
                            cutoff1 = cutoff
                            cutoff2 = 45000
                            
                            for i in range(0, len(clicks)):
                                dur = clicks.iloc[i,4] - clicks.iloc[i,3]
                                buf = (length - dur)/2
                                if buf*sr < nfft/2 -1:
                                    continue

                                start = clicks.iloc[i,3] - buf
                                end = clicks.iloc[i,4] + buf
                                dur = end-start

                                if ch == 0:
                                    mono = True
                                else:
                                    mono = False

                                clip, sr = librosa.load(wavpath, sr=sr, offset=start, duration=dur, mono=mono)
                                if mono == False:
                                    clip = clip[ch]
                                amp_clip = np.mean(np.abs(clip))
                                clips.append(clip)
                                clip_id.append(int(clicks.Selection[i]))
                                clip_sp.append(str(sp))
                                clip_enc.append(str(enc))
                                clip_rec.append(str(rec))
                                clip_survey.append(int(survey))
                                clipstart.append(float(start+buf))
                                

                                if remove_noise==True:
                                    bg, sr = librosa.load(wavpath, sr=sr, offset=start-(2*dur), duration=dur, mono=mono)
                                    if mono == False:
                                        bg = bg[ch]
                                    amp_bg = np.mean(np.abs(bg))
                                    clips_bg.append(bg)
                                    snr.append(amp_clip/amp_bg)
                                else:
                                    clips_bg.append(0)
                                    
                            temp = pd.DataFrame([clip_id, clip_sp, clip_enc, clip_rec, clip_survey, clipstart, clips, clips_bg])
                            temp = np.transpose(temp)
                            temp.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']

                            if clipcount == 0:
                                sp_clips = temp

                            else:
                                sp_clips = sp_clips.append(temp)
                            
                            data, avgspectra, avgbg = clicksegments2(clips=clips, sr=sr, seg=seg, maxsegs=maxsegs, cutoff=cutoff, rseed=rseed, 
                                                             smoothing=smoothing, nfft=nfft, remove_noise=remove_noise, clips_bg=clips_bg)

                            data['survey'] = [survey]*len(data)
                            data['recording'] = [rec]*len(data)
                            data['enc_id'] = [enc]*len(data)
                            data['species'] = [sp]*len(data)
                            data['snr'] = snr
                            
                            data = data[data.snr > min_snr].reset_index().drop(columns=['index'])
                            data = data[data.snr < max_snr].reset_index().drop(columns=['index'])

                            if count == 1:
                                df = data

                            else:
                                df = df.append(data)
                            
                            if verbose == True:
                                print(f'{sp} - {enc}/{rec}: {len(data)} segments loaded.')
                            clipcount += len(data)


        if spcount == 0:
            df_clips = sp_clips
        else:
            df_clips = df_clips.append(sp_clips)
                    
        print(f'{clipcount} click segments loaded from {enccount} encounters for {sp}.')
        spcount += 1
        

    data = df.reset_index().drop(columns=['index'])
    df_clips.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']
    df_clips = df_clips.reset_index().drop(columns=['index'])
    
    for enc in np.unique(data.enc_id):
        temp = data[data.enc_id==enc].reset_index().drop(columns=['index'])
        if len(temp) > maxsegs:
            frac = maxsegs/len(temp)
            temp = temp.sample(frac=frac, random_state=rseed)
        if enc == np.unique(data.enc_id)[0]:
            x = temp
        else:
            x = x.append(temp)
    
    data = x.reset_index().drop(columns=['index'])
    
    for rec in np.unique(data.recording):
        temp1 = df_clips[df_clips.rec_id==rec].reset_index().drop(columns=['index'])
        temp2 = data[data.recording==rec].reset_index().drop(columns=['index'])
        starttime = []
        for i in range(0, len(temp2)):
            st = temp1[temp1.id==temp2.startclick[i]].clipstart.values[0]
            starttime.append(st)
        temp2.insert(2, 'starttime', starttime)
        if rec == np.unique(data.recording)[0]:
            x = temp2
        else:
            x = x.append(temp2)
            
    data = x

    if verbose == True:
        print()
        print(f'{len(df)} total click segments loaded.')
        
    df_spectra = 1
    df_avgbg = 1

    return data, df_spectra, df_clips, df_avgbg

def compile_clicks_db(data, audiofolder, sr=96000, nfft=128, seg = 1, maxsegs = 1000, rseed = 42, length = 0.010, 
                   smoothing = 3, remove_noise=True, cutoff = 2000, order = 10, verbose=True):

    count = 0
    spcount = 0
    for enc in np.unique(data.enc_id):
        clipcount = 0
        count += 1
        clicks = data[data.enc_id==enc].reset_index().drop(columns=['index'])
        clicks = clicks[clicks['STARTSECONDS'] > 1].reset_index().drop(columns=['index'])
        if len(clicks) > maxsegs:
            clicks = clicks.sample(n=maxsegs, random_state=rseed).reset_index().drop(columns=['index'])
        clicks['Selection'] = list(np.arange(1, len(clicks)+1, 1))
        survey = enc
        cruise = enc
        sp = 'unk'
        rec = list(clicks.rec_id)[0]
        ch = 1
        wavpath = f'{audiofolder}/{rec}.wav'

        clip_id = []
        clip_sp = []
        clip_enc = []
        clip_rec = []
        clip_survey = []
        clipstart = []
        clips = []
        clips_bg = []
        snr = []

        cutoff1 = cutoff
        cutoff2 = 45000

        for i in range(0, len(clicks)):
            buf = length/2
            if buf*sr < nfft/2 -1:
                continue

            start = list(clicks['STARTSECONDS'])[i] - buf
            end = list(clicks['STARTSECONDS'])[i] + buf
            dur = end-start

            if ch == 0:
                mono = True
            else:
                mono = False

            clip, sr = librosa.load(wavpath, sr=sr, offset=start, duration=dur, mono=mono)
            if mono == False:
                clip = clip[ch]
            amp_clip = np.mean(np.abs(clip))
            clips.append(clip)
            clip_id.append(int(clicks.Selection[i]))
            clip_sp.append(str(sp))
            clip_enc.append(str(enc))
            clip_rec.append(str(rec))
            clip_survey.append(int(survey))
            clipstart.append(float(start+buf))


            if remove_noise==True:
                bg, sr = librosa.load(wavpath, sr=sr, offset=start-(2*dur), duration=dur, mono=mono)
                if mono == False:
                    bg = bg[ch]
                amp_bg = np.mean(np.abs(bg))
                clips_bg.append(bg)
                snr.append(amp_clip/amp_bg)
            else:
                clips_bg.append(0)

        temp = pd.DataFrame([clip_id, clip_sp, clip_enc, clip_rec, clip_survey, clipstart, clips, clips_bg])
        temp = np.transpose(temp)
        temp.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']

        if clipcount == 0:
            sp_clips = temp

        else:
            sp_clips = sp_clips.append(temp)

        data, avgspectra, avgbg = clicksegments(clips=clips, sr=sr, seg=seg, maxsegs=maxsegs, cutoff=cutoff, rseed=rseed, 
                                         smoothing=smoothing, nfft=nfft, remove_noise=remove_noise, clips_bg=clips_bg)

                
        data['survey'] = [survey]*len(data)
        data['recording'] = [rec]*len(data)
        data['enc_id']= [enc]*len(data)
        data['species']= [sp]*len(data)
        data['snr'] = snr
    
        
        if count == 1:
            df = data

        else:
            df = df.append(data)

        if verbose == True:
            print(f'{sp} - enc{enc}/{rec}: {len(data)} segments loaded.')
        clipcount += len(clips)


        if spcount == 0:
            df_clips = sp_clips
        else:
            df_clips = df_clips.append(sp_clips)

        spcount += 1
        

    data = df.reset_index().drop(columns=['index'])
    df_clips.columns = ['id', 'species', 'enc_id', 'rec_id', 'survey', 'clipstart', 'clip', 'clip_bg']
    df_clips = df_clips.reset_index().drop(columns=['index'])
    
    for enc in np.unique(data.enc_id):
        temp = data[data.enc_id==enc].reset_index().drop(columns=['index'])
        if len(temp) > maxsegs:
            frac = maxsegs/len(temp)
            temp = temp.sample(frac=frac, random_state=rseed)
        if enc == np.unique(data.enc_id)[0]:
            x = temp
        else:
            x = x.append(temp)
    
    data = x.reset_index().drop(columns=['index'])
    
    for rec in np.unique(data.recording):
        temp1 = df_clips[df_clips.rec_id==rec].reset_index().drop(columns=['index'])
        temp2 = data[data.recording==rec].reset_index().drop(columns=['index'])
        starttime = []
        for i in range(0, len(temp2)):
            st = temp1[temp1.id==temp2.startclick[i]].clipstart.values[0]
            starttime.append(st)
        temp2.insert(2, 'starttime', starttime)
        if rec == np.unique(data.recording)[0]:
            x = temp2
        else:
            x = x.append(temp2)
            
    data = x

    if verbose == True:
        print()
        print(f'{len(df)} total click segments loaded.')

    return data, df_clips

def loadDBs(folder, N = 40, N_min = 1, randseed = 42):
    dbcount = 1
    dfc = 0
    dfw = 0

    for db in os.listdir(folder):
        if 'sqlite3' in db:
            enc = str(dbcount)
            if len(enc) < 2:
                enc = f'00{enc}'
            elif len(enc) < 3:
                enc = f'0{enc}'

            cnx = sqlite3.connect(f'{folder}/{db}')
            df_cd = pd.read_sql_query("SELECT * FROM Click_Detector_Clicks", cnx)
            df_wd = pd.read_sql_query("SELECT * FROM Rocca_Whistle_Stats", cnx)

            for col in df_cd.columns:
                df_cd = df_cd.rename(columns={col:col.upper()})
            for col in df_wd.columns:
                df_wd = df_wd.rename(columns={col:col.upper()})
            start = np.where(df_wd.columns=='FREQMAX')[0][0]
            end = np.where(df_wd.columns=='STEPDUR')[0][0]
            df_wd = df_wd.iloc[:,start:end]

            c = 0
            if len(df_cd) > N_min:
                c = 1
                df_cd.insert(0, 'startclick', list(np.arange(1,len(df_cd)+1,1)))
                df_cd.insert(0, 'rec_id', db[:-8])
                df_cd.insert(0, 'enc_id', enc)
                df_cd.insert(0, 'survey', enc)
                df_cd.insert(0, 'data_source', enc)
                df_cd.insert(0, 'species', ['unk']*len(df_cd))

                if dfc == 0:
                    dfc = df_cd
                    tot = len(df_cd)
                else:
                    dfc = dfc.append(df_cd)
                    tot += len(df_wd)

            w = 0
            if len(df_wd) > N_min:
                w = 1
                df_wd.insert(0, 'sel_id', list(np.arange(1,len(df_wd)+1,1)))
                df_wd.insert(0, 'rec_id', db[:-8])
                df_wd.insert(0, 'enc_id', enc)
                df_wd.insert(0, 'survey', enc)
                df_wd.insert(0, 'data_source', enc)
                df_wd.insert(0, 'species', ['unk']*len(df_wd))

                if dfw == 0:
                    dfw = df_wd
                    tot = len(df_wd)
                else:
                    dfw = dfw.append(df_wd)
                    tot += len(df_wd)

            if c == 1 or w == 1:
                dbcount += 1

    print(f'{dbcount-1} database(s) found, {tot} tonals loaded.')

    return dfw, dfc

def classify(data, omitcols, target='species', control = 'encounter', nmax=100, nmin = 1, n_estimators=1000, max_depth=None, max_features='sqrt', 
             model_type = 'rf_bal', downsample=False, cf_thr=0, verbose=False, rseed=42):

    df = data.sample(frac=1)
    enclist = list(np.unique(df.enc_id))
    
    keepencs = []
    for enc in enclist:
        temp = df[df.enc_id==enc]
        if len(temp) >= nmin:
            keepencs.append(enc)
    
    df = df[df.enc_id.isin(keepencs)].reset_index().drop(columns=['index'])

            
    if control == 'encounter':
        control = 'enc_id'
    elif control == 'survey':
        control = 'survey'
    elif control == 'source':
        control = 'data_source'

    count = 0
    test_acc_list = []    
    for enc_test in keepencs:
        x_test = df[df.enc_id == enc_test]
        z = list(df[df.enc_id==enc_test][control])[0]
        
        x_train = df[df[control] != z].reset_index().drop(columns=['index'])
        x_test = df[df.enc_id == enc_test].reset_index().drop(columns=['index'])

        y_train = df[df[control]!= z][target].reset_index()[target]
        y_test = df[df.enc_id == enc_test][target].reset_index()[target]

        if len(y_test) < nmin:
            continue

        keep = []
        for enc in np.unique(x_train.enc_id):
            inds = list(x_train[x_train.enc_id==enc].index)
            if len(inds) > nmax:
                inds = random.sample(inds, nmax)
            keep.extend(inds)

        x_train = x_train.iloc[keep,:]
        y_train = y_train[keep]
        
        if downsample == True:
            low = 9999999
            for sp in np.unique(y_train):
                n = sum(y_train==sp)
                if n < low:
                    low = n

            keep = []
            for sp in np.unique(y_train):
                inds = list(y_train[y_train==sp].index)
                inds = random.sample(inds, low)
                keep.extend(inds)
                
            x_train = x_train.loc[keep,:]
            y_train = y_train[keep]
        
        if omitcols[1]==None: 
            x_train = x_train.iloc[:,omitcols[0]:omitcols[1]].values
            x_test_org = x_test.iloc[:,omitcols[0]:omitcols[1]].reset_index().drop(columns=['index'])    
            x_test = x_test.iloc[:,omitcols[0]:omitcols[1]].values
        else:
            x_train = x_train.iloc[:,omitcols[0]:-omitcols[1]].values
            x_test_org = x_test.iloc[:,omitcols[0]:-omitcols[1]].reset_index().drop(columns=['index'])
            x_test = x_test.iloc[:,omitcols[0]:-omitcols[1]].values

        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        model = BalancedRandomForestClassifier(oob_score=True, max_depth = max_depth, 
                                               n_estimators = n_estimators, max_features=max_features)
        

        M = model.fit(x_train, y_train)

        fi = model.feature_importances_
        if enc_test == keepencs[0]:
            feat_imps = fi
        else:
            feat_imps = feat_imps + fi

        oob = model.oob_score_
        predictions = model.predict(x_test)
        probs = model.predict_proba(x_test)

        keep_pred = []

        for i in range(0, len(probs)):
            if np.max(probs[i]) > cf_thr:
                keep_pred.append(True)       
            else:
                keep_pred.append(False)

        x_test = x_test[keep_pred]
        x_test_org = x_test_org.loc[keep_pred, :]
        y_test = y_test[keep_pred]
        predictions = predictions[keep_pred]
        probs = probs[keep_pred]

        if len(predictions) < nmin:
            continue

        temp = np.transpose(pd.DataFrame([[enc_test]*len(y_test), list(y_test), predictions], index=['enc_id', 'y_test', 'pred']))
        temp1 = x_test_org.reset_index().drop(columns=['index'])
        temp1['enc_id'], temp1['y_test'], temp1['pred'] = ([enc_test]*len(y_test), list(y_test), predictions)

        for i in range(0, len(np.unique(y_train))):
            temp[np.unique(y_train)[i]] = probs[:,i]
            temp1[np.unique(y_train)[i]] = probs[:,i]

        if count == 0:
            results = temp1.reset_index().drop(columns=['index'])
        else:
            results = results.append(temp1).reset_index().drop(columns=['index'])

        train_acc = model.score(x_train,y_train)
        test_acc = model.score(x_test,y_test)
        ntotal = len(x_train) + len(x_test)
        ntrain = len(x_train)
        test_acc_list.append(test_acc)

        if verbose == True:
            print(f'{ntotal} samples split into {int(ntrain/ntotal*100)}% train, {int(100*(1-ntrain/ntotal))}% test')
            print(f'Training accuracy: {train_acc:.3f}')
            print(f'Test accuracy: {test_acc:.3f}')
            if 'rf' in model_type:
                print(f'OOB accuracy: {oob:.3f}')


        #fig, ax = plt.subplots(figsize=(8,5))
        test_acc = np.sum(temp.pred==temp.y_test)/len(temp)
        print(f'Encounter {enc_test} ({list(y_test)[0]}): acc. {round(100*test_acc, 1)}%  (train={len(y_train)}, test={len(y_test)})')
        count += 1

    results = pd.DataFrame(results).reset_index().drop(columns=['index'])
    acc = sum(results.pred==results.y_test)/len(results)
    std = np.std(test_acc_list)
    feat_imps = feat_imps/len(keepencs)
    
    return results, acc, std, feat_imps

def classify1(data, omitcols, target='species', control = 'encounter', nmax=100, nmin=1, n_estimators=1000, model_type = 'rf_bal',
             max_depth=None, max_features='sqrt', downsample=False,
             cf_thr=0, verbose=False, rseed=42):

    df = data.sample(frac=1)
    enclist = list(np.unique(df.enc_id))
    
    keepencs = []
    for enc in enclist:
        temp = df[df.enc_id==enc]
        if len(temp) >= nmin:
            keepencs.append(enc)
    
    df = df[df.enc_id.isin(keepencs)].reset_index().drop(columns=['index'])
    
    if control == 'encounter':
        control = 'enc_id'
    elif control == 'survey':
        control = 'survey'
    elif control == 'source':
        control = 'data_source'

    count = 0
    test_acc_list = []
    for enc_test in keepencs:
        x_test = df[df.enc_id == enc_test]
        x_test_id = x_test['startclick'].values
        x_test_rec = x_test['rec_id'].values
        x_test_st = x_test['starttime'].values
        
        z = list(df[df.enc_id==enc_test][control])[0]
        
        x_train = df[df[control] != z].reset_index().drop(columns=['index'])
        x_test = df[df.enc_id == enc_test].reset_index().drop(columns=['index'])

        y_train = df[df[control]!= z][target].reset_index()[target]
        y_test = df[df.enc_id == enc_test][target].reset_index()[target]

        keep = []
        for enc in np.unique(x_train.enc_id):
            inds = list(x_train[x_train.enc_id==enc].index)
            if len(inds) > nmax:
                inds = random.sample(inds, nmax)
            keep.extend(inds)

        if omitcols[1]==None: 
            x_train = x_train.iloc[:,omitcols[0]:omitcols[1]].values
            x_test = x_test.iloc[:,omitcols[0]:omitcols[1]].values
        else:
            x_train = x_train.iloc[:,omitcols[0]:-omitcols[1]].values
            x_test = x_test.iloc[:,omitcols[0]:-omitcols[1]].values

        x_train = x_train[keep]
        y_train = y_train[keep]

        if len(y_test) < nmin:
            print(len(y_test), nmin)
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if downsample == True:
            low = 9999999
            for sp in np.unique(y_train):
                n = sum(y_train==sp)
                if n < low:
                    low = n

            keep = []
            for sp in np.unique(y_train):
                inds = list(y_train[y_train==sp].index)
                inds = random.sample(inds, low)
                keep.extend(inds)

            x_train = x_train[keep]
            y_train = y_train[keep]

        if model_type == 'rf_bal':
            model = BalancedRandomForestClassifier(oob_score=True, max_depth = max_depth, random_state=rseed,
                                               n_estimators = n_estimators, max_features=max_features)

        elif model_type == 'gnb':
            priors = []
            for sp in np.unique(y_train):
                priors.append(sum(y_train==sp)/len(y_train))
            model = GaussianNB(priors=priors)
            
        M = model.fit(x_train, y_train)
        if 'rf' in model_type:
            fi = model.feature_importances_
            oob = model.oob_score_
        predictions = model.predict(x_test)
        probs = model.predict_proba(x_test)

        keep_pred = []

        for i in range(0, len(probs)):
            if np.max(probs[i]) > cf_thr:
                keep_pred.append(True)       
            else:
                keep_pred.append(False)

        x_test = x_test[keep_pred]
        y_test = y_test[keep_pred]
        predictions = predictions[keep_pred]
        probs = probs[keep_pred]

        if len(predictions) < nmin:
            print(len(predictions), nmin)
            continue
        
        x_test_id = x_test_id[keep_pred]
        x_test_rec = x_test_rec[keep_pred]
        x_test_st = x_test_st[keep_pred]

        temp = np.transpose(pd.DataFrame([[enc_test]*len(y_test), x_test_rec, x_test_id, x_test_st, list(y_test), predictions], index=['enc_id', 'rec_id', 'click_id', 'starttime', 'y_test', 'pred']))

        for i in range(0, len(np.unique(y_train))):
            temp[np.unique(y_train)[i]] = probs[:,i]

        if count == 0:
            results = temp.reset_index().drop(columns=['index'])
        else:
            results = results.append(temp).reset_index().drop(columns=['index'])

        train_acc = model.score(x_train,y_train)
        test_acc = model.score(x_test,y_test)
        ntotal = len(x_train) + len(x_test)
        ntrain = len(x_train)
        test_acc_list.append(test_acc)

        if verbose == True:
            print(f'{ntotal} samples split into {int(ntrain/ntotal*100)}% train, {int(100*(1-ntrain/ntotal))}% test')
            print(f'Training accuracy: {train_acc:.3f}')
            print(f'Test accuracy: {test_acc:.3f}')
            if 'rf' in model_type:
                print(f'OOB accuracy: {oob:.3f}')


        #fig, ax = plt.subplots(figsize=(8,5))
        test_acc = np.sum(temp.pred==temp.y_test)/len(temp)
        print(f'Encounter {enc_test} ({list(y_test)[0]}): acc. {round(100*test_acc, 1)}%  (train={len(y_train)}, test={len(y_test)})')
        count += 1

    results = pd.DataFrame(results).reset_index().drop(columns=['index'])
    acc = sum(results.y_test==results.pred)/len(results)
    std = np.std(test_acc_list)
    
    return results, acc, std

def combined_predict(data, verbose=True):
    for i in range(0, len(data)):
        if data.y_test[i] == data.pred_wc[i]:
            acc = 100
        else:
            acc = 0
        print(f'{list(data.y_test)[i]} (enc{list(data.enc_id)[i]}): acc. {acc}% (nc: {data.n_c[i]}, nw: {data.n_w[i]})')

    acc_ovr = np.sum(data.y_test==data.pred_wc)/len(data)
    
    results = data.loc[:, ['enc_id', 'y_test', 'pred_wc']]
    results = results.rename(columns={'pred_wc':'pred'})
    
    if verbose == True:
        print('')
        print(f'{int(np.sum(data.y_test==data.pred_wc))}/{len(data)} correct ({round(acc_ovr, 2)}%)')
    
    return results, acc_ovr

def save_model(data, omitcols, destpath, target='species', n_estimators=500, model_type='rf_bal',
               max_depth=None, max_features='sqrt', downsample=False, cf_thr=0, 
               verbose=False, rseed=42, compress=0):

    df = data.sample(frac=1)
    if omitcols[1] == None:
        x_train = df.iloc[:,omitcols[0]:omitcols[1]].values
        x_test = df.sample(frac=0.1, random_state=rseed).iloc[:,omitcols[0]:omitcols[1]].values
    else:
        x_train = df.iloc[:,omitcols[0]:-omitcols[1]].values
        x_test = df.sample(frac=0.1, random_state=rseed).iloc[:,omitcols[0]:-omitcols[1]].values

    y_train = df[target].reset_index()[target]
    y_test = df.sample(frac=0.1, random_state=rseed)[target].reset_index()[target]  

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if downsample == True:
        nmin = 9999999
        for sp in np.unique(y_train):
            n = sum(y_train==sp)
            if n < nmin:
                nmin = n

        keep = []
        for sp in np.unique(y_train):
            inds = list(y_train[y_train==sp].index)
            inds = random.sample(inds, nmin)
            keep.extend(inds)

        x_train = x_train[keep]
        y_train = y_train[keep]

    if model_type == 'rf_bal':
        model = BalancedRandomForestClassifier(oob_score=True, max_depth = max_depth, random_state=rseed,
                                           n_estimators = n_estimators, max_features=max_features)

    elif model_type == 'gnb':
        priors = []
        for sp in np.unique(y_train):
            priors.append(sum(y_train==sp)/len(y_train))
        model = GaussianNB(priors=priors)

    M = model.fit(x_train, y_train)
    joblib.dump(model, destpath, compress=compress)
    print(f'Saved model to {destpath}')
    
def formatwhistles(data, outpath):
    df_wd = data
    source = []
    for i in range(0, len(df_wd)):
        a = df_wd.Source[i]
        b = a[:a.index('wav')-1] + '_Aw.wav'
        source.append(b)

    startcol = np.where(df_wd.columns=='Source')[0][0]
    table_wd = df_wd.iloc[:,startcol:]
    table_wd['Source'] = source
    for i in range(1,6):
        table_wd.insert(1, f'temp{i}', [0]*len(table_wd))

    for i in range(6,8):
        table_wd[f'temp{i}'] = [0]*len(table_wd)

    for col in table_wd.columns[1:]:
        table_wd = table_wd.rename(columns={col:col.upper()})

    table_wd['Filename'] = table_wd['Source']
    ind = int(np.where(table_wd.columns=='FREQENDSWEEP')[0][0])
    table_wd.insert(ind+1, 'OVERLAP', [0]*len(table_wd))

    table_wd.reset_index().drop(columns=['index'])
    table_wd.to_csv(outpath, sep=',', index=True)
    
def formatclicks(data, outpath, sr=96000, min_idi=0.01, num_ch=1):
    df_cd = data
    selectcols = ['Id', 'channelMap', 'startSeconds', 'duration']
    rename = {'Id':'Selection', 'channelMap':'Channel', 'startSeconds':'Begin Time (s)', 'duration': 'Dur 90% (s)'}

    table_cd = df_cd.loc[:,selectcols].rename(columns=rename)
    table_cd.Channel = (table_cd.Channel/2).astype(int)
    table_cd['Dur 90% (s)'] = table_cd['Dur 90% (s)']/sr
    table_cd.insert(3, 'End Time (s)', table_cd['Begin Time (s)'] + table_cd['Dur 90% (s)'])

    min_idi = 0.010 #minimum allowed inter-detection interval

    table_cd = table_cd.reset_index().drop(columns=['index'])

    idis = [] #add inter-detection interval field
    for i in range(0, len(table_cd)):
        if i == len(table_cd)-1:
            idis.append(1)
        else:
            idi = table_cd['Begin Time (s)'][i+1] - table_cd['Begin Time (s)'][i]
            idis.append(idi)

    table_cd['IDI (s)'] = idis
    table_cd = table_cd[table_cd['IDI (s)'] > min_idi]

    table_cd = table_cd.reset_index().drop(columns=['index'])
    table_cd['Selection'] = list(np.arange(1, len(table_cd)+1, 1))
    table_cd.to_csv(outpath, sep='\t', index=False)
    
def confusionmatrix(results, tag='', save=False, outpath=None, cmap='crest', figsize=(12,8)):

    ovr_testacc = np.sum(results.pred==results.y_test)/len(results)
    #print(f'Overall test accuracy: {round(100*ovr_testacc, 1)}%')

    fig, ax = plt.subplots(figsize=figsize)
    cm = confusion_matrix(list(results.y_test), list(results.pred))
    target_names = list(np.unique(results.y_test))
    totals = []

    for i in range(0, len(cm)):
        totals.append(cm[i,:].sum())
    totals = np.asarray(totals).astype(int)
    totals = totals.reshape((len(cm),1))
    cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
    cm = np.append(cm, totals, axis=1)

    mask = np.zeros((len(target_names), len(target_names)+1))
    mask[:,len(target_names)] = True

    sns.heatmap(cm, cmap=cmap, mask=mask, annot=True, fmt='.1f', 
                xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=100)
    
    plt.title(f'{tag} classification', fontsize=20)
    
    if save == True:
        plt.savefig(outpath)
    
    for (j,i), label in np.ndenumerate(cm):
        if i == len(target_names):
            ax.text(i+0.5, j+0.5, 'n = ' + str(int(label)), 
                    fontdict=dict(ha='center',  va='center',color='black'))      
    
    plt.show()
    
def plot_speciesprob(data, title='', color1='maroon', color2='skyblue'):
    d = {}
    for c in data.columns:
        if len(c) == 3:
            d[c] = data[c].values[0]

    names = list(d.keys())
    values = list(d.values())

    colors = []
    for v in values:
        if v == np.max(values):
            colors.append(color1)
        else:
            colors.append(color2)

    plt.bar(range(len(d)), values, tick_label=names, color=colors)
    plt.title(title, fontsize=15)
    plt.ylabel('Probability of Prediction')
    plt.ylim(0,.5)
    plt.show()
    
def densitycluster(data, omitcols, red=0.1, rseed=42):

    data_ = data.sample(frac=1, random_state=rseed).reset_index().drop(columns=['index'])
    N1 = len(data_)

    if omitcols[1] == None:
        df = data_.iloc[:,omitcols[0]:omitcols[1]]
    else:
        df = data_.iloc[:,omitcols[0]:-omitcols[1]]

    ss = StandardScaler()
    x = ss.fit_transform(df.values)

    pca = PCA(n_components=2, random_state=rseed)
    pca.fit(x)
    X = pca.transform(x)

    data_.insert(0, 'PC2', X[:,0])
    data_.insert(0, 'PC1', X[:,1])

    keep = []
    for sp in np.unique(data_.species):
        temp = data_[data_.species==sp]
        n = int(red*len(temp))
        u_pc1 = np.mean(temp.PC1)
        u_pc2 = np.mean(temp.PC2)
        dist = []
        for i in range(0, len(temp)):
            x = list(temp.PC1)[i]-u_pc1
            y = list(temp.PC2)[i]-u_pc2
            dist.append(np.sqrt((x**2)+(y**2)))
        temp.insert(2, 'dist', dist)
        temp = temp.sort_values(by='dist', ascending=False)
        temp = temp.iloc[n:,:]
        keep.extend(list(temp.index))

    N2 = len(keep)

    print(f'Size of dataset reduced from {N1} to {N2}')

    return data_.iloc[keep,:].iloc[:,2:]

def bindata(data, dtype='whistles', N=5, vartype='band', rseed=42):
    
    if dtype == 'whistles':
        stats = ['species', 'data_source', 'survey', 'enc_id', 'rec_id', 'sel_id', 'n']
        for c in data.columns:
            if c not in stats:
                stats.append(c)
        index = 7
        
    elif dtype == 'clicks':
        stats = ['species', 'data_source', 'survey', 'enc_id', 'rec_id', 'sel_id', 'n']
        for c in data.columns:
            if vartype == 'band':
                if vartype in c:
                    stats.append(c)
            else:
                if c not in stats:
                    stats.append(c)

            
        index = 7

    d = {}
    for var in stats[:index]:
        d[var] = []
    for var in stats[index:]:
        d[f'{var}_u'] = []
        d[f'{var}_s'] = []

    count = 1
    for enc in np.unique(data.enc_id):
        temp = data[data.enc_id==enc].reset_index().drop(columns=['index'])
        temp = temp.sample(frac=1, random_state=rseed)
        while len(temp) > N:
            sub = temp.iloc[:N,:]
            temp = temp.iloc[N:,:]
            d['species'].append(list(temp.species)[0])
            d['data_source'].append(list(temp.data_source)[0])
            d['survey'].append(list(temp.survey)[0])
            d['enc_id'].append(list(temp.enc_id)[0])
            d['rec_id'].append(list(temp.rec_id)[0])
            d['sel_id'].append(list(temp.sel_id)[0])
            d['n'].append(N)
            for var in stats[index:]:
                d[f'{var}_u'].append(np.mean(sub[var]))
                d[f'{var}_s'].append(np.std(sub[var]))
            count += 1
            
    d = pd.DataFrame(d).reset_index().drop(columns=['index'])
    
    return d