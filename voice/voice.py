# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:41:42 2017
@author: Nguyen
"""

import numpy as np
import pandas as pd

import glob
from scipy.io import wavfile
import scipy.fftpack

from sklearn.decomposition import PCA

features = []
labels = []

pca = PCA(n_components = 5)
for filename in glob.glob('Test/*.wav'):
    rate, data = wavfile.read('Test/03-01-01-01-01-01-08.wav') 
    f = filename.split('-')
    labels.append(f[2])
    transform = scipy.fftpack.hilbert(data)
    pcaApply = pca.fit_transform(transform)
    print(pcaApply)
    features.append(pcaApply[0])
    break



print(np.asarray(features))
print(np.asarray(features).shape)
