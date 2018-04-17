import copy
import os

import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
import peakutils.peak

from scipy.fftpack import fft
import activity_predictor

def mean(values):
    return sum(values)/len(values)


def sdt_dev(values):
    a = np.asarray(values)
    return np.std(a)


def autocorr(x, t=1):
    x = np.asarray(x)
    x = x-np.mean(x)
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0, 1]


def fft123(values):
    # Number of sample points
    N = len(values)
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = values
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yclean = 2.0/N * np.abs(yf[0:N//2])

    # ArgMax0 not useful, remove it
    index = np.argmax(yclean)
    yclean[index] = 0

    # ArgMax 1st dominant Freq
    i1 = np.argmax(yclean)
    a1 = yclean[i1]
    yclean[i1] = 0

    # ArgMax 2nd dominant Freq
    i2 = np.argmax(yclean)
    a2 = yclean[i2]
    yclean[i2] = 0

    # ArgMax 3st dominant Freq
    i3 = np.argmax(yclean)
    a3 = yclean[i3]

    return [i1, i2, i3, a1, a2, a3]


def fft1(values):
    return fft123(values)[0]


def fft2(values):
    return fft123(values)[1]


def fft3(values):
    return fft123(values)[2]

def ffta1(values):
    return fft123(values)[3]


def ffta2(values):
    return fft123(values)[4]


def ffta3(values):
    return fft123(values)[5]


def demean(values):
    v = np.asarray(values)
    return v - np.mean(v)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def peaks(values, thres=0):
    z = copy.copy(values)
    z = demean(z)
    # plt.plot(z)
    w = moving_average(z, 4)
    # plt.plot(w)
    indexes = peakutils.peak.indexes(np.array(w), thres=7.0/max(w), min_dist=9)

    if thres > 0:
        new_indexes = list()
        for i in indexes:
            if z[i] > thres:
                new_indexes.append(i)
        return new_indexes
    else:
        return indexes


def peaks_std(values):
    return peaks(values, sdt_dev(values))


def valeys(values, thres=0):
    z = copy.copy(values)
    z = demean(z)
    z = (z * -1)
    # plt.plot(z)
    w = moving_average(z, 4)
    # plt.plot(w)
    indexes = peakutils.peak.indexes(np.array(w), thres=7.0/max(w), min_dist=9)

    if thres > 0:
        new_indexes = list()
        for i in indexes:
            if z[i] > thres:
                new_indexes.append(i)
        return new_indexes
    else:
        return indexes


def valeys_std(values):
    return valeys(values, sdt_dev(values))


def airtime(values):
    values = moving_average(values, 5)
    thres = 1150
    return np.sum(np.asarray(values) < thres) / float(len(values))


def rising_edge(values):
    values = moving_average(values, 5)
    mean_val = mean(values)
    
    val_copy = list()
    for i in xrange(len(values)):
        if values[i] < mean_val:
            val_copy.append(0)
        else:
            val_copy.append(1)

    rising_edges = 0
    for i in xrange(len(val_copy) - 1):
        if val_copy[i] < val_copy[i+1]:
            rising_edges = rising_edges + 1

    return rising_edges


def falling_edge(values):
    values = moving_average(values, 5)
    mean_val = mean(values)
    
    val_copy = list()
    for i in xrange(len(values)):
        if values[i] > mean_val:
            val_copy.append(0)
        else:
            val_copy.append(1)

    falling_edges = 0
    for i in xrange(len(val_copy) - 1):
        if val_copy[i] < val_copy[i+1]:
            falling_edges = falling_edges + 1

    return falling_edges

############# Main method #################
def main(front, back):
    
    feature_names = ['min_f', 'max_f', 'mean_f', 'std_dev_f', 'auto_f',
                     'fft1_f', 'fft2_f', 'fft3_f', 'peaks_f', 'valleys_f',
                     'peaks_std_f', 'valleys_std_f', 'air_f', 'ffta1f', 'ffta2f', 'ffta3f', 're_f', 'fe_f',
                     'min_b', 'max_b', 'mean_b', 'std_dev_b', 'auto_b',
                     'fft1_b', 'fft2_b', 'fft3_b', 'peaks_b', 'valleys_b',
                     'peaks_std_b', 'valleys_std_b', 'air_b', 'ffta1b', 'ffta2b', 'ffta3b', 're_b', 'fe_b',
                     'min_a', 'max_a', 'mean_a', 'std_dev_a', 'auto_a',
                     'fft1_a', 'fft2_a', 'fft3_a', 'peaks_a', 'valleys_a',
                     'peaks_std_a', 'valleys_std_a', 'air_a', 'ffta1a', 'ffta2a', 'ffta3a', 're_a', 'fe_a']

    both = list()

    both = np.array((np.asarray(front) + np.asarray(back))/2)

    (fft1, fft2, fft3, af1, af2, af3) = fft123(front)
    features_front = [min(front), max(front), mean(front), sdt_dev(front),
                      autocorr(front), fft1, fft2, fft3,
                      len(peaks(front)), len(valeys(front)), len(peaks_std(front)), 
                      len(valeys_std(front)), airtime(front), af1, af2, af3, rising_edge(front), falling_edge(front)]

    (fft1, fft2, fft3, af1, af2, af3) = fft123(back)
    features_back = [min(back), max(back), mean(back), sdt_dev(back),
                     autocorr(back), fft1, fft2, fft3,
                     len(peaks(back)), len(valeys(back)), len(peaks_std(back)),
                     len(valeys_std(back)), airtime(back), af1, af2, af3, rising_edge(back), falling_edge(back)]

    (fft1, fft2, fft3, af1, af2, af3) = fft123(both)
    feature_avg = [min(both), max(both), mean(both), sdt_dev(both),
                   autocorr(both), fft1, fft2, fft3,
                   len(peaks(both)), len(valeys(both)), len(peaks_std(both)), 
                   len(valeys_std(both)), airtime(both), af1, af2, af3, rising_edge(both), falling_edge(both)]


    features_all = features_front + features_back + feature_avg

    features_dict = dict()

    for i in range(len(feature_names)):
        features_dict[feature_names[i]] = [features_all[i]]

    activity_predictor.main(features_dict)


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
