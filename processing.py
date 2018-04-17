
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import peakutils.peak

from scipy.fftpack import fft


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


files = list()
folders = ['data_Artur/walking', 'data_Artur/standing_only', 'data_Artur/standing_up', 'data_Artur/squatting', 'data_Artur/sitting_down', 'data_Artur/sitting_only',
 'data_Artur/jumping','data_Advait/walking', 'data_Advait/standing_only', 'data_Advait/standing_up', 'data_Advait/squatting', 'data_Advait/sitting_down',
 'data_Advait/sitting_only', 'data_Advait/jumping','data_Anchan/walking', 'data_Anchan/standing_only', 'data_Anchan/standing_up', 'data_Anchan/squatting', 'data_Anchan/sitting_down',
 'data_Anchan/sitting_only', 'data_Anchan/jumping']

for folder in folders:
    for file_name in os.listdir(folder):
        files.append(folder+'/'+file_name)

for file_name in files:

    #file_name = 'jumping_10_test9.txt'
    #print(file_name)
    file = open(file_name, "r")
    array = file.readlines()
    file.close()

    front = list()
    back = list()
    both = list()

    for i in range(len(array)):
        vals = array[i].split(" ")
        front.append(int(vals[1]))
        back.append(int(vals[2].rstrip('\n')))

    both = np.array((np.asarray(front) + np.asarray(back))/2)

    print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        
        
        # Name
        file_name.split("/")[1],
        
        # Front
        # basics
        min(front), max(front), mean(front), sdt_dev(front),
        # fft and corr
        autocorr(front), fft1(front), fft2(front), fft3(front),
        # peaks and valeys
        #(len(peaks(front)) / float(len(front))), (len(valeys(front)) / float(len(front))),
        len(peaks(front)), len(valeys(front)),
        len(peaks_std(front)), len(valeys_std(front)),
        # airtime
        airtime(front), ffta1(front), ffta2(front), ffta3(front), rising_edge(front), falling_edge(front),

    	# Back
    	# basics
        min(back), max(back), mean(back), sdt_dev(back),
        # fft and corr
        autocorr(back), fft1(back), fft2(back), fft3(back),
        # peaks and valeys
        len(peaks(back)), len(valeys(back)),
        len(peaks_std(back)), len(valeys_std(back)),
        # airtime
        airtime(back), ffta1(back), ffta2(back), ffta3(back), rising_edge(back), falling_edge(back),

    	# Both
    	# basics
        min(both), max(both), mean(both), sdt_dev(both),
        # fft and corr
        autocorr(both), fft1(both), fft2(both), fft3(both),
        # peaks and valeys
        len(peaks(both)), len(valeys(both)),
        len(peaks_std(both)), len(valeys_std(both)),
        # airtime
        airtime(both), ffta1(both), ffta2(both), ffta3(both), rising_edge(both), falling_edge(both)
    ))
