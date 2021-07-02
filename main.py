#!/usr/bin/env python
# coding: utf-8

# # summary data 
# - record 04015 ([1:8])
# - record 04043 ([1:16])
# - record 04048 ([1:6])
# - record 04126 ([1:])
# - record 04746 (tidak di proses, karena kebanyakan data N)
# - record 04908 ([1:]) -> annotate.txt nya di custom
# - record 04936 ([4:])
# - record 05091 ([1:])
# - record 05121 ([1:])
# - record 05261 ([1:18])
# - record 06426 ([1:])
# - record 06453 ([1:])
# - record 06995 ([1:])
# - record 07162 (tidak di proses, isinya hanya AF)
# - record 07859 (tidak di proses, isinya hanya AF)
# - record 07879 (tidak di proses, karena kebanyakan data N)
# - record 07910 ([1:10])
# - record 08215 ([1:])
# - record 08219 ([1:])
# - record 08378 ([5:])
# - record 08405 (tidak di proses, karena kebanyakan data N)
# - record 08434 (tidak di proses, karena semua data N)
# - record 08455 ([1:])

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fs = 250 #hz

print("[INFO] read annotation file")
record = "04015"

dataset_dir = "dataset/AFDB record_%s/" % record

csv_filenames = []
for filename in os.listdir(dataset_dir) :
    if filename.find(".csv") > -1:
        csv_filenames.append(filename)

file = open(dataset_dir + 'annotation.txt',"r") 
annotations = file.readlines()
file.close()

label_idx = []
for item in annotations[1:8] :
    item_split = item.split()
    label_idx.append([item_split[0].replace("[", "").replace("]", ""), item_split[-1].replace("(", "")])

print("[INFO] read csv file")
def read_csv_to_df(filename, folder, sep=";"):
    df = pd.read_csv(folder + filename, sep=sep)
    print("[INFO] finish read file - %s" % filename)
    
    #df = df.drop(0) 
    df.columns = ['Time', 'ECG1', 'ECG2']

    df['ECG1'] = pd.to_numeric(df['ECG1'])
    df['ECG2'] = pd.to_numeric(df['ECG2'])
    
    # peak reduction
    df[df['ECG1'] > 2] = 2
    df[df['ECG1'] < -2] = -2
    df[df['ECG2'] > 2] = 2
    df[df['ECG2'] < -2] = -2
    print("[INFO] finish data cleansing - %s" % filename)

    df["Time"] = df['Time'].str.replace("[", "")
    df["Time"] = df['Time'].str.replace("]", "")
    df["Time"] = df['Time'].str.replace("'", "")

    df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
    print("[INFO] finish time cleansing -  %s" % filename)
    
    df.set_index("Time", inplace=True)
    return df


list_df_ecg = []
for name in csv_filenames:
    df = read_csv_to_df(name, dataset_dir)
    list_df_ecg.append(df)
    
df_ecg = pd.concat(list_df_ecg)


print("[INFO] Split time range AFIB - N")
N_range = []
AFIB_range = []

for i in range(len(label_idx) - 1):
    tm_str = label_idx[i][0]
    next_tm_str = label_idx[i + 1][0]
    tm = pd.to_datetime(tm_str)
    next_tm = pd.to_datetime(next_tm_str)
    
    if label_idx[i][1] == 'N' :
        N_range.append([tm, next_tm])
    else :
        AFIB_range.append([tm, next_tm])

N = []
for nr in N_range :
    result = df_ecg.between_time(nr[0].time(), nr[1].time())
    N.append(result)

AFIB = []
for ar in AFIB_range :
    result = df_ecg.between_time(ar[0].time(), ar[1].time())
    AFIB.append(result)


print("[INFO] Split CSV file based time range AFIB - N per-16s & apply Baseline Wander Removal using ALS")
# - split each N & AFIB dataframe to 16s sequence and apply Baseline Removal 
from scipy import sparse
from scipy.sparse.linalg import spsolve
from datetime import timedelta

def baseline_als(y, lam=10000, p=0.05, n_iter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
        
time_interval_N = []
for N_item in N:
    intr = [time_result for time_result in perdelta(N_item.index[0], N_item.index[-1], timedelta(seconds=16))]
    time_interval_N.append(intr)

time_interval_AFIB = []
for AFIB_item in AFIB:
    intr = [time_result for time_result in perdelta(AFIB_item.index[0], AFIB_item.index[-1], timedelta(seconds=16))]
    time_interval_AFIB.append(intr)

ECG_ALS = []
ECG_ALS_label = []

for time_interval in time_interval_N :
    for time_intv in list(zip(time_interval, time_interval[1:])):
        X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
        ecg1 = X['ECG1'].values
        ecg2 = X['ECG2'].values
        
        if len(ecg1) > 0 and len(ecg2) > 0:
            ALS1 = ecg1 - baseline_als(ecg1)
            ALS2 = ecg2 - baseline_als(ecg2)

            ECG_ALS.append(np.array([ALS1, ALS2]))
            ECG_ALS_label.append('N')
        
for time_interval in time_interval_AFIB :
    for time_intv in list(zip(time_interval, time_interval[1:])):
        X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
        ecg1 = X['ECG1'].values
        ecg2 = X['ECG2'].values
        
        if len(ecg1) > 0 and len(ecg2) > 0:
            ALS1 = ecg1 - baseline_als(ecg1)
            ALS2 = ecg2 - baseline_als(ecg2)

            ECG_ALS.append(np.array([ALS1, ALS2]))
            ECG_ALS_label.append('AF')

            
print("[INFO] Signal Normalization from -1 to 1")
# - Signal normalization from -1 to 1
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

scaler = MaxAbsScaler()

ECG_ALS_Norm = []

for als in ECG_ALS :
    als1 = np.expand_dims(als[0], 1)
    als2 = np.expand_dims(als[1], 1)
    
    scaler.fit(als1)
    
    als_norm1 = scaler.transform(als1)
    als_norm2 = scaler.transform(als2)
    
    ECG_ALS_Norm.append([als_norm1, als_norm2])

print("[INFO] R-R peak detection & split 16s signal to 1.2R-R peak")
# - QRS Detection
from ecgdetectors import Detectors

detectors = Detectors(fs)

# - Split each 16s to 1.2 x R-R sequence
# - Padding the sequence with zero for length 300 point
ECG_split = []
ECG_split_label = []
for i in range(len(ECG_ALS_Norm)) :
    data = np.array(ECG_ALS_Norm[i])
    if len(data) > 0:
        r_peaks = []
        try :
            r_peaks = detectors.christov_detector(data[0])
        except :
            print("cannot find R peaks in ALS Norm, idx %d" % i)
        RRs = np.diff(r_peaks)
        RRs_med = np.median(RRs)
        if not np.isnan(RRs_med) and RRs_med > 0:
            for rp in r_peaks :
                split1 = data[0][:,0][rp : rp + int(RRs_med * 1.2)] 
                split2 = data[1][:,0][rp : rp + int(RRs_med * 1.2)] 
                
                n1 = len(split1) if len(split1) <= 300 else 300
                n2 = len(split2) if len(split2) <= 300 else 300
                pad1 = np.zeros(300)
                pad2 = np.copy(pad1)
                pad1[0:n1] = split1[0:n1]
                pad2[0:n2] = split2[0:n2]
                ECG_split.append([pad1, pad2])
                ECG_split_label.append(ECG_ALS_label[i])


print("[INFO] Save Preprocessed File as CSV")
data = []
for i in range(len(ECG_split)):
    x = list(ECG_split[i][0])
    x.extend(list(ECG_split[i][1]))
    x.append(ECG_split_label[i])
    data.append(x)

ECG = pd.DataFrame(data)
ECG.to_csv("dataset/AFDB_%s_sequence_300_pt_2_ch.csv" % record, index=False, header=False)




