#!/usr/bin/env python
# coding: utf-8
print("[INFO] Import Library...")
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import shutil

import warnings
warnings.filterwarnings('ignore')

def preprocessing_AFDB(record, start=1, stop=None, sep=",", fs=250, sample_size=6):
    dataset_dir = "dataset/AFDB record_%s/" % record
    csv_filenames = []
    for filename in os.listdir(dataset_dir) :
        if filename.find(".csv") > -1:
            csv_filenames.append(filename)
    print("[INFO] detected CSV file :", csv_filenames)
            
    print("[INFO] Read annotation file...")
    file = open(dataset_dir + 'annotation.txt',"r") 
    annotations = file.readlines()
    file.close()

    label_idx = []
    for item in annotations[start:stop] :
        item_split = item.split()
        label_idx.append([item_split[0].replace("[", "").replace("]", ""), item_split[-1].replace("(", "")])

    print("[INFO] Read CSV...")
    # - Read & formatting ECG data
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


    # - concate datafarame
    list_df_ecg = []
    for name in csv_filenames:
        df = read_csv_to_df(name, dataset_dir, sep=sep)
        list_df_ecg.append(df)

    df_ecg = pd.concat(list_df_ecg)
    label_idx.append([str(df_ecg.index[-1].time()), ''])

    # - Split Normal (N) and AFIB data
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
    
    if not os.path.exists("dataset_split_per_class"):
        os.mkdir("dataset_split_per_class")
    
    N = []
    for ix, nr in enumerate(N_range) :
        result = df_ecg.between_time(nr[0].time(), nr[1].time())
        result.to_csv("dataset_split_per_class/%s_%s_%s_%s.csv" % 
                      ('N', record, 'ECG1', ix))
        N.append(result)

    AFIB = []
    for ix, ar in enumerate(AFIB_range) :
        result = df_ecg.between_time(ar[0].time(), ar[1].time())
        result.to_csv("dataset_split_per_class/%s_%s_%s_%s.csv" % 
                      ('AF', record, 'ECG1', ix))
        AFIB.append(result)


    print("[INFO] Split per-%ss & apply Baseline Wander Removal" % sample_size)
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
        if len(N_item) > 0:
            intr = [time_result for time_result in perdelta(N_item.index[0], N_item.index[-1], timedelta(seconds=sample_size))]
            time_interval_N.append(intr)


    time_interval_AFIB = []
    for AFIB_item in AFIB:
        if len(AFIB_item) > 0:
            intr = [time_result for time_result in perdelta(AFIB_item.index[0], AFIB_item.index[-1], timedelta(seconds=sample_size))]
            time_interval_AFIB.append(intr)

    ECG_ALS = []
    ECG_ALS_label = []

    for time_interval in time_interval_N :
        for time_intv in list(zip(time_interval, time_interval[1:])):
            X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
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
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
                ecg1 = X['ECG1'].values
                ecg2 = X['ECG2'].values

                if len(ecg1) > 0 and len(ecg2) > 0:
                    ALS1 = ecg1 - baseline_als(ecg1)
                    ALS2 = ecg2 - baseline_als(ecg2)

                    ECG_ALS.append(np.array([ALS1, ALS2]))
                    ECG_ALS_label.append('AF')


    print("[INFO] Signal Normalization...")
    # - Signal normalization from -1 to 1
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

    scaler = MaxAbsScaler()
    ECG_ALS_Norm = []

    for als in ECG_ALS :
        als1 = np.expand_dims(als[0], 1)
        als2 = np.expand_dims(als[1], 1)

        scaler.fit(als1)

        als_norm1 = scaler.transform(als1)
        als_norm2 = scaler.transform(als2)

        ECG_ALS_Norm.append([als_norm1, als_norm2])
    
    print("[INFO] Save preprocessed data to CSV file for record %s..." % record)
    data = []
    pad_size = sample_size*fs # 16s x 250hz
    for i in range(len(ECG_ALS_Norm)):
        signal_ch = []
        for ch in [0, 1] :
            signal = np.array(ECG_ALS_Norm[i])[ch, :, 0]
            n = len(signal) if len(signal) <= pad_size else pad_size
            pad = np.zeros(pad_size)
            pad[0:n] = signal[0:n] 
            signal_ch.extend(list(pad))    
        signal_ch.append(ECG_ALS_label[i])
        data.append(signal_ch)

    ECG = pd.DataFrame(data)
    ECG.to_csv("dataset/AFDB_%s_sequence_%ds_2_ch.csv" % (record, sample_size), index=False, header=False)

    print("-------------------------- *** --------------------------\n\n")

def preprocessing_NSRDB(record, fs = 128, sample_size=6):
    dataset_dir = "dataset/NSRDB/%s/" % record 
    
    csv_filenames = []
    for filename in os.listdir(dataset_dir) :
        if filename.find(".csv") > -1:
            csv_filenames.append(filename)
    print("[INFO] detected CSV file :", csv_filenames)
            
    print("[INFO] Read CSV...")
    def read_csv_to_df(filename, folder, sep=","):
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

    print("[INFO] Split per-16s & apply Baseline Wander Removal")
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
            
    time_interval = []
    if len(df_ecg) > 0:
        intr = [time_result for time_result in perdelta(df_ecg.index[0], df_ecg.index[-1], timedelta(seconds=sample_size))]
        time_interval.append(intr)
        
    ECG_ALS = []
    ECG_ALS_label = []

    for tm_int in time_interval :
        for time_intv in list(zip(tm_int, tm_int[1:])):
            X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
                ecg1 = X['ECG1'].values
                ecg2 = X['ECG2'].values

                if len(ecg1) > 0 and len(ecg2) > 0:
                    ALS1 = ecg1 - baseline_als(ecg1)
                    ALS2 = ecg2 - baseline_als(ecg2)

                    ECG_ALS.append(np.array([ALS1, ALS2]))
                    ECG_ALS_label.append('N')
                
    print("[INFO] Signal Normalization...")
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
    scaler = MaxAbsScaler()
    ECG_ALS_Norm = []

    for als in ECG_ALS :
        als1 = np.expand_dims(als[0], 1)
        als2 = np.expand_dims(als[1], 1)

        scaler.fit(als1)

        als_norm1 = scaler.transform(als1)
        als_norm2 = scaler.transform(als2)

        ECG_ALS_Norm.append([als_norm1, als_norm2])
        
    print("[INFO] upsampling signal to 250Hz ...")
    def upsampling_twice(data):
        # upsampling interpolation
        result = np.zeros(2*len(data)-1)
        result[0::2] = data
        result[1::2] = (data[1:] + data[:-1]) / 2
        return result
    
    new_fs = 250 # Hz 
    ECG_ALS_Norm_Up = []
    for data in ECG_ALS_Norm :
        data1 = np.array(data[0][:,0])
        data2 = np.array(data[1][:,0])
        data1 = upsampling_twice(data1).reshape(-1, 1)  
        data2 = upsampling_twice(data2).reshape(-1, 1)  
        ECG_ALS_Norm_Up.append([data1, data2])
        
    print("[INFO] Save preprocessed data to CSV file for record %s..." % record)
    data = []
    pad_size = sample_size*new_fs # 16s x 250hz
    for i in range(len(ECG_ALS_Norm_Up)):
        signal_ch = []
        for ch in [0, 1] :
            signal = np.array(ECG_ALS_Norm_Up[i])[ch, :, 0]
            n = len(signal) if len(signal) <= pad_size else pad_size
            pad = np.zeros(pad_size)
            pad[0:n] = signal[0:n] 
            signal_ch.extend(list(pad))    
        signal_ch.append(ECG_ALS_label[i])
        data.append(signal_ch)
        
    ECG = pd.DataFrame(data)
    ECG.to_csv("dataset/NSRDB_%s_sequence_%ss_2_ch.csv" % (record, sample_size), index=False, header=False)
    print("-------------------------- *** --------------------------\n\n")    


def merging_dataset(fs=250, sample_size=6):
    dataset_folder = 'dataset/'
    print("[INFO] read all AFDB dataset...")
    filenames = []
    for filename in os.listdir(dataset_folder):
        if filename.find("sequence_%ss_2_ch.csv" % sample_size) > -1 and filename.find("AFDB_") > -1:
            filenames.append(filename)
   
    dfs = []
    for name in filenames :
        try :
            print("[INFO] reading file ", name)
            df = pd.read_csv(dataset_folder + name, header=None)
            dfs.append(df)
        except Exception as e :
            print("[ERROR] failed to read file ", name, ", empty!")
            
    print("[INFO] merging all AFDB dataset...")
    dfs_all = pd.concat(dfs, ignore_index=True)
    dfs_all.to_csv(dataset_folder + "AFDB_all.csv", index=None, header=None)
    
    print("[INFO] read all NSRDB dataset...")
    filenames = []
    for filename in os.listdir(dataset_folder):
        if  filename.find("sequence_%ss_2_ch.csv" % sample_size) > -1 and filename.find("NSRDB_") > -1:
            filenames.append(filename)
            
    normal_dfs = []
    for name in filenames :
        try :
            print("[INFO] reading file ", name)
            normal_df = pd.read_csv(dataset_folder + name, header=None)
            normal_dfs.append(normal_df)
        except Exception as e :
            print("[ERROR] failed to read file ", name, ", empty!")
            
    print("[INFO] merging all NSRDB dataset...")
    normal_dfs_all = pd.concat(normal_dfs, ignore_index=True)
    normal_dfs_all.to_csv(dataset_folder + "NSRDB_all.csv", index=None, header=None)
    
    print("[INFO] removing normal dataset from AFDB...")
    dfs_all_AF = dfs_all[dfs_all[fs*sample_size*2] == 'AF']
    
    print("[INFO] merge AFDB & NSRDB dataset...")
    dfs_AF_N = pd.concat([dfs_all_AF, normal_dfs_all])
    
    print("[INFO] balancing dataset after merging AFDB & NSRDB Dataset...")
    dfs_AF_N[fs*sample_size*2]=dfs_AF_N[fs*sample_size*2]
    equilibre=dfs_AF_N[fs*sample_size*2].value_counts()
    print(equilibre)
    
    from sklearn.utils import resample

    dfs = []
    n_samples = equilibre.mean().astype(np.int0())
    print("[INFO] Resample to", n_samples)
    for i, key in enumerate(equilibre.keys()):
        dfs.append(dfs_AF_N[dfs_AF_N[fs*sample_size*2]==key])
        dfs[i]=resample(dfs[i],replace=True,n_samples=n_samples,random_state=123)
    df_AF_N_balanced =pd.concat(dfs)
    
    print("[INFO] Save balanced AFDB & NSRDB Dataset to csv...")
    df_AF_N_balanced.to_csv(dataset_folder + "AFDB_NSRDB_all.csv", index=None, header=None)
    
    print("-------------------------- *** --------------------------\n\n")
    
def denoising(fs=250, sample_size=6):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    print("[INFO] load final dataset...")
    dataset_folder = 'dataset/'
    ecg_dfs = pd.read_csv(dataset_folder + "AFDB_NSRDB_all.csv", header=None)
    
    print("[INFO] Label encoding...")
    X = ecg_dfs.iloc[:,:fs*sample_size*2].values
    y = ecg_dfs.iloc[:,fs*sample_size*2].values

    le = LabelEncoder()
    le.fit(y)

    labels = le.classes_
    print("[INFO] Categorical label : ", labels)

    y = le.transform(y)
    
    print("[INFO] Split Dataset...")
    test_split_size = 0.33 # 33% for testing, 67% for training
    print("[INFO] %d%% for testing, %d%% for training..." % (test_split_size*100, (1-test_split_size)*100))
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_split_size, random_state=42)
    print("[INFO] dataset after splitting,\n", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    def add_AWGN_noise(signal, target_noise_db = -20):
        mean_noise = 0
        target_noise_watts = 10 ** (target_noise_db / 10)
        sigma = np.sqrt(target_noise_watts)

        noise = np.random.normal(mean_noise, sigma, len(signal))

        return (signal+noise)
    
    def scaler(X):
        res = []
        for x in X :
            global_min = x.min()
            x = np.reshape(x, (2, sample_size*fs))
            for i in range(len(x)):
                idx = np.max(np.nonzero(x[i]))
                x[i][idx+1:] = global_min
            x = np.reshape(x, (sample_size*fs*2))
            res.append((x - x.min())/(x.max() - x.min()))
        return np.array(res)
    
    print("[INFO] inject noise to dataset...")
    
    X_train = scaler(X_train)
    X_test = scaler(X_test)
    
    X_train_noised = np.array([add_AWGN_noise(signal) for signal in X_train])
    X_test_noised = np.array([add_AWGN_noise(signal) for signal in X_test])
    
    def calc_snr(signal, noised_signal):
        noise = np.array(noised_signal - signal)
        std_noise = noise.std(axis=1)
        signal_avg = signal.mean(axis=1)

        SNR  =  np.where(signal_avg <= 0, 1, signal_avg/std_noise)
        SNR_db = 10*np.log(SNR)

        return SNR_db
    
    def calc_psnr(signal, noised_signal, max_peak=1):
        noise = np.array(noised_signal - signal)
        mse = (np.square(signal - noise)).mean(axis=1)
        SNR = np.where(mse == 0, 0, max_peak/mse)
        SNR_db = 10*np.log(SNR)
        return SNR_db
    
    print("\n------ SNR Noised Signal ------")
    SNR_db = calc_snr(X_train, X_train_noised)
    for i in range(len(X_train)):
        if i % int(len(X_train)/10) == 0:
            print('Sample %d \t : SNR %.4f db' % (i, SNR_db[i]))
    
    print("\n------ PSNR Noised Signal ------")
    PSNR_db = calc_psnr(X_train, X_train_noised)
    for i in range(len(X_train)):
        if i % int(len(X_train)/10) == 0:
            print('Sample %d \t : PSNR %.4f db' % (i, PSNR_db[i]))
    
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    
    print("\n\n")
    print("[INFO] **************************************************")    
    print("[INFO] build model Convolutional Autoencoder...")
    # Convolution Autoencoder (CNN AE)
    from keras.models import Sequential
    from keras.layers import Dense, InputLayer
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras import backend as K
    from keras.layers import Conv1D, MaxPooling1D as MaxP1D, UpSampling1D as UpSm1D
    
    X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
    X_train_noised = X_train_noised.reshape(len(X_train_noised), X_train_noised.shape[1], 1)
    X_test_noised = X_test_noised.reshape(len(X_test_noised), X_test_noised.shape[1], 1)
    
    def ConvAutoEncoder(input_dim):
        conv_net = Sequential(name="conv_autoencoder")
        conv_net.add(Conv1D(256, 3, activation='relu', padding='same', name="encode_1", input_shape=(input_dim,1)))
        conv_net.add(MaxP1D(2, padding='same', name="encode_2"))
        conv_net.add(Conv1D(128, 3, activation='relu', padding='same', name="encode_3"))
        conv_net.add(MaxP1D(2, padding='same', name="encode_4"))
        conv_net.add(Conv1D(64, 3, activation='relu', padding='same', name="encode_5"))
        conv_net.add(MaxP1D(2, padding='same', name="encode_6"))

        conv_net.add(Conv1D(64, 3, activation='relu', padding='same', name="decode_1"))
        conv_net.add(UpSm1D(2, name="decode_2"))
        conv_net.add(Conv1D(128, 3, activation='relu', padding='same', name="decode_3"))
        conv_net.add(UpSm1D(2, name="decode_4"))
        conv_net.add(Conv1D(256, 3, activation='relu', padding='same', name="decode_5"))
        conv_net.add(UpSm1D(2, name="decode_6"))
        conv_net.add(Conv1D(1, 3, activation='sigmoid', padding='same', name="decode_7"))

        conv_net.summary()

        conv_net.compile(
                        optimizer = 'adam', 
                        loss = rmse)

        return conv_net

    def model_fit(model, name, X_train_noised, X_train,  X_test_noised, X_test, epochs = 10, batch_size = 32):
        callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5),
                    ModelCheckpoint(
                             filepath = "best_" + name, 
                             monitor = 'val_loss',
                             save_best_only = True)]

        return model.fit(X_train_noised, X_train,
                        epochs = epochs,
                        batch_size = batch_size,
                        callbacks = callbacks,
                        shuffle = True,
                        validation_data = (X_test_noised, X_test))
    
    print("[INFO] train model Convolutional Autoencoder...")
    model_name_conv_AE = 'denoising_conv_AE.h5'

    input_dim = X_train_noised.shape[1]
    conv_autoencoder = ConvAutoEncoder(input_dim)

    history =  model_fit(conv_autoencoder, 
                         model_name_conv_AE, 
                         X_train_noised, X_train,  
                         X_test_noised, X_test, 
                         epochs = 10, batch_size = 32)

    conv_autoencoder.save(model_name_conv_AE)
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("history_Training_Denoising_Conv_AE.csv", index=False)
    
    print("[INFO] Denoising all noised signal using Convolution Autoencoder...")
    X_train_denoised = conv_autoencoder.predict(X_train_noised)
    X_test_denoised = conv_autoencoder.predict(X_test_noised)
    
    print("\n------ SNR Denoised Signal ConvAE------")
    SNR_db = calc_snr(X_train, X_train_denoised)
    for i in range(len(X_train)):
        if i % int(len(X_train)/10) == 0:
            print('Sample %d \t : SNR %.4f db' % (i, SNR_db[i]))

    print("\n------ PSNR Noised Signal ConvAE------")
    PSNR_db = calc_psnr(X_train, X_train_denoised)
    for i in range(len(X_train)):
        if i % int(len(X_train)/10) == 0:
            print('Sample %d \t : PSNR %.4f db' % (i, PSNR_db[i]))
       
    print("[INFO] Save denoised dataset - Convolution Autoencoder...")
    X_train_denoised = X_train_denoised.reshape(len(X_train_denoised), X_train_denoised.shape[1])
    X_test_denoised = X_test_denoised.reshape(len(X_test_denoised), X_test_denoised.shape[1])

    train_denoised_df = pd.DataFrame(np.hstack((X_train_denoised, np.expand_dims(y_train, 1))))
    train_denoised_df.to_csv(dataset_folder + "train_all_Conv_AE.csv", index=None, header=None)

    test_denoised_df = pd.DataFrame(np.hstack((X_test_denoised, np.expand_dims(y_test, 1))))
    test_denoised_df.to_csv(dataset_folder + "test_all_Conv_AE.csv", index=None, header=None)

    print("-------------------------- *** --------------------------\n\n")
    
def feature_extraction(fs = 250, sample_size = 6, label_name = ['AF', 'N']):
    global nk
    try :
        import neurokit2 as nk
        print("[INFO] using Neuro-Kit :", nk.__version__)
    except :
        print('[ERROR] neurokit2 library not found. install using conda `pip install neurokit2`...')
        return False
        
    dataset_folder = 'dataset/'
    filenames = []
    for filename in os.listdir(dataset_folder):
        if filename.find("_all_") > -1:
            filenames.append(filename)
    
    print("[INFO] Load denoised dataset...")        
    test_df = pd.read_csv(dataset_folder + "test_all_Conv_AE.csv", header=None)
    train_df = pd.read_csv(dataset_folder + "train_all_Conv_AE.csv", header=None)
    
    print("[INFO] Convert Dataset from Dataframe to Array...")
    # convert dataframe to numpy array
    test_data = np.array(test_df.values.tolist())
    train_data = np.array(train_df.values.tolist())

    # split input & label data
    test_label = test_data[:, -1]
    test_x = test_data[:, :-1]

    train_label = train_data[:, -1]
    train_x = train_data[:, :-1]

    # reshape input data
    test_signal = []
    for item in test_x :
        test_signal.append(item.reshape(2, fs*sample_size, -1))
        
    train_signal = []
    for item in train_x :
        train_signal.append(item.reshape(2, fs*sample_size, -1))

    print("[INFO] find R-R Interval for test dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    RR_Interval_test = []
    RR_Interval_label_test = []
    for i in range(len(test_signal)) :
        ecg_signal = np.array(test_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    rr_intv = np.diff(r_peaks['ECG_R_Peaks'])
                    if len(rr_intv) > pad_size :
                        print("[INFO] number of peak more than %d : %d" % (pad_size, len(rr_intv)))
                    n = len(rr_intv) if len(rr_intv) <= pad_size else pad_size
                    
                    pad = np.zeros(pad_size)
                    pad[0:n] = rr_intv[0:n]
                    signal_ch.append(pad)
                RR_Interval_test.append(signal_ch)
                RR_Interval_label_test.append(test_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d  : %s" % (i, e))
                
    print("[INFO] find R-R Interval for train dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    RR_Interval_train = []
    RR_Interval_label_train = []
    for i in range(len(train_signal)) :
        ecg_signal = np.array(train_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    rr_intv = np.diff(r_peaks['ECG_R_Peaks'])
                    if len(rr_intv) > pad_size :
                        print("[INFO] number of peak more than %d : %d" % (pad_size, len(rr_intv)))
                    n = len(rr_intv) if len(rr_intv) <= pad_size else pad_size
                    
                    pad = np.zeros(pad_size)
                    pad[0:n] = rr_intv[0:n]
                    signal_ch.append(pad)
                RR_Interval_train.append(signal_ch)
                RR_Interval_label_train.append(train_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d : %s" % (i, e))
    
    print("[INFO] save R-R Interval feature dataset...")
    # save test dataset
    data_test = []
    for i in range(len(RR_Interval_test)):
        x = list(RR_Interval_test[i][0])
        x.extend(list(RR_Interval_test[i][1]))
        x.append(RR_Interval_label_test[i])
        data_test.append(x)
        
    # save train dataset
    data_train = []
    for i in range(len(RR_Interval_train)):
        x = list(RR_Interval_train[i][0])
        x.extend(list(RR_Interval_train[i][1]))
        x.append(RR_Interval_label_train[i])
        data_train.append(x)  
        
    ECG_test = pd.DataFrame(data_test)
    ECG_train = pd.DataFrame(data_train)        
    
    # save R-R Interval 
    ECG_test.to_csv("dataset/test_all_feature_rr_interval.csv", index=False, header=False)
    ECG_train.to_csv("dataset/train_all_feature_rr_interval.csv", index=False, header=False)   


    print("[INFO] find QRS Complex for test dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    QRS_Complex_test = []
    QRS_Complex_label_test = []
    for i in range(len(test_signal)) :
        ecg_signal = np.array(test_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    _, waves_peak = nk.ecg_delineate(ecg_signal[ch, :, 0], r_peaks, sampling_rate=fs, method="dwt")
                    r_onsets = waves_peak['ECG_R_Onsets']
                    r_offsets = waves_peak['ECG_R_Offsets']
                    
                    qrs_complex =  np.nan_to_num(np.diff(np.array([r_onsets, r_offsets]).T))[:, 0]
                    
                    n = len(qrs_complex) if len(qrs_complex) <= pad_size else pad_size
                    pad = np.zeros(pad_size)
                    pad[0:n] = qrs_complex[0:n]  
                    signal_ch.append(pad)
                QRS_Complex_test.append(signal_ch)
                QRS_Complex_label_test.append(test_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d  : %s" % (i, e))  

    print("[INFO] find QRS Complex for train dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    QRS_Complex_train = []
    QRS_Complex_label_train = []
    for i in range(len(train_signal)) :
        ecg_signal = np.array(train_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    _, waves_peak = nk.ecg_delineate(ecg_signal[ch, :, 0], r_peaks, sampling_rate=fs, method="dwt")
                    r_onsets = waves_peak['ECG_R_Onsets']
                    r_offsets = waves_peak['ECG_R_Offsets']
                    
                    qrs_complex =  np.nan_to_num(np.diff(np.array([r_onsets, r_offsets]).T))[:, 0]
                    
                    n = len(qrs_complex) if len(qrs_complex) <= pad_size else pad_size
                    pad = np.zeros(pad_size)
                    pad[0:n] = qrs_complex[0:n]  
                    signal_ch.append(pad)
                QRS_Complex_train.append(signal_ch)
                QRS_Complex_label_train.append(train_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d  : %s" % (i, e)) 

    print("[INFO] save QRS Complex feature dataset...")
    # save test dataset
    data_test = []
    for i in range(len(QRS_Complex_test)):
        x = list(QRS_Complex_test[i][0])
        x.extend(list(QRS_Complex_test[i][1]))
        x.append(QRS_Complex_label_test[i])
        data_test.append(x)
        
    # save train dataset
    data_train = []
    for i in range(len(QRS_Complex_train)):
        x = list(QRS_Complex_train[i][0])
        x.extend(list(QRS_Complex_train[i][1]))
        x.append(QRS_Complex_label_train[i])
        data_train.append(x)
        
    ECG_test = pd.DataFrame(data_test)
    ECG_train = pd.DataFrame(data_train)

    # save QRS Complex 
    ECG_test.to_csv("dataset/test_all_feature_qrs_complex.csv", index=False, header=False)
    ECG_train.to_csv("dataset/train_all_feature_qrs_complex.csv", index=False, header=False)
    
    
    print("[INFO] find QT Interval for test dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    QT_Interval_test = []
    QT_Interval_label_test = []
    for i in range(len(test_signal)) :
        ecg_signal = np.array(test_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    _, waves_peak = nk.ecg_delineate(ecg_signal[ch, :, 0], r_peaks, sampling_rate=fs, method="dwt")
                    r_onsets = waves_peak['ECG_R_Onsets']
                    t_offsets = waves_peak['ECG_T_Offsets']
                    
                    qt_interval =  np.nan_to_num(np.diff(np.array([r_onsets, t_offsets]).T))[:, 0]
                    
                    n = len(qt_interval) if len(qt_interval) <= pad_size else pad_size
                    pad = np.zeros(pad_size)
                    pad[0:n] = qt_interval[0:n]  
                    signal_ch.append(pad)
                QT_Interval_test.append(signal_ch)
                QT_Interval_label_test.append(test_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d  : %s" % (i, e))  

    print("[INFO] find QT Interval for train dataset...")
    pad_size = 15 # set 15 if using 6s, set 50 if using 16s
    QT_Interval_train = []
    QT_Interval_label_train = []
    for i in range(len(train_signal)) :
        ecg_signal = np.array(train_signal[i])
        if len(ecg_signal) > 0:
            r_peaks = []
            try :
                signal_ch = []
                for ch in [0, 1]:
                    _, r_peaks = nk.ecg_peaks(ecg_signal[ch, :, 0], sampling_rate=fs)
                    if len(r_peaks['ECG_R_Peaks']) < 2 :
                        raise "R Peaks not found"
                    
                    _, waves_peak = nk.ecg_delineate(ecg_signal[ch, :, 0], r_peaks, sampling_rate=fs, method="dwt")
                    r_onsets = waves_peak['ECG_R_Onsets']
                    t_offsets = waves_peak['ECG_T_Offsets']
                    
                    qt_interval =  np.nan_to_num(np.diff(np.array([r_onsets, t_offsets]).T))[:, 0]
                    
                    n = len(qt_interval) if len(qt_interval) <= pad_size else pad_size
                    pad = np.zeros(pad_size)
                    pad[0:n] = qt_interval[0:n]  
                    signal_ch.append(pad)
                QT_Interval_train.append(signal_ch)
                QT_Interval_label_train.append(train_label[i])
            except Exception as e:
                print("[ERROR] processing data in idx %d  : %s" % (i, e)) 

    print("[INFO] save QT Interval feature dataset...")
    # save test dataset
    data_test = []
    for i in range(len(QT_Interval_test)):
        x = list(QT_Interval_test[i][0])
        x.extend(list(QT_Interval_test[i][1]))
        x.append(QT_Interval_label_test[i])
        data_test.append(x)
        
    # save train dataset
    data_train = []
    for i in range(len(QT_Interval_train)):
        x = list(QT_Interval_train[i][0])
        x.extend(list(QT_Interval_train[i][1]))
        x.append(QT_Interval_label_train[i])
        data_train.append(x)
        
    ECG_test = pd.DataFrame(data_test)
    ECG_train = pd.DataFrame(data_train)

    # save QRS Complex 
    ECG_test.to_csv("dataset/test_all_feature_qt_interval.csv", index=False, header=False)
    ECG_train.to_csv("dataset/train_all_feature_qt_interval.csv", index=False, header=False)
    print("-------------------------- *** --------------------------\n\n")
    
def classification(cv_splits=5, 
                    feature_type = 'rr_interval', 
                    EPOCHS = 10, 
                    BATCH_SIZE = 32, 
                    threshold_acc = 0.9, 
                    fs = 250, 
                    sample_size=6, 
                    feature_pad = 15, 
                    labels = ['AF', 'N']):   

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.utils import class_weight
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import StratifiedKFold
    
    import json
    import itertools
    
    def writeJson_config(Path, Name, Data, append):
        mode = 'a+' if append else 'w'
        full_path = Path + Name

        with open(full_path, mode=mode) as json_config:
            json.dump(Data, json.load(json_config) if append else json_config)
        
        return 'success' 

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, 
                              fold_var=0):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(5, 5))
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(experiment_folder + experiment_name +
                   "/plot-confusion-matrix-%s-cv%d.png" % (feature_type, fold_var))
        
        plt.show()
            
    def evaluate_model(history, fold_var=0):
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model - Accuracy')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()
        fig1.savefig(experiment_folder + experiment_name +
                   "/plot-accuracy-%s-cv%d.png" % (feature_type, fold_var))
        
        fig2, ax_loss = plt.subplots()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model- Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.grid()
        plt.show()
        fig2.savefig(experiment_folder + experiment_name +
                   "/plot-loss-%s-cv%d.png" % (feature_type, fold_var))
               
    print("[INFO] create experiment folder...")
    experiment_folder = "experiment/"
    experiment_name = feature_type + "_" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    if not os.path.exists(experiment_folder + experiment_name) :
        os.mkdir(experiment_folder + experiment_name)
     
    print("[INFO] loading feature %s dataset" % feature_type)     
    dataset_folder = 'dataset/'
    test_df = []
    train_df = []

    if feature_type == 'rr_interval' :
        test_df = pd.read_csv(dataset_folder + "test_all_feature_rr_interval.csv", header=None)
        train_df = pd.read_csv(dataset_folder + "train_all_feature_rr_interval.csv", header=None)
    if feature_type == 'qrs_complex' :
        test_df = pd.read_csv(dataset_folder + "test_all_feature_qrs_complex.csv", header=None)
        train_df = pd.read_csv(dataset_folder + "train_all_feature_qrs_complex.csv", header=None)
    if feature_type == 'qt_interval' :
        test_df = pd.read_csv(dataset_folder + "test_all_feature_qt_interval.csv", header=None)
        train_df = pd.read_csv(dataset_folder + "train_all_feature_qt_interval.csv", header=None)

    featurs_dfs = pd.concat([test_df, train_df])
    X = featurs_dfs.values[:, :feature_pad*2].reshape(-1, feature_pad*2, 1)
    y = featurs_dfs.values[:, feature_pad*2]
    
    print("[INFO] encoding label to categorical...")
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))
    print("Categories :", enc.categories_[0])
    y_categorical = enc.transform(y.reshape(-1, 1)).toarray()

    print("\n\n")
    print("[INFO] ---------- Classification CNN ---------------")     
    print("[INFO] build model ...")
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
    from keras.layers import Input
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    import keras
    
    def cnn_model(max_len):
    
        model = Sequential()

        model.add(Conv1D(filters=64,
                         kernel_size=5,
                         activation='relu',
                         input_shape=(max_len, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=2,
                            strides=2,
                            padding='same'))

        model.add(Conv1D(filters=64,
                         kernel_size=3,
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=2,
                            strides=2,
                            padding='same'))

        # model.add(Conv1D(filters=64,
                         # kernel_size=3,
                         # activation='relu'))
        # model.add(BatchNormalization())
        # model.add(MaxPool1D(pool_size=2,
                            # strides=2,
                            # padding='same'))

        # Fully Connected layer (FC)
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(128, 
                        activation='relu'))
        model.add(Dense(32, 
                        activation='relu'))
        model.add(Dense(2, 
                        activation='softmax'))

        model.summary()
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])

        return model
    
    def train_model(model_, x, y, x_val, y_val, epochs_, batch_size_):
        callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                         ModelCheckpoint(filepath='best_model.h5', 
                                         monitor='val_loss', save_best_only=True, mode='min')]

        hist = model_.fit(x, 
                          y,
                          epochs=epochs_,
                          callbacks=callbacks, 
                          batch_size=batch_size_,
                          shuffle=True,
                          validation_data=(x_val,y_val))
        print("[INFO] reload model 'best_model.h5'...")
        model_.load_weights('best_model.h5')
        return hist 
    
    print("[INFO] start cross validation split with number of split : %d ..." % cv_splits)
    kf = StratifiedKFold(n_splits = cv_splits, random_state = 7, shuffle = True)
    fold_var = 1
    n_samples = len(y)

    for train_index, val_index in kf.split(np.zeros(n_samples), y):
        print("\n")
        print("[INFO] Train model... cv %d" % fold_var)
        print("\n")
        
        X_train = X[train_index] 
        X_test = X[val_index]
        y_train = y_categorical[train_index]
        y_test = y_categorical[val_index]

        X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
        X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
        
        max_len = X_train.shape[1]  
        model = cnn_model(max_len)
        
        with open(experiment_folder + experiment_name +
              '/model_summary_%s.txt' % feature_type, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    
        history = train_model(model, X_train,y_train,X_test,y_test, EPOCHS, BATCH_SIZE)
        evaluate_model(history, fold_var)
        
        shutil.copy('best_model.h5' , experiment_folder + experiment_name 
                                    + "/CNN_Classification_model_%s_cv%d.h5" % (feature_type, fold_var))
        pd.DataFrame.from_dict(history.history).to_csv(experiment_folder + experiment_name + 
                                                        '/history_train_CNN_feature_%s_cv%d.csv' % 
                                                       (feature_type, fold_var) ,index=False) 

        print("\n")
        print("[INFO] evaluate model - cv %d..." % fold_var) 
        print("\n")   
        # predict test data
        y_pred=model.predict(X_test)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        plot_confusion_matrix(cnf_matrix, classes=['AF', 'N'],normalize=True,
                      title='Confusion matrix, with normalization', fold_var=fold_var)

        # print classification recport
        report_dict = classification_report(y_test.argmax(axis=1), 
                            y_pred.argmax(axis=1), 
                            target_names=['AF', 'N'],
                            output_dict=True)
        writeJson_config(experiment_folder + experiment_name + "/", 
                 "report-%s-cv%d.json" % (feature_type, fold_var), report_dict, False)

        print("[INFO] save model spec...")
        model_spec_dict = {}
        model_spec_dict['train_size'] = [y_train.shape, X_train.shape]
        model_spec_dict['test_size'] =  [y_test.shape, X_test.shape]
        model_spec_dict['epoch'] =  EPOCHS
        model_spec_dict['batch_size'] =  BATCH_SIZE
        model_spec_dict['current_fold'] = fold_var
        model_spec_dict['cv_splits'] = cv_splits
        model_spec_dict['sample_size'] = sample_size
        model_spec_dict['feature_pad'] = feature_pad

        writeJson_config(experiment_folder + experiment_name + "/", 
                         "model-spec-%s-cv%d.json" % (feature_type, fold_var), model_spec_dict, False)
                         
        print("[INFO] update experiment header...")
        with open(experiment_folder + 
              '/experiment_header.txt', 'a') as f:
            f.write("Experiment Name \t: %s \n" % experiment_name)
            f.write("Accuracy \t\t: %.4f\n\n\n" % report_dict['accuracy'])
            
        print("[INFO] move training result to experiment folder %s ..." % (experiment_folder + experiment_name))
        if report_dict['accuracy'] >= threshold_acc:
            
            shutil.copy(experiment_folder + experiment_name + "/plot-accuracy-%s-cv%d.png" % (feature_type, fold_var), "5. plot-accuracy-%s.png" % feature_type)
            shutil.copy(experiment_folder + experiment_name + "/plot-loss-%s-cv%d.png" % (feature_type, fold_var), "5. plot-loss-%s.png" % feature_type)
            shutil.copy(experiment_folder + experiment_name + "/plot-confusion-matrix-%s-cv%d.png" % (feature_type, fold_var), "5. plot-confusion-matrix-%s.png" % feature_type)
            shutil.copy(experiment_folder + experiment_name + "/history_train_CNN_feature_%s_cv%d.csv" % (feature_type, fold_var), "history_train_CNN_feature_%s.csv" % feature_type)
            shutil.copy(experiment_folder + experiment_name + "/report-%s-cv%d.json" % (feature_type, fold_var), "classification-report-%s.json" % feature_type)
            shutil.copy(experiment_folder + experiment_name + "/CNN_Classification_model_%s_cv%d.h5" % (feature_type, fold_var), "CNN_Classification_model_%s.h5" % feature_type)
            print("[INFO] success move best result to main dir!")
        else :
            print("[INFO] accuracy %.4f, is under threshold !" % report_dict['accuracy'])
            
        # clear session             
        keras.backend.clear_session()
        fold_var += 1
    print("-------------------------- *** --------------------------\n\n")
    
if __name__ == "__main__" :
    records = {
        "04015" : [1, None, None, ';'], #8, 400
        "04043" : [1, None, None, ';'], #16, 1000
        "04048" : [1, None, None, ';'], #6, 900
        "04126" : [1, None, None, ';'],
        "04746" : [1, None, None, ';'],
        "04908" : [1, None, None, ';'],
        "04936" : [1, None, None, ';'], #4, 2000
        "05091" : [1, None, None, ';'], #1000
        "05121" : [1, None, None, ';'], #1000
        "05261" : [1, None, None, ';'], #18, 1000
        "06426" : [1, None, None, ';'], #2000
        "06453" : [1, None, None, ';'], #300
        "06995" : [1, None, None, ';'], #900
        "07162" : [1, None, None, ';'],
        "07859" : [1, None, None, ';'],
        "07879" : [1, None, None, ';'],
        "07910" : [1, None, None, ';'], #10, 320
        "08215" : [1, None, None, ';'], #400
        "08219" : [1, None, None, ';'], #5, 5000
        "08378" : [1, None, None, ';'], #220
        "08405" : [1, None, None, ';'],
        "08434" : [1, None, None, ';'],
        "08455" : [1, None, None, ';'], #90
    }
    
    print("\n\n")
    print("============================ *** ============================")
    answer = {}
    answer_type = ["Preprocessing AFDB", 
                    "Preprocessing NSRDB", 
                    "Merging Dataset", 
                    "Denoising Conv AE", 
                    "Feature Extraction", 
                    "Classification CNN"]
    for item in answer_type :
        answer[item] = input("Run %s [y/n]?" % item)
        while answer[item] not in ["y", "n"] : 
            print ("Please choose correct answer.")
            answer[item] = input("Run %s [y/n]?" % item)
    
    feature_type = ['rr_interval', 'qrs_complex', 'qt_interval']
    selected_feature_type = {}
    if answer["Classification CNN"] == "y":
        for feature in feature_type :
            selected_feature_type[feature] = input("Use feature %s for classification [y/n]?" % feature)
            while selected_feature_type[feature] not in ["y", "n"] :
                print ("Please choose correct answer.")
                selected_feature_type[feature] = input("Use feature %s for classification [y/n]?" % feature)
    
    print("============================ *** ============================")
    print("\n\n")        
        
    if answer[answer_type[0]] == "y" :
        print("============================ *** ============================")
        print("=                 PREPROCESSING DATASET AFDB                =") 
        print("============================ *** ============================")
        for record in records :
            print("[INFO] processing recod %s..." % record)
            start = records[record][0]
            stop = records[record][1]
            separator = records[record][3]
            preprocessing_AFDB(record, start=start, stop=stop, sep=separator, fs=250, sample_size=6)
     
     
    if answer[answer_type[1]] == "y" :
        print("============================ *** ============================")
        print("=                PREPROCESSING DATASET NSRDB                =") 
        print("============================ *** ============================")
        nsrdb_dir = os.listdir("dataset/NSRDB")
        for record in nsrdb_dir :
            print("[INFO] processing recod %s..." % record)
            preprocessing_NSRDB(record, fs = 128, sample_size=6)
        

    if answer[answer_type[2]] == "y" :
        print("============================ *** ============================")    
        print("=                      MERGING DATASET                      =") 
        print("============================ *** ============================") 
        merging_dataset(fs=250, sample_size=6)
    
    
    if answer[answer_type[3]] == "y" :
        print("============================ *** ============================") 
        print("=                         DENOISING                         =") 
        print("============================ *** ============================") 
        denoising(fs=250, sample_size=6)


    if answer[answer_type[4]] == "y" :
        print("============================ *** ============================") 
        print("=                    FEATURE EXTRACTION                     =") 
        print("============================ *** ============================") 
        feature_extraction(fs = 250, sample_size = 6, label_name = ['AF', 'N'])
        
    if answer[answer_type[5]] == "y" :    
        print("============================ *** ============================") 
        print("=                      CLASSIFICATION                       =") 
        print("============================ *** ============================") 
        for feature in selected_feature_type.keys() :
            if selected_feature_type[feature] == "y" :
                print("\n\n[INFO] ---------- classification for %s feature ----------" % feature)
                classification(cv_splits=5, 
                            feature_type = feature, 
                            EPOCHS = 10, 
                            BATCH_SIZE = 32, 
                            threshold_acc = 0.97, 
                            fs = 250, 
                            sample_size=6, 
                            feature_pad = 15, 
                            labels = ['AF', 'N'])