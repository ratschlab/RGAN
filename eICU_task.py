#!/usr/bin/env ipython
#
# Scripts for helping us to identify doable tasks in the eICU.
#
#
import pandas as pd
import numpy as np
import pdb
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import itertools

def get_train_data(df, n_hours, seq_length, resample_time,
        future_window_size=0,
        sao2_low=96, heartrate_low=75, respiration_low=15, systemicmean_low=75,
        heartrate_high=100, respiration_high=20, systemicmean_high=100):
    """
    seq_length is how many measurements we use for training
    """
    patients = set(df.pid)
    window_size = int(n_hours*60/resample_time)      # this is how many rows in the window
    X = np.empty(shape=(len(patients), 4*seq_length))  # we have 4*seq_length features
    Y = np.empty(shape=(len(patients), 7))        # we have 7 labels
    i = 0
    kept_patients = [-1337]*len(patients)
    for pat in patients:
        df_pat_withlabels = get_labels(df, pat, seq_length, window_size, 
                future_window_size, sao2_low, heartrate_low, respiration_low, 
                systemicmean_low, heartrate_high, respiration_high, systemicmean_high)
        if df_pat_withlabels is None:
            print('Skipping patient', pat, 'for having too little data')
            continue
        # subset to train period
        df_pat_train = df_pat_withlabels.head(seq_length)
        if df_pat_train.shape[0] < seq_length:
            print('Skipping patient', pat, 'for having too little data')
            continue
        X_pat = df_pat_train[['sao2', 'heartrate', 'respiration', 'systemicmean']].values.reshape(4*seq_length)
        if np.isnan(X_pat).any():
            # this should not happen any more btw
            print('Dropping patient', pat, 'for having NAs')
            # just ignore this row
        else:
            X[i, :] = X_pat
            Y[i, :] = df_pat_train.tail(1)[['low_sao2', 'low_heartrate', 'low_respiration', \
                    'low_systemicmean', 'high_heartrate', 'high_respiration', 'high_systemicmean']].values.reshape(7)*1
            kept_patients[i] = pat
            i += 1
    print('Kept data on', i, 'patients (started with', len(patients), ')')
    # delete the remaining rows
    X = X[:i]
    Y = Y[:i]
    return X, Y, kept_patients

def train_model(X, Y, X_test=None, Y_test=None):
    """
    simple classification task
    """
    results = []
    n_labels = Y.shape[1]
    if X_test is None and Y_test is None:
        n_samples = X.shape[0]
        n_train = int(0.8*n_samples)
        perm = np.random.permutation(n_samples)
        train_indices = perm[:n_train]
        test_indices = perm[n_train:]
        X_test = X[test_indices]
        X_train = X[train_indices]
        Y_test = Y[test_indices]
        Y_train = Y[train_indices]
    else:
        X_train = X
        Y_train = Y
    for label in range(n_labels):
        y = Y[:, label]
        model = RandomForestClassifier(n_estimators=50).fit(X_train, Y_train[:, label])
        predict = model.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(Y_test[:, label], predict)
        precision = sklearn.metrics.precision_score(Y_test[:, label], predict)
        recall = sklearn.metrics.recall_score(Y_test[:, label], predict)
        results.append([accuracy, precision, recall])
    return results

#sao2_high = 105 # does not exist
heartrate_low = 75
heartrate_high = 100
respiration_low = 15
respiration_high = 20
systemicmean_low = 65
systemicmean_high = 100

def get_df(resample_time=15, n_hours=1, seq_length=16):
    derived_dir = 'REDACTED'
    print('getting patients')
    patients = list(map(int, np.loadtxt(derived_dir + 'cohort_complete_resampled_pats_' + str(resample_time) + 'min.csv')))
    data = derived_dir + 'resampled_pats_' + str(resample_time) + 'min.csv'
    print('getting data')
    df = pd.read_csv(data)
    print('subsetting to "complete" patients')
    max_offset = 1.5*(seq_length*resample_time + 60*n_hours)        # for good measure
    print('subsetting by time')
    df = df[df.offset < max_offset]
    # drop patients missing any data in this region (this is slow but it's worth it)
    df = df.groupby('pid').filter(lambda x: np.all(np.isfinite(x.values)) and x.shape[0] > seq_length)
    return df

def run_grid(seq_length=16, future_window_size=0):
    sao2_low_vals = [95, 90, 80]
    heartrate_low_vals = [50, 60, 75]
    respiration_low_vals = [5, 10, 15]
    systemicmean_low_vals = [65, 75]
    heartrate_high_vals = [90, 100, 110]
    respiration_high_vals = [20, 30]
    systemicmean_high_vals = [110, 130, 150]
    derived_dir = 'REDACTED'
    grid_file = open('grid_out.txt', 'w')
    grid_file.write('resample_time n_hours seq_length future_window_size sao2_low heartrate_low \
            respiration_low systemicmean_low heartrate_high respiration_high \
            systemicmean_high \
            sao2_low_accuracy sao2_low_precision sao2_low_recall \
            heartrate_low_accuracy heartrate_low_precision heartrate_low_recall \
            respiration_low_accuracy respiration_low_precision respiration_low_recall \
            systemicmean_low_accuracy systemicmean_low_precision systemicmean_low_recall \
            heartrate_high_accuracy heartrate_high_precision heartrate_high_recall \
            respiration_high_accuracy respiration_high_precision respiration_high_recall \
            systemicmean_high_accuracy systemicmean_high_precision systemicmean_high_recall\n')
    for resample_time in [15, 30]:
        print('getting patients')
        patients = list(map(int, np.loadtxt(derived_dir + 'cohort_complete_resampled_pats_' + str(resample_time) + 'min.csv')))
        data = derived_dir + 'resampled_pats_' + str(resample_time) + 'min.csv'
        print('getting data')
        df = pd.read_csv(data)
        print('subsetting to "complete" patients')
        df = df[df.pid.isin(patients)]
        for n_hours in [1, 2, 3, 4, 5, 10]:
            max_offset = 1.5*(seq_length*resample_time + 60*n_hours)        # for good measure
            print('subsetting by time')
            df = df[df.offset < max_offset]
            # drop patients missing any data in this region (this is slow but it's worth it)
            print('deleting patients missing data and with too little data')
            df = df.groupby('pid').filter(lambda x: np.all(np.isfinite(x.values)) and x.shape[0] > seq_length)
            for (sao2_low, heartrate_low, respiration_low, systemicmean_low, heartrate_high, respiration_high, systemicmean_high) in itertools.product(sao2_low_vals, heartrate_low_vals, respiration_low_vals, systemicmean_low_vals, heartrate_high_vals, respiration_high_vals, systemicmean_high_vals):
        # restrict
                settings = [resample_time, n_hours, seq_length, future_window_size, sao2_low, 
                    heartrate_low, respiration_low, systemicmean_low,
                    heartrate_high, respiration_high, systemicmean_high]
                print(settings)
                grid_file.write(' '.join(map(str, settings)) + ' ')
                X, Y = get_train_data(df, n_hours, seq_length, resample_time, future_window_size,
                    sao2_low, heartrate_low, respiration_low, systemicmean_low,
                    heartrate_high, respiration_high, systemicmean_high)
                results = train_model(X, Y)
                for accuracy, precision, recall in results:
                    grid_file.write(str(accuracy) + ' ' + str(precision) + ' ' + str(recall))
                grid_file.write('\n')
                grid_file.flush()
    return True

def get_labels(df, patient, seq_length, window_size, future_window_size,
        sao2_low, heartrate_low, respiration_low, systemicmean_low,
        heartrate_high, respiration_high, systemicmean_high):
    df_pat = df[df.pid == patient]
    if df_pat.shape[0] < seq_length + window_size:
        return None
    df_pat.set_index('offset', inplace=True)
    df_pat.sort_index(inplace=True)
    df_pat_rollmins = df_pat.fillna(1337).rolling(window_size).min()
    df_pat_rollmaxs = df_pat.fillna(-1337).rolling(window_size).max()
    # get thresholds
    low_sao2 = df_pat_rollmins.sao2 < sao2_low
    low_heartrate = df_pat_rollmins.heartrate < heartrate_low
    low_respiration = df_pat_rollmins.respiration < respiration_low
    low_systemicmean = df_pat_rollmins.systemicmean < systemicmean_low
    high_heartrate = df_pat_rollmaxs.heartrate > heartrate_high
    high_respiration = df_pat_rollmaxs.respiration > respiration_high
    high_systemicmean = df_pat_rollmaxs.systemicmean > systemicmean_high
    # extremes
    df_pat_labels = pd.DataFrame({'low_sao2': low_sao2.values, 
                        'low_heartrate': low_heartrate.values,
                        'low_respiration': low_respiration.values,
                        'low_systemicmean': low_systemicmean.values,
                        'high_heartrate': high_heartrate.values,
                        'high_respiration': high_respiration.values,
                        'high_systemicmean': high_systemicmean.values})
    # now we need to align it - first move it back to 0 (subtract window_size),
    # then shift it forward by future_window_size (when we want to make 
    # predictions about)
    df_pat_labels_aligned = df_pat_labels.shift(-window_size + 1 + future_window_size)
    df_pat_labels_aligned.index = df_pat.index
    df_pat_withlabels = pd.concat([df_pat, df_pat_labels_aligned], axis=1)
    return df_pat_withlabels

def gen_data():
    """ just run the whole thing """
    df = get_df(resample_time=15, n_hours=1, seq_length=16)
    X, Y, pids = get_train_data(df, 1, 16, 15, sao2_low=95, respiration_low=13, respiration_high=25, heartrate_low=70, heartrate_high=100, systemicmean_low=70, systemicmean_high=110)
    extreme_heartrate = Y[:, 1] + Y[:, 4]
    extreme_respiration = Y[:, 2] + Y[:, 5]
    extreme_MAP = Y[:, 3] + Y[:, 6]
    Y_OR = np.vstack((extreme_heartrate, extreme_respiration, extreme_MAP)).T
    Y_OR = (Y_OR>0)*1
    pp = [p for p in pids if not p == -1337]
    X_train, X_test, Y_train, Y_test, pids_train, pids_test = sklearn.model_selection.train_test_split(X, Y, pp, test_size=0.2, stratify=Y)
    X_train, X_vali, Y_train, Y_vali, pids_train, pids_vali = sklearn.model_selection.train_test_split(X_train, Y_train, pids_train, test_size=0.25, stratify=Y_train)
    Y_columns = ['low_sao2', 'low_heartrate', 'low_respiration', 'low_systemicmean', 'high_heartrate', 'high_respiration', 'high_systemicmean']
    data = dict()
    data['X_train'] = X_train
    data['X_vali'] = X_vali
    data['X_test'] = X_test
    data['Y_train'] = Y_train
    data['Y_vali'] = Y_vali
    data['Y_test'] = Y_test
    data['pids_train'] = pids_train
    data['pids_vali'] = pids_vali
    data['pids_test'] = pids_test
    data['Y_columns'] = Y_columns
    data['Y_ORs'] = Y_OR
    np.save('eICU_task_data2.npy', data)
