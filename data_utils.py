import numpy as np
import pandas as pd
import pdb
import re
from time import time
import json
import random
import os

import model
import paths

from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, invgamma, mode
from scipy.special import gamma
from scipy.misc import imresize
from functools import partial
from math import ceil

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

# --- to do with loading --- #
def get_samples_and_labels(settings):
    """
    Parse settings options to load or generate correct type of data,
    perform test/train split as necessary, and reform into 'samples' and 'labels'
    dictionaries.
    """
    if settings['data_load_from']:
        data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
        print('Loading data from', data_path)
        samples, pdf, labels = get_data('load', data_path)
        train, vali, test = samples['train'], samples['vali'], samples['test']
        train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
        del samples, labels
    elif settings['data'] == 'eICU_task':
        # always load eICU
        samples, pdf, labels = get_data('eICU_task', {})
#		del samples, labels
        train, vali, test = samples['train'], samples['vali'], samples['test']
        train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
        assert train_labels.shape[1] == settings['cond_dim']
        # normalise to between -1, 1
        train, vali, test = normalise_data(train, vali, test)
    else:
        # generate the data
        data_vars = ['num_samples', 'seq_length', 'num_signals', 'freq_low',
                'freq_high', 'amplitude_low', 'amplitude_high', 'scale',
                'full_mnist']
        data_settings = dict((k, settings[k]) for k in data_vars if k in settings.keys())
        samples, pdf, labels = get_data(settings['data'], data_settings)
        if 'multivariate_mnist' in settings and settings['multivariate_mnist']:
            seq_length = samples.shape[1]
            samples = samples.reshape(-1, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
        if 'normalise' in settings and settings['normalise']: # TODO this is a mess, fix
            print(settings['normalise'])
            norm = True
        else:
            norm = False
        if labels is None:
            train, vali, test = split(samples, [0.6, 0.2, 0.2], normalise=norm)
            train_labels, vali_labels, test_labels = None, None, None
        else:
            train, vali, test, labels_list = split(samples, [0.6, 0.2, 0.2], normalise=norm, labels=labels)
            train_labels, vali_labels, test_labels = labels_list

    labels = dict()
    labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

    samples = dict()
    samples['train'], samples['vali'], samples['test'] = train, vali, test

    # futz around with labels
    # TODO refactor cause this is messy
    if 'one_hot' in settings and settings['one_hot'] and not settings['data_load_from']:
        if len(labels['train'].shape) == 1:
            # ASSUME labels go from 0 to max_val inclusive, find max-val
            max_val = int(np.max([labels['train'].max(), labels['test'].max(), labels['vali'].max()]))
            # now we have max_val + 1 dimensions
            print('Setting cond_dim to', max_val + 1, 'from', settings['cond_dim'])
            settings['cond_dim'] = max_val + 1
            print('Setting max_val to 1 from', settings['max_val'])
            settings['max_val'] = 1

            labels_oh = dict()
            for (k, v) in labels.items():
                A = np.zeros(shape=(len(v), settings['cond_dim']))
                A[np.arange(len(v)), (v).astype(int)] = 1
                labels_oh[k] = A
            labels = labels_oh
        else:
            assert settings['max_val'] == 1
            # this is already one-hot!

    if 'predict_labels' in settings and settings['predict_labels']:
        samples, labels = data_utils.make_predict_labels(samples, labels)
        print('Setting cond_dim to 0 from', settings['cond_dim'])
        settings['cond_dim'] = 0

    # update the settings dictionary to update erroneous settings
    # (mostly about the sequence length etc. - it gets set by the data!)
    settings['seq_length'] = samples['train'].shape[1]
    settings['num_samples'] = samples['train'].shape[0] + samples['vali'].shape[0] + samples['test'].shape[0]
    settings['num_signals'] = samples['train'].shape[2]
    settings['num_generated_features'] = samples['train'].shape[2]

    return samples, pdf, labels

def get_data(data_type, data_options=None):
    """
    Helper/wrapper function to get the requested data.
    """
    labels = None
    pdf = None
    if data_type == 'load':
        data_dict = np.load(data_options).item()
        samples = data_dict['samples']
        pdf = data_dict['pdf']
        labels = data_dict['labels']
    elif data_type == 'sine':
        samples = sine_wave(**data_options)
    elif data_type == 'mnist':
        if data_options['full_mnist']:
            samples, labels = mnist()
        else:
            #samples, labels = load_resized_mnist_0_5(14)
            samples, labels = load_resized_mnist(14)       # this is the 0-2 setting
    elif data_type == 'gp_rbf':
        print(data_options)
        samples, pdf = GP(**data_options, kernel='rbf')
    elif data_type == 'linear':
        samples, pdf = linear(**data_options)
    elif data_type == 'eICU_task':
        samples, labels = eICU_task()
    elif data_type == 'resampled_eICU':
        samples, labels = resampled_eICU(**data_options)
    else:
        raise ValueError(data_type)
    print('Generated/loaded', len(samples), 'samples from data-type', data_type)
    return samples, pdf, labels


def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]

def normalise_data(train, vali, test, low=-1, high=1):
    """ Apply some sort of whitening procedure
    """
    # remember, data is num_samples x seq_length x signals
    # whiten each signal - mean 0, std 1
    mean = np.mean(np.vstack([train, vali]), axis=(0, 1))
    std = np.std(np.vstack([train-mean, vali-mean]), axis=(0, 1))

    normalised_train = (train - mean)/std
    normalised_vali = (vali - mean)/std
    normalised_test = (test - mean)/std
#    normalised_data = data - np.nanmean(data, axis=(0, 1))
#    normalised_data /= np.std(data, axis=(0, 1))

#    # normalise samples to be between -1 and +1
    # normalise just using train and vali
#    min_val = np.nanmin(np.vstack([train, vali]), axis=(0, 1))
#    max_val = np.nanmax(np.vstack([train, vali]), axis=(0, 1))
#
#    normalised_train = (train - min_val)/(max_val - min_val)
#    normalised_train = (high - low)*normalised_train + low
#
#    normalised_vali = (vali - min_val)/(max_val - min_val)
#    normalised_vali = (high - low)*normalised_vali + low
#
#    normalised_test = (test - min_val)/(max_val - min_val)
#    normalised_test = (high - low)*normalised_test + low
    return normalised_train, normalised_vali, normalised_test

def scale_data(train, vali, test, scale_range=(-1, 1)):
    signal_length = train.shape[1]
    num_signals = train.shape[2]
    # reshape everything
    train_r = train.reshape(-1, signal_length*num_signals)
    vali_r = vali.reshape(-1, signal_length*num_signals)
    test_r = test.reshape(-1, signal_length*num_signals)
    # fit scaler using train, vali
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([train_r, vali_r]))
    # scale everything
    scaled_train = scaler.transform(train_r).reshape(-1, signal_length, num_signals)
    scaled_vali = scaler.transform(vali_r).reshape(-1, signal_length, num_signals)
    scaled_test = scaler.transform(test_r).reshape(-1, signal_length, num_signals)
    return scaled_train, scaled_vali, scaled_test

def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Return train/validation/test split.
    """
    if random_seed != None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total*proportions[0])
    n_test = ceil(n_total*proportions[2])
    n_vali = n_total - (n_train + n_test)
    # permutation to shuffle the samples
    shuff = np.random.permutation(n_total)
    train_indices = shuff[:n_train]
    vali_indices = shuff[n_train:(n_train + n_vali)]
    test_indices = shuff[(n_train + n_vali):]
    # TODO when we want to scale we can just return the indices
    assert len(set(train_indices).intersection(vali_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(vali_indices).intersection(test_indices)) == 0
    # split up the samples
    train = samples[train_indices]
    vali = samples[vali_indices]
    test = samples[test_indices]
    # apply the same normalisation scheme to all parts of the split
    if normalise:
        if scale: raise ValueError(normalise, scale)        # mutually exclusive
        train, vali, test = normalise_data(train, vali, test)
    elif scale:
        train, vali, test = scale_data(train, vali, test)
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split


def make_predict_labels(samples, labels):
    """ Given two dictionaries of samples, labels (already normalised, split etc)
        append the labels on as additional signals in the data
    """
    print('Appending label to samples')
    assert not labels is None
    if len(labels['train'].shape) > 1:
        num_labels = labels['train'].shape[1]
    else:
        num_labels = 1
    seq_length = samples['train'].shape[1]
    num_signals = samples['train'].shape[2]
    new_samples = dict()
    new_labels = dict()
    for (k, X) in samples.items():
        num_samples = X.shape[0]
        lab = labels[k]
        # slow code because i am sick and don't want to try to be smart
        new_X = np.zeros(shape=(num_samples, seq_length, num_signals + num_labels))
        for row in range(num_samples):
            new_X[row, :, :] = np.hstack([X[row, :, :], np.array(seq_length*[(2*lab[row]-1).reshape(num_labels)])])
        new_samples[k] = new_X
        new_labels[k] = None
    return new_samples, new_labels

# --- specific data-types --- #

def eICU_task(predict_label=False):
    """
    Load the eICU data for the extreme-value prediction task
    """
    path = 'REDACTED'
    data = np.load(path).item()
    # convert it into similar format
    labels = {'train': data['Y_train'], 'vali': data['Y_vali'], 'test': data['Y_test']}
    samples = {'train': data['X_train'], 'vali': data['X_vali'], 'test': data['X_test']}
    # reshape
    for (k, X) in samples.items():
        samples[k] = X.reshape(-1, 16, 4)
    return samples, labels

def mnist(randomize=False):
    """ Load and serialise """
    try:
        train = np.load('./data/mnist_train.npy')
        print('Loaded mnist from .npy')
    except IOError:
        print('Failed to load MNIST data from .npy, loading from csv')
        # read from the csv
        train = np.loadtxt(open('./data/mnist_train.csv', 'r'), delimiter=',')
        # scale samples from 0 to 1
        train[:, 1:] /= 255
        # scale from -1 to 1
        train[:, 1:] = 2*train[:, 1:] - 1
        # save to the npy
        np.save('./data/mnist_train.npy', train)
    # the first column is labels, kill them
    labels = train[:, 0]
    samples = train[:, 1:]
    if randomize:
        # not needed for GAN experiments...
        print('Applying fixed permutation to mnist digits.')
        fixed_permutation = np.random.permutation(28*28)
        samples = train[:, fixed_permutation]
    samples = samples.reshape(-1, 28*28, 1)     # add redundant additional signals
    return samples, labels


def load_resized_mnist_0_5(new_size, randomize=False):
    """ Load resised mnist digits from 0 to 5 """

    samples, labels = mnist()
    print('Resizing...')
    samples = samples[np.in1d(labels,[0,1,2,3,4,5])]
    labels = labels[np.in1d(labels,[0,1,2,3,4,5])]
    if new_size != 28:
        resized_imgs = [imresize(img.reshape([28,28]), [new_size,new_size], interp='lanczos').ravel()[np.newaxis].T
                        for img in samples]
        resized_imgs = np.array(resized_imgs)
        resized_imgs = resized_imgs.astype(float)
        resized_imgs /= 255.0
        resized_imgs = 2*resized_imgs - 1
        np.save('./data/resized_mnist_1_5_samples.npy', resized_imgs)
        np.save('./data/resized_mnist_1_5_labels.npy', labels)
        return resized_imgs, labels
    else:
        return samples, labels


def load_resized_mnist(new_size, from_to_digits=(0,2), randomize=False):
    """ Load resised mnist digits from 0 to 5 """

    samples, labels = mnist()
    print('Resizing...')
    samples = samples[np.in1d(labels,np.arange(from_to_digits[0], from_to_digits[1]+1))]
    labels = labels[np.in1d(labels,np.arange(from_to_digits[0], from_to_digits[1]+1))]
    if new_size != 28:
        resized_imgs = [imresize(img.reshape([28,28]), [new_size,new_size], interp='lanczos').ravel()[np.newaxis].T
                        for img in samples]
        resized_imgs = np.array(resized_imgs)
        resized_imgs = resized_imgs.astype(float)
        resized_imgs /= 255.0
        resized_imgs = 2*resized_imgs - 1
        np.save('./data/resized_mnist_'+ str(from_to_digits[0]) + '_' + str(from_to_digits[1]) + '_5_samples.npy', resized_imgs)
        np.save('./data/resized_mnist_'+ str(from_to_digits[0]) + '_' + str(from_to_digits[1]) + '_labels.npy', labels)
        return resized_imgs, labels
    else:
        return samples, labels

def resampled_eICU(seq_length=16, resample_rate_in_min=15,
        variables=['sao2', 'heartrate', 'respiration', 'systemicmean'], **kwargs):
    """
    Note: resampling rate is 15 minutes
    """
    print('Getting resampled eICU data')
    try:
        data = np.load(paths.eICU_proc_dir + 'eICU_' + str(resample_rate_in_min) + '.npy').item()
        samples = data['samples']
        pids = data['pids']
        print('Loaded from file!')
        return samples, pids
    except FileNotFoundError:
        # in this case, we go into the main logic of the function
        pass
    resampled_data_path = paths.eICU_proc_dir + 'complete_resampled_pats_' + str(resample_rate_in_min) + 'min.csv'
    resampled_pids_path = paths.eICU_proc_dir + 'cohort_complete_resampled_pats_' + str(resample_rate_in_min) + 'min.csv'

    if not os.path.isfile(resampled_data_path):
        generate_eICU_resampled_patients(resample_factor_in_min=resample_rate_in_min, upto_in_minutes=None)
        get_cohort_of_complete_downsampled_patients(time_in_hours=1.5*resample_rate_in_min*seq_length, resample_factor_in_min=resample_rate_in_min)

    pids = set(np.loadtxt(resampled_pids_path, dtype=int))
    df = pd.read_csv(resampled_data_path)
    # restrict to variables
    df_restricted = df.loc[:, variables + ['offset', 'pid']]
    # restrict to patients in the "good list"
    df_restricted = df_restricted.where(df_restricted.pid.isin(pids)).dropna()
    # assert no negative offsets
    assert np.all(df_restricted.offset >= 0)
    # restrict to 1.5 time the region length
#    df_restricted = df_restricted.loc[np.all([df_restricted.offset <= 1.5*resample_rate_in_min*seq_length, df_restricted.offset >= 0], axis=0), :]
    df_restricted = df_restricted.loc[df_restricted.offset <= 1.5*resample_rate_in_min*seq_length, :]
    # for each patient, return the first seq_length observations
    patient_starts = df_restricted.groupby('pid').head(seq_length)
    n_pats_prefilter = len(set(patient_starts.pid))
    # filter out patients who have fewer than seq_length observations
    patient_starts = patient_starts.groupby('pid').filter(lambda x: x.pid.count() == seq_length)
    n_pats_postfilter = len(set(patient_starts.pid))
    print('Removed', n_pats_prefilter - n_pats_postfilter, 'patients with <', seq_length, 'observations in the first', 1.5*resample_rate_in_min*seq_length, 'minutes, leaving', n_pats_postfilter, 'patients remaining.')
    # convert to samples - shape is [n_pats, seq_length, num_signals]
    n_patients = n_pats_postfilter
    num_signals = len(variables)
    samples = np.empty(shape=(n_patients, seq_length, num_signals))
    pats_grouped = patient_starts.groupby('pid')
    pids = []
    for (i, patient) in enumerate(pats_grouped.groups):
        samples[i, :, :] = pats_grouped.get_group(patient).loc[:, variables].values
        pids.append(patient)
    assert i == n_patients - 1
    assert np.mean(np.isnan(samples) == 0)
    np.save(paths.eICU_proc_dir + 'eICU_' + str(resample_rate_in_min) + '.npy', {'samples': samples, 'pids': pids})
    print('Saved to file!')
    return samples, pids

def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples

def periodic_kernel(T, f=1.45/30, gamma=7.0, A=0.1):
    """
    Calculates periodic kernel between all pairs of time points (there
    should be seq_length of those), returns the Gram matrix.
    f is frequency - higher means more peaks
    gamma is a scale, smaller makes the covariance peaks shallower (smoother)

    Heuristic for non-singular rbf:
        periodic_kernel(np.arange(len), f=1.0/(0.79*len), A=1.0, gamma=len/4.0)
    """
    dists = squareform(pdist(T.reshape(-1, 1)))
    cov = A*np.exp(-gamma*(np.sin(2*np.pi*dists*f)**2))
    return cov


def GP(seq_length=30, num_samples=28*5*100, num_signals=1, scale=0.1, kernel='rbf', **kwargs):
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.empty(shape=(num_samples, seq_length, num_signals))
    #T = np.arange(seq_length)/seq_length    # note, between 0 and 1
    T = np.arange(seq_length)    # note, not between 0 and 1
    if kernel == 'periodic':
        cov = periodic_kernel(T)
    elif kernel =='rbf':
        cov = rbf_kernel(T.reshape(-1, 1), gamma=scale)
    else:
        raise NotImplementedError
    # scale the covariance
    cov *= 0.2
    # define the distribution
    mu = np.zeros(seq_length)
    print(np.linalg.det(cov))
    distribution = multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
    pdf = distribution.logpdf
    # now generate samples
    for i in range(num_signals):
        samples[:, :, i] = distribution.rvs(size=num_samples)
    return samples, pdf

def linear_marginal_likelihood(Y, X, a0, b0, mu0, lambda0, log=True, **kwargs):
    """
    Marginal likelihood for linear model.
    See https://en.wikipedia.org/wiki/Bayesian_linear_regression pretty much
    """
    seq_length = Y.shape[1]         # note, y is just a line (one channel) TODO
    n = seq_length
    an = a0 + 0.5*n
    XtX = np.dot(X.T, X)
    lambdan = XtX + lambda0
    prefactor = (2*np.pi)**(-0.5*n)
    dets = np.sqrt(np.linalg.det(lambda0)/np.linalg.det(lambdan))
    marginals = np.empty(Y.shape[0])
    for (i, y) in enumerate(Y):
        y_reshaped = y.reshape(seq_length)
        betahat = np.dot(np.linalg.inv(XtX), np.dot(X.T, y_reshaped))
        mun = np.dot(np.linalg.inv(lambdan), np.dot(XtX, betahat) + np.dot(lambda0, mu0))
        bn = b0 + 0.5*(np.dot(y_reshaped.T, y_reshaped) + np.dot(np.dot(mu0.T, lambda0), mu0) - np.dot(np.dot(mun.T, lambdan), mun))
        bs = (b0**a0)/(bn**an)
        gammas = gamma(an)/gamma(a0)
        marginals[i] = prefactor*dets*bs*gammas
    if log:
        marginals = np.log(marginals)
    return marginals

def linear(seq_length=30, num_samples=28*5*100, a0=10, b0=0.01, k=2, **kwargs):
    """
    Generate data from linear trend from probabilistic model.

    The invgamma function in scipy corresponds to wiki defn. of inverse gamma:
        scipy a = wiki alpha = a0
        scipy scale = wiki beta = b0

    k is the number of regression coefficients (just 2 here, slope and intercept)
    """
    T = np.zeros(shape=(seq_length, 2))
    T[:, 0] = np.arange(seq_length)
    T[:, 1] = 1                         # equivalent to X
    lambda0 = 0.01*np.eye(k)     # diagonal covariance for beta
    y = np.zeros(shape=(num_samples, seq_length, 1))
    sigmasq = invgamma.rvs(a=a0, scale=b0, size=num_samples)
    increasing = np.random.choice([-1, 1], num_samples)     # flip slope
    for n in range(num_samples):
        sigmasq_n = sigmasq[n]
        offset = np.random.uniform(low=-0.5, high=0.5)     # todo limits
        mu0 = np.array([increasing[n]*(1.0-offset)/seq_length, offset])
        beta = multivariate_normal.rvs(mean=mu0, cov=sigmasq_n*lambda0)
        epsilon = np.random.normal(loc=0, scale=np.sqrt(sigmasq_n), size=seq_length)
        y[n, :, :] = (np.dot(T, beta) + epsilon).reshape(seq_length, 1)
    marginal = partial(linear_marginal_likelihood, X=T, a0=a0, b0=b0, mu0=mu0, lambda0=lambda0)
    samples = y
    pdf = marginal
    return samples, pdf

def changepoint_pdf(Y, cov_ms, cov_Ms):
    """
    """
    seq_length = Y.shape[0]
    logpdf = []
    for (i, m) in enumerate(range(int(seq_length/2), seq_length-1)):
        Y_m = Y[:m, 0]
        Y_M = Y[m:, 0]
        M = seq_length - m
        # generate mean function for second part
        Ymin = np.min(Y_m)
        initial_val = Y_m[-1]
        if Ymin > 1:
            final_val = (1.0 - M/seq_length)*Ymin
        else:
            final_val = (1.0 + M/seq_length)*Ymin
        mu_M = np.linspace(initial_val, final_val, M)
        # ah yeah
        logpY_m = multivariate_normal.logpdf(Y_m, mean=np.zeros(m), cov=cov_ms[i])
        logpY_M = multivariate_normal.logpdf(Y_M, mean=mu_M, cov=cov_Ms[i])
        logpdf_m = logpY_m + logpY_M
        logpdf.append(logpdf_m)
    return logpdf

def changepoint_cristobal(seq_length=30, num_samples=28*5*100):
    """
    Porting Cristobal's code for generating data with a changepoint.
    """
    raise NotImplementedError

    basal_values_signal_a = np.random.randn(n_samples) * 0.33
    trends_seed_a = np.random.randn(n_samples) * 0.005
    trends = np.array([i*trends_seed_a for i in range(51)[1:]]).T
    signal_a = (basal_values_signal_a + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_a = time_noise + signal_a

    basal_values_signal_b = np.random.randn(n_samples) * 0.33
    trends_seed_b = np.random.randn(n_samples) * 0.005
    trends = np.array([i*trends_seed_b for i in range(51)[1:]]).T
    signal_b = (basal_values_signal_b + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_b = time_noise + signal_b

    signal_a = np.clip(signal_a, -1, 1)
    signal_b = np.clip(signal_b, -1, 1)

    # the change in the trend is based on the top extreme values of each
    # signal in the first half
    time_steps_until_change = np.max(np.abs(signal_a), axis=1) + np.max(np.abs(signal_b), axis=1)*100
    # noise added to the starting point
    time_steps_until_change += np.random.randn(n_samples) * 5
    time_steps_until_change = np.round(time_steps_until_change)
    time_steps_until_change = np.clip(time_steps_until_change, 0, n_steps-1)
    time_steps_until_change = n_steps - 1 - time_steps_until_change

    trends = np.array([i*trends_seed_a for i in range(101)[51:]]).T
    signal_a_target = (basal_values_signal_a + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_a_target = time_noise + signal_a_target

    trends = np.array([i*trends_seed_b for i in range(101)[51:]]).T
    signal_b_target = (basal_values_signal_b + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_b_target = time_noise + signal_b_target

    signal_multipliers = []
    for ts in time_steps_until_change:
        signal_multiplier = []
        if ts > 0:
            for i in range(int(ts)):
                signal_multiplier.append(1)
            i += 1
        else:
            i = 0
        multiplier = 1.25
        while(i<n_steps):
            signal_multiplier.append(multiplier)
            multiplier += 0.25
            i+=1
        signal_multipliers.append(signal_multiplier)
    signal_multipliers = np.array(signal_multipliers)

    for s_idx, signal_choice in enumerate(basal_values_signal_b > basal_values_signal_a):
        if signal_choice == False:
            signal_a_target[s_idx] *= signal_multipliers[s_idx]
        else:
            signal_b_target[s_idx] *= signal_multipliers[s_idx]

    signal_a_target = np.clip(signal_a_target, -1, 1)
    signal_b_target = np.clip(signal_b_target, -1, 1)

    # merging signals
    signal_a = np.swapaxes(signal_a[np.newaxis].T, 0, 1)
    signal_b = np.swapaxes(signal_b[np.newaxis].T, 0, 1)
    signal_a_target = np.swapaxes(signal_a_target[np.newaxis].T, 0, 1)
    signal_b_target = np.swapaxes(signal_b_target[np.newaxis].T, 0, 1)
    input_seqs = np.dstack((signal_a,signal_b))
    target_seqs = np.dstack((signal_a_target,signal_b_target))
    return False

def changepoint(seq_length=30, num_samples=28*5*100):
    """
    Generate data from two GPs, roughly speaking.
    The first part (up to m) is as a normal GP.
    The second part (m to end) has a linear downwards trend conditioned on the
    first part.
    """
    print('Generating samples from changepoint...')
    T = np.arange(seq_length)
    # sample breakpoint from latter half of sequence
    m_s = np.random.choice(np.arange(int(seq_length/2), seq_length-1), size=num_samples)
    samples = np.zeros(shape=(num_samples, seq_length, 1))
    # kernel parameters and stuff
    gamma=5.0/seq_length
    A = 0.01
    sigmasq = 0.8*A
    lamb = 0.0  # if non-zero, cov_M risks not being positive semidefinite...
    kernel = partial(rbf_kernel, gamma=gamma)
    # multiple values per m
    N_ms = []
    cov_ms = []
    cov_Ms = []
    pdfs = []
    for m in range(int(seq_length/2), seq_length-1):
        # first part
        M = seq_length - m
        T_m = T[:m].reshape(m, 1)
        cov_m = A*kernel(T_m.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_ms.append(cov_m)
        # the second part
        T_M = T[m:].reshape(M, 1)
        cov_mM = kernel(T_M.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_M = sigmasq*(np.eye(M) - lamb*np.dot(np.dot(cov_mM, np.linalg.inv(cov_m)), cov_mM.T))
        cov_Ms.append(cov_M)
    for n in range(num_samples):
        m = m_s[n]
        M = seq_length-m
        # sample the first m
        cov_m = cov_ms[m - int(seq_length/2)]
        Xm = multivariate_normal.rvs(cov=cov_m)
        # generate mean function for second
        Xmin = np.min(Xm)
        initial_val = Xm[-1]
        if Xmin > 1:
            final_val = (1.0 - M/seq_length)*Xmin
        else:
            final_val = (1.0 + M/seq_length)*Xmin
        mu_M = np.linspace(initial_val, final_val, M)
        # sample the rest
        cov_M = cov_Ms[m -int(seq_length/2)]
        XM = multivariate_normal.rvs(mean=mu_M, cov=cov_M)
        # combine the sequence
        # NOTE: just one dimension
        samples[n, :, 0] = np.concatenate([Xm, XM])
    pdf = partial(changepoint_pdf, cov_ms=cov_ms, cov_Ms=cov_Ms)
    return samples, pdf, m_s

def resample_eICU_patient(pid, resample_factor_in_min, variables, upto_in_minutes):
    """
    Resample a *single* patient.
    """
    pat_df = pd.read_hdf(paths.eICU_hdf_dir + '/vitalPeriodic.h5',
                         where='patientunitstayid = ' + str(pid),
                         columns=['observationoffset', 'patientunitstayid'] + variables,
                         mode='r')
    # sometimes it's empty
    if pat_df.empty:
        return None
    if not upto_in_minutes is None:
        pat_df = pat_df.loc[0:upto_in_minutes*60]
    # convert the offset to a TimedeltaIndex (necessary for resampling)
    pat_df.observationoffset = pd.TimedeltaIndex(pat_df.observationoffset, unit='m')
    pat_df.set_index('observationoffset', inplace=True)
    pat_df.sort_index(inplace=True)
    # resample by time
    pat_df_resampled = pat_df.resample(str(resample_factor_in_min) + 'T').median()  # pandas ignores NA in median by default
    # rename pid, cast to int
    pat_df_resampled.rename(columns={'patientunitstayid': 'pid'}, inplace=True)
    pat_df_resampled['pid'] = np.int32(pat_df_resampled['pid'])
    # get offsets in minutes from index
    pat_df_resampled['offset'] = np.int32(pat_df_resampled.index.total_seconds()/60)
    return pat_df_resampled

def generate_eICU_resampled_patients(resample_factor_in_min=15,
        upto_in_minutes=None):
    """
    Generates a dataframe with resampled patients. One sample every "resample_factor_in_min" minutes.
    """
    pids = set(np.loadtxt(paths.eICU_proc_dir + 'pids.txt', dtype=int))
    exclude_pids = set(np.loadtxt(paths.eICU_proc_dir + 'pids_missing_vitals.txt', dtype=int))
    print('Excluding', len(exclude_pids), 'patients for not having vitals information')
    pids = pids.difference(exclude_pids)

    variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']

    num_pat = 0
    num_miss = 0
    f_miss = open(paths.eICU_proc_dir + 'pids_missing_vitals.txt', 'a')
    for pid in pids:    # have to go patient by patient
        pat_df_resampled = resample_eICU_patient(pid, resample_factor_in_min, variables, upto_in_minutes)
        if pat_df_resampled is None:
            f_miss.write(str(pid) + '\n')
            num_miss += 1
            continue
        else:
            if num_pat == 0:
                f = open(paths.eICU_proc_dir + 'resampled_pats' + str(resample_factor_in_min) +'min.csv', 'w')
                pat_df_resampled.to_csv(f, header=True, index=False)
            else:
                pat_df_resampled.to_csv(f, header=False, index=False)
            num_pat += 1
        if num_pat % 100 == 0:
            print(num_pat)
            f.flush()
            f_miss.flush()

    print('Acquired data on', num_pat, 'patients.')
    print('Skipped', num_miss, 'patients.')
    return True

def get_cohort_of_complete_downsampled_patients(time_in_hours=4, resample_factor_in_min=15):
    """
    Finds the set of patients that have no missing data during the first "time_in_hours".

    """

    resampled_pats = pd.read_csv(paths.eICU_proc_dir + 'resampled_pats' + str(resample_factor_in_min) + 'min.csv')

    time_in_minutes = time_in_hours * 60

    # delete patients with any negative offset
    print('Deleting patients with negative offsets...')
    df_posoffset = resampled_pats.groupby('pid').filter(lambda x: np.all(x.offset >= 0))

    # restrict time consideration
    print('Restricting to offsets below', time_in_minutes)
    df = df_posoffset.loc[df_posoffset.offset <= time_in_minutes]

    #variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']
    variables = ['sao2', 'heartrate', 'respiration']

    # patients with no missing values in those variables (this is slow)
    print('Finding patients with no missing values in', ','.join(variables))
    good_patients = df.groupby('pid').filter(lambda x: np.all(x.loc[:, variables].isnull().sum() == 0))

    # extract the pids, save the cohort
    cohort = good_patients.pid.drop_duplicates()

    if cohort.shape[0] < 2:
        print('ERROR: not enough patients in cohort.', cohort.shape[0])
        return False
    else:
        print('Saving...')
        cohort.to_csv(paths.eICU_proc_dir + 'cohort_complete_resampled_pats_' + str(resample_factor_in_min) + 'min.csv', header=False, index=False)
        # save the full data (not just cohort)
        good_patients.to_csv(paths.eICU_proc_dir + 'complete_resampled_pats_' + str(resample_factor_in_min) + 'min.csv', index=False)
        return True

def get_eICU_with_targets(use_age=False, use_gender=False, save=False):
    """
    Load resampled eICU data and get static prediction targets from demographics
    (patients) file
    """
    if use_age: print('Using age!')
    if use_gender: print('Using gender!')
    if save: print('Save!')
    # load resampled eICU data (the labels are the patientunitstayids)
    samples, pdf, labels = get_data('resampled_eICU', {})

    # load patients static information
    eICU_dir = 'REDACTED'
    pat_dfs = pd.read_hdf(eICU_dir + '/patient.h5', mode='r')

    # keep only static information of patients that are in the resampled table
    pat_dfs = pat_dfs[pat_dfs.patientunitstayid.isin(labels)]

    # reordering df to have the same order as samples and labels
    pat_dfs.set_index('patientunitstayid', inplace=True)
    pat_dfs.reindex(labels)

    # target variables to keep. For now we don't use hospitaldischargeoffset since it is the only integer variable.
    #target_vars = ['hospitaldischargeoffset', 'hospitaldischargestatus', 'apacheadmissiondx', 'hospitaldischargelocation', 'unittype', 'unitadmitsource']
    real_vars = ['age']
    binary_vars = ['hospitaldischargestatus', 'gender']
    categorical_vars = ['apacheadmissiondx', 'hospitaldischargelocation', 'unittype', 'unitadmitsource']

    target_vars = categorical_vars + ['hospitaldischargestatus']
    if use_age: target_vars += ['age']
    if use_gender: target_vars += ['gender']

    targets_df = pat_dfs.loc[:, target_vars]

    # remove patients by criteria
            # missing data in any target
    targets_df.dropna(how='any', inplace=True)
    if use_age:
                # age belonw 18 or above 89
        targets_df = targets_df[targets_df.age != '> 89']       # yes, some ages are strings
        targets_df.age = list(map(int, targets_df.age))
        targets_df = targets_df[targets_df.age >= 18]
    if use_gender:
                # remove non-binary genders (sorry!)
        targets_df['gender'] = targets_df['gender'].replace(['Female', 'Male', 'Other', 'Unknown'], [0, 1, -1, -1])
        targets_df = targets_df[targets_df.gender >= 0]
    # record patients to keep
    keep_indices = [i for (i, pid) in enumerate(labels) if pid in targets_df.index]
    assert len(keep_indices) == targets_df.shape[0]
    new_samples = samples[keep_indices]
    new_labels = np.array(labels)[keep_indices]

    # triple check the labels are correct
    assert np.array_equal(targets_df.index, new_labels)

    # getn non-one-hot targets (strings)
    targets = targets_df.values

    # one hot encoding of categorical variables
    dummies = pd.get_dummies(targets_df[categorical_vars], dummy_na=True)
    targets_df_oh = pd.DataFrame()
    targets_df_oh[dummies.columns] = dummies
    # convert binary variables to one-hot, too
    targets_df_oh['hospitaldischargestatus']= targets_df['hospitaldischargestatus'].replace(['Alive', 'Expired'],[1, 0])
    if use_gender:
        targets_df_oh['gender'] = targets_df['gender']  # already binarised
    if use_age:
        targets_df_oh['age'] = 2*targets_df['age']/89 - 1     # 89 is max

    # drop dummy columns marking missing data (they should be empty)
    nancols = [col for col in targets_df_oh.columns if col.endswith('nan')]
    assert np.all(targets_df_oh[nancols].sum() == 0)
    targets_df_oh.drop(nancols, axis=1, inplace=True)
    targets_oh = targets_df_oh.values

    if save:
        # save!
        # merge with training data, for LR saving
        assert new_samples.shape[0] == targets_df_oh.shape[0]
        flat_samples = new_samples.reshape(new_samples.shape[0], -1)
        features_df = pd.DataFrame(flat_samples)
        features_df.index = targets_df_oh.index
        features_df.columns = ['feature_' + str(i) for i in range(features_df.shape[1])]
        all_data = pd.concat([targets_df_oh, features_df], axis=1)
        all_data.to_csv('./data/eICU_with_targets.csv')

    # do the split
    proportions = [0.6, 0.2, 0.2]
    labels = {'targets': targets, 'targets_oh': targets_oh}
    train_seqs, vali_seqs, test_seqs, labels_split = split(new_samples, proportions, scale=True, labels=labels)
    train_targets, vali_targets, test_targets = labels_split['targets']
    train_targets_oh, vali_targets_oh, test_targets_oh = labels_split['targets_oh']

    return train_seqs, vali_seqs, test_seqs, train_targets, vali_targets, test_targets, train_targets_oh, vali_targets_oh, test_targets_oh


### --- TSTR ---- ####
def generate_synthetic(identifier, epoch, n_train, predict_labels=False):
    """
    - Load a CGAN pretrained model
    - Load its corresponding test data (+ labels)
    - Generate num_examples synthetic training data (+labels)
    - Save to format easy for training classifier on (see Eval)
    """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    if not settings['cond_dim'] > 0:
        assert settings['predict_labels']
        assert predict_labels
    # get the test data
    print('Loading test (real) data for', identifier)
    data_dict = np.load('./experiments/data/' + identifier + '.data.npy').item()
    test_data = data_dict['samples']['test']
    test_labels = data_dict['labels']['test']
    train_data = data_dict['samples']['train']
    train_labels = data_dict['labels']['train']
    print('Loaded', test_data.shape[0], 'test examples')
    print('Sampling', n_train, 'train examples from the model')
    if not predict_labels:
        assert test_data.shape[0] == test_labels.shape[0]
        if 'eICU' in settings['data']:
            synth_labels = train_labels[np.random.choice(train_labels.shape[0], n_train), :]
        else:
            # this doesn't really work for eICU...
            synth_labels = model.sample_C(n_train, settings['cond_dim'], settings['max_val'], settings['one_hot'])
            synth_data = model.sample_trained_model(settings, epoch, n_train, Z_samples=None, cond_dim=settings['cond_dim'], C_samples=synth_labels)
    else:
        assert settings['predict_labels']
        synth_data = model.sample_trained_model(settings, epoch, n_train, Z_samples=None, cond_dim=0)
        # extract the labels
        if 'eICU' in settings['data']:
            n_labels = 7
            synth_labels = synth_data[:, :, -n_labels:]
            train_labels = train_data[:, :, -n_labels:]
            test_labels = test_data[:, :, -n_labels:]
        else:
            n_labels = 6        # mnist
            synth_labels, _ = mode(np.argmax(synth_data[:, :, -n_labels:], axis=2), axis=1)
            train_labels, _ = mode(np.argmax(train_data[:, :, -n_labels:], axis=2), axis=1)
            test_labels, _ = mode(np.argmax(test_data[:, :, -n_labels:], axis=2), axis=1)
        synth_data = synth_data[:, :, :-n_labels]
        train_data = train_data[:, :, :-n_labels]
        test_data = test_data[:, :, :-n_labels]
    # package up, save
    exp_data = dict()
    exp_data['test_data'] = test_data
    exp_data['test_labels'] = test_labels
    exp_data['train_data'] = train_data
    exp_data['train_labels'] = train_labels
    exp_data['synth_data'] = synth_data
    exp_data['synth_labels'] = synth_labels
    # save it all up
    np.save('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy', exp_data)
    return True
