#!/usr/bin/env ipython
# Evaluation of models
#

import json
import pdb
import numpy as np
import pandas as pd
from eugenium_mmd import MMD_3_Sample_Test
from scipy.stats import ks_2samp
import mmd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model
import data_utils
import plotting

import pickle

def assert_same_data(A, B):
    # case 0, both loaded
    if A['data'] == 'load' and B['data'] == 'load':
        assert A['data_load_from'] == B['data_load_from']
        data_path = './experiments/data/' + A['data_load_from']
    elif A['data'] == 'load' and (not B['data'] == 'load'):
        assert A['data_load_from'] == B['identifier']
        data_path = './experiments/data/' + A['data_load_from']
    elif (not A['data'] == 'load') and B['data'] == 'load':
        assert B['data_load_from'] == A['identifier']
        data_path = './experiments/data/' + A['identifier']
    else:
        raise ValueError(A['data'], B['data'])
    return data_path

def model_memorisation(identifier, epoch, max_samples=2000):
    """
    Compare samples from a model against training set and validation set in mmd
    """
    if identifier == 'cristobal_eICU':
        model_samples = pickle.load(open('REDACTED', 'rb'))
        samples, labels = data_utils.eICU_task()
        train = samples['train'].reshape(-1,16,4)
        vali = samples['vali'].reshape(-1,16,4)
        test = samples['test'].reshape(-1,16,4)
        #train_targets = labels['train']
        #vali_targets = labels['vali']
        #test_targets = labels['test']
        train, vali, test = data_utils.scale_data(train, vali, test)
        n_samples = test.shape[0]
        if n_samples > max_samples:
            n_samples = max_samples
            test = np.random.permutation(test)[:n_samples]
        if model_samples.shape[0] > n_samples:
            model_samples = np.random.permutation(model_samples)[:n_samples]
    elif identifier == 'cristobal_MNIST':
        the_dir = 'REDACTED'
        # pick a random one
        which = np.random.choice(['NEW_OK_', '_r4', '_r5', '_r6', '_r7'])
        model_samples, model_labels = pickle.load(open(the_dir + 'synth_mnist_minist_cdgan_1_2_100_multivar_14_nolr_rdim3_0_2_' + which + '_190.pk', 'rb'))
        # get test and train...
        # (generated with fixed seed...)
        mnist_resized_dim = 14
        samples, labels = data_utils.load_resized_mnist(mnist_resized_dim)
        proportions = [0.6, 0.2, 0.2]
        train, vali, test, labels_split = data_utils.split(samples, labels=labels, random_seed=1, proportions=proportions)
        np.random.seed()
        train = train.reshape(-1, 14, 14)
        test = test.reshape(-1, 14, 14)
        vali = vali.reshape(-1, 14, 14)
        n_samples = test.shape[0]
        if n_samples > max_samples:
            n_samples = max_samples
            test = np.random.permutation(test)[:n_samples]
        if model_samples.shape[0] > n_samples:
            model_samples = np.random.permutation(model_samples)[:n_samples]
    else:
        settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
        # get the test, train sets
        data = np.load('./experiments/data/' + identifier + '.data.npy').item()
        train = data['samples']['train']
        test = data['samples']['test']
        n_samples = test.shape[0]
        if n_samples > max_samples:
            n_samples = max_samples
            test = np.random.permutation(test)[:n_samples]
        model_samples = model.sample_trained_model(settings, epoch, n_samples)
    all_samples = np.vstack([train, test, model_samples])
    heuristic_sigma = mmd.median_pairwise_distance(all_samples)
    print('heuristic sigma:', heuristic_sigma)
    pvalue, tstat, sigma, MMDXY, MMDXZ = MMD_3_Sample_Test(model_samples, test, np.random.permutation(train)[:n_samples], sigma=heuristic_sigma, computeMMDs=False)
#    if pvalue < 0.05:
#        print('At confidence level 0.05, we reject the null hypothesis that MMDXY <= MMDXZ, and conclude that the test data has a smaller MMD with the true data than the generated data')
        # the function takes (X, Y, Z) as its first arguments, it's testing if MMDXY (i.e. MMD between model and train) is less than MMDXZ (MMd between model and test)
#    else:
#        print('We have failed to reject the null hypothesis that MMDXY <= MMDXZ, and cannot conclu#de that the test data has a smaller MMD with the true data than the generated data')
    return pvalue, tstat, sigma

def model_comparison(identifier_A, identifier_B, epoch_A=99, epoch_B=99):
    """
    Compare two models using relative MMD test
    """
    # make sure they used the same data
    settings_A = json.load(open('./experiments/settings/' + identifier_A + '.txt', 'r'))
    settings_B = json.load(open('./experiments/settings/' + identifier_B + '.txt', 'r'))
    data_path = assert_same_data(settings_A, settings_B)
    # now load the data
    data = np.load(data_path + '.data.npy').item()['samples']['vali']
    n_samples = data.shape[0]
    A_samples = model.sample_trained_model(settings_A, epoch_A, n_samples)
    B_samples = model.sample_trained_model(settings_B, epoch_B, n_samples)
    # do the comparison
    # TODO: support multiple signals
    ## some notes about this test:
    ## MMD_3_Sample_Test(X, Y, Z) tests the hypothesis that Px is closer to Pz than Py
    ## that is, test the null hypothesis H0:
    ##   MMD(F, Px, Py) <= MMD(F, Px, Pz)
    ## versus the alternate hypothesis:
    ##   MMD(F, Px, Py) > MMD(F, Px, Pz)
    ## at significance level that we select later (just the threshold on the p-value)
    pvalue, tstat, sigma, MMDXY, MMDXZ = MMD_3_Sample_Test(data[:, :, 0], A_samples[:, :, 0], B_samples[:, :, 0], computeMMDs=True)
    print(pvalue, tstat, sigma)
    if pvalue < 0.05:
        print('At confidence level 0.05, we reject the null hypothesis that MMDXY <= MMDXZ, and conclude that', identifier_B, 'has a smaller MMD with the true data than', identifier_A)
    else:
        print('We have failed to reject the null hypothesis that MMDXY <= MMDXZ, and cannot conclude that', identifier_B, 'has a smaller MMD with the true data than', identifier_A)
    return pvalue, tstat, sigma, MMDXY, MMDXZ

# --- to do with reconstruction --- #

def get_reconstruction_errors(identifier, epoch, g_tolerance=0.05, max_samples=10000, rerun=False):
    """
    Get the reconstruction error of every point in the training set of a given
    experiment.
    """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    if settings['data_load_from']:
        data_dict = np.load('./experiments/data/' + settings['data_load_from'] + '.data.npy').item()
    else:
        data_dict = np.load('./experiments/data/' + identifier + '.data.npy').item()
    samples = data_dict['samples']
    train = samples['train']
    vali = samples['vali']
    test = samples['test']
    try:
        if rerun:
            raise FileNotFoundError
        errors = np.load('./experiments/eval/' + identifier + '_' + str(epoch) + '_' + str(g_tolerance) + '.reconstruction_errors.npy').item()
        train_errors = errors['train']
        test_errors = errors['test']
        generated_errors = errors['generated']
        noisy_errors = errors['noisy']
        print('Loaded precomputed errors')
    except FileNotFoundError:
        n_eval = 500
        # generate "easy" samples from the distribution
        generated = model.sample_trained_model(settings, epoch, n_eval)
        # generate "hard' random samples, not from train/test distribution
        # TODO: use original validation examples, add noise etc.
    ##    random_samples = np.random.normal(size=generated.shape)
    #    random_samples -= np.mean(random_samples, axis=0) 
    #    random_samples += np.mean(vali, axis=0)
    #    random_samples /= np.std(random_samples, axis=0)
    #    random_samples *= np.std(vali, axis=0)

        # get all the errors
        print('Getting reconstruction errors on train set')
        if train.shape[0] > max_samples:
            train = np.random.permutation(train)[:max_samples]
        train_errors = error_per_sample(identifier, epoch, train, n_rep=5, g_tolerance=g_tolerance)
        print('Getting reconstruction errors on test set')
        if test.shape[0] > max_samples:
            test = np.random.permutation(test)[:max_samples]
        test_errors = error_per_sample(identifier, epoch, test, n_rep=5, g_tolerance=g_tolerance)
        print('Getting reconstruction errors on generated set')
        generated_errors = error_per_sample(identifier, epoch, generated, n_rep=5, g_tolerance=g_tolerance)
        print('Getting reconstruction errors on noisy set')
        alpha = 0.5
        noisy_samples = alpha*vali + (1-alpha)*np.random.permutation(vali)
        noisy_errors = error_per_sample(identifier, epoch, noisy_samples, n_rep=5, g_tolerance=g_tolerance)
        # save!
        errors = {'train': train_errors, 'test': test_errors, 'generated': generated_errors, 'noisy': noisy_errors}
        np.save('./experiments/eval/' + identifier + '_' + str(epoch) + '_' + str(g_tolerance) + '.reconstruction_errors.npy', errors)
    # do two-sample Kolomogorov-Smirnov test for equality
    D_test, p_test = ks_2samp(train_errors, test_errors)
    print('KS statistic and p-value for train v. test erors:', D_test, p_test)
    D_gen, p_gen = ks_2samp(generated_errors, train_errors)
    print('KS statistic and p-value for train v. gen erors:', D_gen, p_gen)
    D_gentest, p_gentest = ks_2samp(generated_errors, test_errors)
    print('KS statistic and p-value for gen v. test erors:', D_gentest, p_gentest)
    # visualise distribution of errors for train and test
    plotting.reconstruction_errors(identifier + '_' + str(epoch) + '_' + str(g_tolerance), train_errors, test_errors, generated_errors, noisy_errors)
    # visualise the "hardest" and "easiest" samples from train
    ranking_train = np.argsort(train_errors)
    easiest_train = ranking_train[:6]
    hardest_train = ranking_train[-6:]
    plotting.save_plot_sample(train[easiest_train], epoch, identifier + '_easytrain', n_samples=6, num_epochs=None, ncol=2)
    plotting.save_plot_sample(train[hardest_train], epoch, identifier + '_hardtrain', n_samples=6, num_epochs=None, ncol=2)
    # visualise the "hardest" and "easiest" samples from random
#    ranking_random = np.argsort(noisy_errors)
#    easiest_random = ranking_random[:6]
#    hardest_random = ranking_random[-6:]
#    plotting.save_plot_sample(random_samples[easiest_random], epoch, identifier + '_easyrandom', n_samples=6, num_epochs=None, ncol=2)
#    plotting.save_plot_sample(random_samples[hardest_random], epoch, identifier + '_hardrandom', n_samples=6, num_epochs=None, ncol=2)
    return True

def error_per_sample(identifier, epoch, samples, n_rep=3, n_iter=None, g_tolerance=0.025, use_min=True):
    """
    Get (average over a few runs) of the reconstruction error per sample
    """
    n_samples = samples.shape[0]
    heuristic_sigma = np.float32(mmd.median_pairwise_distance(samples))
    errors = np.zeros(shape=(n_samples, n_rep))
    for rep in range(n_rep):
        Z, rep_errors, sigma = model.invert(identifier, epoch, samples, n_iter=n_iter, heuristic_sigma=heuristic_sigma, g_tolerance=g_tolerance)
        errors[:, rep] = rep_errors
    # return min, or average?
    if use_min:
        errors = np.min(errors, axis=1)
    else:
        # use mean
        errors = np.mean(errors, axis=1)
    return errors

# --- visualisation evaluation --- #

def view_digit(identifier, epoch, digit, n_samples=6):
    """
    Generate a bunch of MNIST digits from a CGAN, view them
    """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    if settings['one_hot']:
        assert settings['max_val'] == 1
        assert digit <= settings['cond_dim']
        C_samples = np.zeros(shape=(n_samples, settings['cond_dim']))
        C_samples[:, digit] = 1
    else:
        assert settings['cond_dim'] == 1
        assert digit <= settings['max_val']
        C_samples = np.array([digit]*n_samples).reshape(-1, 1)
    digit_samples = model.sample_trained_model(settings, epoch, n_samples, Z_samples=None, cond_dim=settings['cond_dim'], C_samples=C_samples)
    digit_samples = digit_samples.reshape(n_samples, -1, 1)
    # visualise
    plotting.save_mnist_plot_sample(digit_samples, digit, identifier + '_' + str(epoch) + '_digit_', n_samples)
    return True

def view_interpolation(identifier, epoch, n_steps=6, input_samples=None, e_tolerance=0.01, sigma=3.29286853021):
    """
    If samples: generate interpolation between real points
    Else:
        Sample two points in the latent space, view a linear interpolation between them.
    """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    if input_samples is None:
        # grab two trainng examples
        data = np.load('./experiments/data/' + identifier + '.data.npy').item()
        train = data['samples']['train']
        input_samples = np.random.permutation(train)[:2]
#        Z_sampleA, Z_sampleB = model.sample_Z(2, settings['seq_length'], settings['latent_dim'], 
#                                          settings['use_time'])
        if sigma is None:
            ## gotta get a sigma somehow
            sigma = mmd.median_pairwise_distance(train)
            print('Calcualted heuristic sigma from training data:', sigma)
    Zs, error, _ = model.invert(settings, epoch, input_samples, e_tolerance=e_tolerance)
    Z_sampleA, Z_sampleB = Zs
    Z_samples = plotting.interpolate(Z_sampleA, Z_sampleB, n_steps=n_steps)
    samples = model.sample_trained_model(settings, epoch, Z_samples.shape[0], Z_samples)
    # get distances from generated samples to target samples
    d_A, d_B = [], []
    for sample in samples:
        d_A.append(sample_distance(sample, samples[0], sigma))
        d_B.append(sample_distance(sample, samples[-1], sigma))
    distances = pd.DataFrame({'dA': d_A, 'dB': d_B})
    plotting.save_plot_interpolate(input_samples, samples, epoch, settings['identifier'] + '_epoch' + str(epoch), distances=distances, sigma=sigma)
    return True

def view_latent_vary(identifier, epoch, n_steps=6):
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    Z_sample = model.sample_Z(1, settings['seq_length'], settings['latent_dim'], 
                                      settings['use_time'])[0]
    samples_dim = []
    for dim in range(settings['latent_dim']):
        Z_samples_dim = plotting.vary_latent_dimension(Z_sample, dim, n_steps)
        samples_dim.append(model.sample_trained_model(settings, epoch, Z_samples_dim.shape[0], Z_samples_dim))
    plotting.save_plot_vary_dimension(samples_dim, epoch, settings['identifier'] + '_varydim', n_dim=settings['latent_dim'])
    return True

def view_reconstruction(identifier, epoch, real_samples, tolerance=1):
    """
    Given a set of real samples, find the "closest" latent space points 
    corresponding to them, generate samples from these, visualise!
    """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    Zs, error, sigma = model.invert(settings, epoch, real_samples, tolerance=tolerance)
    plotting.visualise_latent(Zs[0], identifier+'_' + str(epoch) + '_0')
    plotting.visualise_latent(Zs[1], identifier+'_' + str(epoch) + '_1')
    model_samples = model.sample_trained_model(settings, epoch, Zs.shape[0], Zs)
    plotting.save_plot_reconstruct(real_samples, model_samples, settings['identifier'])
    return True

def view_fixed(identifier, epoch, n_samples=6, dim=None):
    """ What happens when we give the same point at each time step? """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    Z_samples = model.sample_Z(n_samples, settings['seq_length'], settings['latent_dim'], 
                                      settings['use_time'])
    # now, propagate forward the value at time 0 (which time doesn't matter)
    for i in range(1, settings['seq_length']):
        if dim is None:
            Z_samples[:, i, :] = Z_samples[:, 0, :]
        else:
            Z_samples[:, i, dim] = Z_samples[:, 0, dim]
    # now generate
    samples = model.sample_trained_model(settings, epoch, n_samples, Z_samples)
    # now visualise
    plotting.save_plot_sample(samples, epoch, identifier + '_fixed', n_samples)
    return True

def view_params(identifier, epoch):
    """ Visualise weight matrices in the GAN """
    settings = json.load(open('./experiments/settings/' + identifier + '.txt', 'r'))
    parameters = model.load_parameters(identifier + '_' + str(epoch))
    plotting.plot_parameters(parameters, identifier + '_' + str(epoch))
    return True

# --- to do with samples --- #

def sample_distance(sampleA, sampleB, sigma):
    """
    I know this isn't the best distance measure, alright.
    """
    # RBF!
    gamma = 1 / (2 * sigma**2)
    similarity = np.exp(-gamma*(np.linalg.norm(sampleA - sampleB)**2))
    distance = 1 - similarity
    return distance

### --- TSTR ---- ###

def TSTR_mnist(identifier, epoch):
    """
    Load synthetic training, real test data, do multi-class SVM
    (basically just this: http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
    """
    exp_data = np.load('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy').item()
    test_X, test_Y = exp_data['test_data'], exp_data['test_labels']
    train_X, train_Y = exp_data['train_data'], exp_data['train_labels']
    synth_X, synth_Y = exp_data['synth_data'], exp_data['synth_labels']
    # if multivariate, reshape
    if len(test_X.shape) == 3:
        test_X = test_X.reshape(test_X.shape[0], -1)
    if len(train_X.shape) == 3:
        train_X = train_X.reshape(train_X.shape[0], -1)
    if len(synth_X.shape) == 3:
        synth_X = synth_X.reshape(synth_X.shape[0], -1)
    # if one hot, fix
    if len(synth_Y.shape) > 1 and not synth_Y.shape[1] == 1:
        synth_Y = np.argmax(synth_Y, axis=1)
        train_Y = np.argmax(train_Y, axis=1)
        test_Y = np.argmax(test_Y, axis=1)
    # make classifier
    synth_classifier = SVC(gamma=0.001)
    real_classifier = SVC(gamma=0.001)
    # fit
    real_classifier.fit(train_X, train_Y)
    synth_classifier.fit(synth_X, synth_Y)
    # test on real
    synth_predY = synth_classifier.predict(test_X)
    real_predY = real_classifier.predict(test_X)
    # report on results
    print(classification_report(test_Y, synth_predY))
    print(classification_report(test_Y, real_predY))
    # visualise results
    plotting.view_mnist_eval(identifier + '_' + str(epoch), train_X, train_Y, synth_X, synth_Y, test_X, test_Y, synth_predY, real_predY)
    return True

def TSTR_eICU(identifier, epoch):
    """
    """
    # get "train" data
    exp_data = np.load('./experiments/tstr/' + identifier + '_' + str(epoch) + '.data.npy').item()
    X_synth = exp_data['synth_data']
    Y_synth = exp_data['synth_labels']
    n_synth = X_synth.shape[0]
    X_synth = X_synth.reshape(n_synth, -1)
    # get test data
    data = np.load('./data/eICU_task_data.npy').item()
    X_test = data['X_test']
    Y_test = data['Y_test']
    # iterate over labels
    results = []
    for label in range(Y_synth.shape[1]):
        print('task:', data['Y_columns'][label])
        print('(', np.mean(Y_synth[:, label]), 'positive in train, ', np.mean(Y_test[:, label]), 'in test)')
        #model = RandomForestClassifier(n_estimators=50).fit(X_synth, Y_synth[:, label])
        model = SVC(gamma=0.001).fit(X_synth, Y_synth[:, label])
        predict = model.predict(X_test)
        print('(predicted', np.mean(predict), 'positive labels)')
        accuracy = sklearn.metrics.accuracy_score(Y_test[:, label], predict)
        precision = sklearn.metrics.precision_score(Y_test[:, label], predict)
        recall = sklearn.metrics.recall_score(Y_test[:, label], predict)
        print('\tacc:', accuracy, '\tprec:', precision, '\trecall:', recall)
        results.append([accuracy, precision, recall])
    # do the OR task
    extreme_heartrate_test = Y_test[:, 1] + Y_test[:, 4]
    extreme_respiration_test = Y_test[:, 2] + Y_test[:, 5]
    extreme_systemicmean_test = Y_test[:, 3] + Y_test[:, 6]
    Y_OR_test = np.vstack([extreme_heartrate_test, extreme_respiration_test, extreme_systemicmean_test]).T
    Y_OR_test = (Y_OR_test > 0)*1

    extreme_heartrate_synth = Y_synth[:, 1] + Y_synth[:, 4]
    extreme_respiration_synth = Y_synth[:, 2] + Y_synth[:, 5]
    extreme_systemicmean_synth = Y_synth[:, 3] + Y_synth[:, 6]
    Y_OR_synth = np.vstack([extreme_heartrate_synth, extreme_respiration_synth, extreme_systemicmean_synth]).T
    Y_OR_synth = (Y_OR_synth > 0)*1

    OR_names = ['extreme heartrate', 'extreme respiration', 'extreme MAP']
    OR_results = []
    for label in range(Y_OR_synth.shape[1]):
        print('task:', OR_names[label])
        print('(', np.mean(Y_OR_synth[:, label]), 'positive in train, ', np.mean(Y_OR_test[:, label]), 'in test)')
        model = RandomForestClassifier(n_estimators=50).fit(X_synth, Y_OR_synth[:, label])
        predict = model.predict(X_test)
        print('(predicted', np.mean(predict), 'positive labels)')
        accuracy = sklearn.metrics.accuracy_score(Y_OR_test[:, label], predict)
        precision = sklearn.metrics.precision_score(Y_OR_test[:, label], predict)
        recall = sklearn.metrics.recall_score(Y_OR_test[:, label], predict)
        print(accuracy, precision, recall)
        OR_results.append([accuracy, precision, recall])
    return results, OR_results

def NIPS_toy_plot(identifier_rbf, epoch_rbf, identifier_sine, epoch_sine, identifier_mnist, epoch_mnist):
    """
    for each experiment:
    - plot a bunch of train examples
    - sample a bunch of generated examples
    - plot all in separate PDFs so i can merge in illustrator

    for sine and rbf, grey background
    MNIST is just MNIST (square though)
    """
    n_samples = 15
    # settings
    settings_rbf = json.load(open('./experiments/settings/' + identifier_rbf + '.txt', 'r'))
    settings_sine = json.load(open('./experiments/settings/' + identifier_sine + '.txt', 'r'))
    settings_mnist = json.load(open('./experiments/settings/' + identifier_mnist + '.txt', 'r'))
    # data
    data_rbf = np.load('./experiments/data/' + identifier_rbf + '.data.npy').item()
    data_sine = np.load('./experiments/data/' + identifier_sine + '.data.npy').item()
    data_mnist = np.load('./experiments/data/' + identifier_mnist + '.data.npy').item()
    train_rbf = data_rbf['samples']['train']
    train_sine = data_sine['samples']['train']
    train_mnist = data_mnist['samples']['train']
    # sample
    samples_rbf = model.sample_trained_model(settings_rbf, epoch_rbf, n_samples)
    samples_sine = model.sample_trained_model(settings_sine, epoch_sine, n_samples)
    samples_mnist = model.sample_trained_model(settings_mnist, epoch_mnist, n_samples)
    # plot them all
    index = 0
    #for sample in np.random.permutation(train_rbf)[:n_samples]:
    #    plotting.nips_plot_rbf(sample, index, 'train')
    #    index += 1
    #for sample in samples_rbf:
    #    plotting.nips_plot_rbf(sample, index, 'GAN')
    #    index += 1
    #for sample in np.random.permutation(train_sine)[:n_samples]:
    #    plotting.nips_plot_sine(sample, index, 'train')
    #    index += 1
    #for sample in samples_sine:
    #    plotting.nips_plot_sine(sample, index, 'GAN')
    #    index += 1
    for sample in np.random.permutation(train_mnist)[:n_samples]:
        plotting.nips_plot_mnist(sample, index, 'train')
        index += 1
    for sample in samples_mnist:
        plotting.nips_plot_mnist(sample, index, 'GAN')
        index += 1
    return True
