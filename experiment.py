import numpy as np
import tensorflow as tf
import pdb
import random
import json
import argparse
from scipy.stats import mode

import data_utils
import plotting
import model

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='Train a GAN to generate \
                                 sequential, real-valued data.')
parser = argparse.ArgumentParser(description='Train a GAN to generate \
                                 sequential, real-valued data.')
# meta-option
parser.add_argument('--settings_file', help='json file of settings, overrides \
                    everything else', type=str, default='')
# options pertaining to data
parser.add_argument('--data', help='what kind of data to train with?', 
                    default='gp_rbf', 
                    choices=['gp_rbf', 'sine', 'mnist', 'load', 
                        'resampled_eICU', 'eICU_task'])
parser.add_argument('--num_samples', type=int, help='how many training examples \
                    to generate?', default=28*5*100)
parser.add_argument('--seq_length', type=int, default=30)
parser.add_argument('--num_signals', type=int, default=1)
parser.add_argument('--normalise', type=bool, default=False, help='normalise the \
        training/vali/test data (during split)?')
parser.add_argument('--cond_dim', type=int, default=0, help='dimension of \
        *conditional* input')
parser.add_argument('--max_val', type=int, default=1, help='assume conditional \
        codes come from [0, max_val)')
parser.add_argument('--one_hot', type=bool, default=False, help='convert categorical \
        conditional information to one-hot encoding')
parser.add_argument('--predict_labels', type=bool, default=False, help='instead \
        of conditioning with labels, require model to output them')
        ### for gp_rbf
parser.add_argument('--scale', type=float, default=0.1)
        ### for sin (should be using subparsers for this...)
parser.add_argument('--freq_low', type=float, default=1.0)
parser.add_argument('--freq_high', type=float, default=5.0)
parser.add_argument('--amplitude_low', type=float, default=0.1)
parser.add_argument('--amplitude_high', type=float, default=0.9)
        ### for mnist
parser.add_argument('--multivariate_mnist', type=bool, default=False)
parser.add_argument('--full_mnist', type=bool, default=False)
        ### for loading
parser.add_argument('--data_load_from', type=str, default='')
        ### for eICU
parser.add_argument('--resample_rate_in_min', type=int, default=15)
# hyperparameters of the model
parser.add_argument('--hidden_units_g', type=int, default=100)
parser.add_argument('--hidden_units_d', type=int, default=100)
parser.add_argument('--kappa', type=float, help='weight between final output \
                    and intermediate steps in discriminator cost (1 = all \
                    intermediate', default=1)
parser.add_argument('--latent_dim', type=int, default=5, help='dimensionality \
                    of the latent/noise space')
parser.add_argument('--batch_mean', type=bool, default=False, help='append the mean \
        of the batch to all variables for calculating discriminator loss')
parser.add_argument('--learn_scale', type=bool, default=True, help='make the \
        "scale" parameter at the output of the generator learnable (else fixed \
        to 1')
# options pertaining to training
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=28)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--D_rounds', type=int, default=5, help='number of rounds \
                    of discriminator training')
parser.add_argument('--G_rounds', type=int, default=1, help='number of rounds \
                    of generator training')
parser.add_argument('--use_time', type=bool, default=False, help='enforce \
                    latent dimension 0 to correspond to time')
parser.add_argument('--WGAN', type=bool, default=False)
parser.add_argument('--WGAN_clip', type=bool, default=False)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--wrong_labels', type=bool, default=False, help='augment \
        discriminator loss with real examples with wrong (~shuffled, sort of) labels')
# options pertaining to evaluation and exploration
parser.add_argument('--identifier', type=str, default='test', help='identifier \
                    string for output files')
settings = vars(parser.parse_args())

if settings['settings_file']:
    settings_path = './experiments/settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings = json.load(open(settings_path, 'r'))

# --- get data, split --- #
if settings['data_load_from']:
    data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
    samples, pdf, labels = data_utils.get_data('load', data_path)
    train, vali, test = samples['train'], samples['vali'], samples['test']
    train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
    del samples, labels
elif settings['data'] == 'eICU_task':
    samples, pdf, labels = data_utils.get_data('eICU_task', {})
    del samples, labels
    train, vali, test = samples['train'], samples['vali'], samples['test']
    train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
    assert train_labels.shape[1] == settings['cond_dim']
    # normalise to between -1, 1
    train, vali, test = data_utils.normalise_data(train, vali, test)
else:
    data_vars = ['num_samples', 'seq_length', 'num_signals', 'freq_low', 
                 'freq_high', 'amplitude_low', 'amplitude_high', 'scale',
                 'full_mnist']
    data_settings = dict((k, settings[k]) for k in data_vars if k in settings.keys())
    samples, pdf, labels = data_utils.get_data(settings['data'], data_settings)
    if 'multivariate_mnist' in settings and settings['multivariate_mnist']:
        seq_length = samples.shape[1]
        samples = samples.reshape(-1, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
    if 'normalise' in settings and settings['normalise']: # TODO this is a mess, fix
        print(settings['normalise'])
        norm = True
    else:
        norm = False
    if labels is None:
        train, vali, test = data_utils.split(samples, [0.6, 0.2, 0.2], normalise=norm)
        train_labels, vali_labels, test_labels = None, None, None
    else:
        train, vali, test, labels_list = data_utils.split(samples, [0.6, 0.2, 0.2], normalise=norm, labels=labels)
        train_labels, vali_labels, test_labels = labels_list

labels = dict()
labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels
del train_labels
del vali_labels
del test_labels

samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test
del train
del vali
del test

# --- futz around with labels --- #
if 'one_hot' in settings and settings['one_hot'] and not settings['data_load_from']:
    if len(labels['train'].shape) == 1:
        # ASSUME labels go from 0 to max_val inclusive, find max-val
        max_val = int(np.max([labels['train'].max(), labels['test'].max(), labels['vali'].max()]))
        # now we have max_val + 1 dimensions
        settings['cond_dim'] = max_val + 1
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
    settings['cond_dim'] = 0

# --- reaffirm/reset erroneous settings --- #
settings['seq_length'] = samples['train'].shape[1]
settings['num_samples'] = samples['train'].shape[0] + samples['vali'].shape[0] + samples['test'].shape[0]
settings['num_signals'] = samples['train'].shape[2]
settings['num_generated_features'] = samples['train'].shape[2]

# --- save settings, data --- #
for (k, v) in settings.items(): print(v, '\t',  k)
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

if not data == 'load':
    data_path = './experiments/data/' + identifier + '.data.npy'
    np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
    print('Saved training data to', data_path)

# --- initialise --- #

Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim)

discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 
                  'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

CGAN = (cond_dim > 0)
if CGAN: assert not predict_labels
D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings, kappa, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
D_solver, G_solver = model.GAN_solvers(D_loss, G_loss, learning_rate)
G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

# --- evaluation --- #

# frequency to do visualisations
vis_freq = max(14000//num_samples, 1)
eval_freq = max(7000//num_samples, 1)

# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000

# optimise sigma using that
batch_multiplier = 5000//batch_size
eval_size = batch_multiplier*batch_size
eval_eval_size = int(0.2*eval_size)
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
if CGAN:
    vis_C = model.sample_C(batch_size, cond_dim, max_val, one_hot)
    if 'mnist' in data:
        # set to be digits 0 to 5
        if cond_dim == 1:
            vis_C[:6] = np.arange(6)
        else: # assume one-hot
            vis_C[:6] = np.eye(6)
    elif 'eICU_task' in data:
        vis_C = train_labels[np.random.choice(labels['train'].shape[0], batch_size, replace=False), :]
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
else:
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})

vis_real_indices = np.random.choice(len(samples['vali']), size=batch_size)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
if not labels['vali'] is None:
    vis_real_labels = labels['vali'][vis_real_indices]
else:
    vis_real_labels = None
if data == 'mnist':
    if predict_labels:
        assert labels['vali'] is None
        n_labels = 1
        if one_hot: 
            n_labels = 6
            lab_votes = np.argmax(vis_real[:, :, -n_labels:], axis=2)
        else:
            lab_votes = vis_real[:, :, -n_labels:]
        labs, _ = mode(lab_votes, axis=1) 
        samps = vis_real[:, :, :-n_labels]
    else:
        labs = None
        samps = vis_real
    if multivariate_mnist:
        plotting.save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), 0, identifier + '_real', n_samples=6, labels=labs)
    else:
        plotting.save_mnist_plot_sample(samps, 0, identifier + '_real', n_samples=6, labels=labs)
elif 'eICU' in data:
    plotting.vis_eICU_patients_downsampled(vis_real, resample_rate_in_min, 
            identifier=identifier + '_real', idx=0)
else:
    plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=6, 
                            num_epochs=num_epochs)


trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2 that ll real_ll\n')

# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val', 
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)


t0 = time()
for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'],
                                        sess, Z, X, CG, CD, CS,
                                        D_loss, G_loss,
                                        D_solver, G_solver, 
                                        **train_settings)
    # -- eval -- #
   
    # if epoch % vis_freq == 0:
    #   eval.visualise(epoch, identifier, seq_length, vis_Z, vis_C, identifier, resample_rate_in_min)
    # visually
    if epoch % vis_freq == 0:
        if CGAN:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
        else:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        if data == 'mnist':
            if predict_labels:
                n_labels = 1
                if one_hot: 
                    n_labels = 6
                    lab_votes = np.argmax(vis_sample[:, :, -n_labels:], axis=2)
                else:
                    lab_votes = vis_sample[:, :, -n_labels:]
                labs, _ = mode(lab_votes, axis=1) 
                samps = vis_sample[:, :, :-n_labels]
            else:
                labs = None
                samps = vis_sample
            if multivariate_mnist:
                plotting.save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), epoch, identifier, n_samples=6, labels=labs)
            else:
                plotting.save_mnist_plot_sample(samps, epoch, identifier, n_samples=6, labels=labs)
        elif 'eICU' in data:
            plotting.vis_eICU_patients_downsampled(vis_sample[:6, :, :], resample_rate_in_min, 
                    identifier=identifier, idx=epoch)
        else:
            plotting.save_plot_sample(vis_sample, epoch, identifier, n_samples=6,
                                  num_epochs=num_epochs)

    # mmd, likelihood
    if epoch % eval_freq == 0:
        ## how many samples to evaluate with?
        eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
        if 'eICU_task' in data:
            eval_C = vali_labels[np.random.choice(vali_labels.shape[0], eval_size), :]
        else:
            eval_C = model.sample_C(eval_size, cond_dim, max_val, one_hot)
        eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))
        for i in range(batch_multiplier):
            if CGAN:
                eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size], CG: eval_C[i*batch_size:(i+1)*batch_size]})
            else:
                eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size]})
        eval_sample = np.float32(eval_sample)
        eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*batch_size), :, :])
       
        eval_eval_real = eval_real[:eval_eval_size]
        eval_test_real = eval_real[eval_eval_size:]
        eval_eval_sample = eval_sample[:eval_eval_size]
        eval_test_sample = eval_sample[eval_eval_size:]
        
        ## MMD
        # reset ADAM variables
        sess.run(tf.initialize_variables(sigma_opt_vars))
        sigma_iter = 0
        that_change = sigma_opt_thresh*2
        old_that = 0
        while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
            new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver], feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
            that_change = np.abs(that_np - old_that)
            old_that = that_np
            sigma_iter += 1
        opt_sigma = sess.run(sigma)
        mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample,biased=False, sigmas=sigma))
       
        ## save parameters
        if mmd2 < best_mmd2_so_far and epoch > 10:
            best_mmd2_so_far = mmd2
            model.dump_parameters(identifier + '_' + str(epoch), sess)
       
        ## likelihood (if available)
        if not pdf is None:
            ll_sample = np.mean(pdf(eval_sample[:, :, 0]))
            ll_real = np.mean(pdf(eval_real[:, :, 0]))
        else:
            ll_sample = 'NA'
            ll_real = 'NA'
    else:
        mmd2 = 'NA'
        ll_sample = 'NA'
        ll_real = 'NA'
  

    ## print
    t = time() - t0
    try:
        print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t %.2f\t %.2f' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, ll_sample, ll_real))
    except TypeError:       # mmd, ll are missing (strings)
        print('%d\t%.2f\t%.4f\t%.4f\t%s\t %s\t %s' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, ll_sample, ll_real))

    ## save trace
    trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, ll_sample, ll_real])) + '\n')
    if epoch % 10 == 0: 
        trace.flush()
        plotting.plot_trace(identifier, xmax=num_epochs)

    if shuffle:     # shuffle the training data 
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
        if labels['train'] is not None:
            labels['train'] = labels['train'][perm]
    
    if epoch % 50 == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)

trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs)
model.dump_parameters(identifier + '_' + str(epoch), sess)

## after-the-fact evaluation
#n_test = vali.shape[0]      # using validation set for now TODO
#n_batches_for_test = floor(n_test/batch_size)
#n_test_eval = n_batches_for_test*batch_size
#test_sample = np.empty(shape=(n_test_eval, seq_length, num_signals))
#test_Z = model.sample_Z(n_test_eval, seq_length, latent_dim, use_time)
#for i in range(n_batches_for_test):
#    test_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: test_Z[i*batch_size:(i+1)*batch_size]})
#test_sample = np.float32(test_sample)
#test_real = np.float32(vali[np.random.choice(n_test, n_test_eval, replace=False), :, :])
## we can only get samples in the size of the batch...
#heuristic_sigma = median_pairwise_distance(test_real, test_sample)
#test_mmd2, that = sess.run(mix_rbf_mmd2_and_ratio(test_real, test_sample, sigmas=heuristic_sigma, biased=False))
##print(test_mmd2, that)
