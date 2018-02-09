# Targets not used, not possible to condition generated sequences 

import data_utils_2
import pandas as pd
import numpy as np
import tensorflow as tf
import math, random, itertools
import pickle
import time
import json
import os
import math
from data_utils_2 import get_data

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib
# change backend so that it can plot figures without an X-server running.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, time, json, random

import glob
import copy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', type=float)
parser.add_argument('-optimizer_str', type=str)
parser.add_argument('-hidden_units_dec', type=int)
parser.add_argument('-hidden_units_enc', type=int)
parser.add_argument('-emb_dim', type=int)
parser.add_argument('-mult', type=float)
parser.add_argument('-experiment_id', type=str)
args = parser.parse_args()

experiment_id = './' + args.experiment_id

# directory where the data will be saved
if not os.path.isdir(experiment_id):
    os.mkdir(experiment_id)

# function for getting one mini batch

def get_batch(samples, batch_idx, batch_size):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]


def save_plot_sample(samples, idx, identifier, n_samples=6, num_epochs=None, ncol=2, path='./'):
    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[1]
  
    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        for n in range(ncol):
            # first column
            sample = samples[n*nrow + m, :, 0]
            axarr[m, n].plot(x_points, sample, color=col)
            axarr[m, n].set_ylim(-1, 1)
    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)
    fig.savefig(path + "/" + identifier + "_sig" + str(idx).zfill(4) + ".png")
    print(path + "/" + identifier + "_sig" + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return

def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1, 
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, 
        random_seed=None, **kwargs):

    
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


########################
# DATA LOADING
########################

print ("loading data...")

samples = sine_wave()
inputs_train, inputs_validation, inputs_test = data_utils_2.split(samples, [0.6, 0.2, 0.2])

save_plot_sample(samples[0:7], '0', 'test_RVAE', path=experiment_id)

print ("data loaded.")

# runs the experiment 5 times
#identifiers = ['eICU_RVAE_synthetic_dataset_VAE_r' + str(i) for i in range(1)]
#for identifier in identifiers:
#identifier = identifiers[0]



#training config
batch_size = 32

print ("data loaded.")

seq_length = inputs_train.shape[1]
num_features = inputs_train.shape[2]
# not used
random_seed = 0


########################
# ENCODER
########################

def encoder(hidden_units_enc, emb_dim, mult):

    with tf.variable_scope("encoder") as scope:

        input_seq_enc = tf.placeholder(tf.float32, [batch_size, seq_length, num_features])

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_enc, state_is_tuple=True)
        enc_rnn_outputs, enc_rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            #sequence_length=[seq_length]*batch_size,
            inputs=input_seq_enc)

        z_mean = tf.layers.dense(enc_rnn_states[1], emb_dim)
        z_log_sigma_sq = tf.layers.dense(enc_rnn_states[1], emb_dim)

        # Draw one sample z from Gaussian distribution with mean 0 and std 1
        eps = tf.random_normal((batch_size, emb_dim), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        latent_emb = tf.add(z_mean, tf.multiply(tf.exp(tf.multiply(z_log_sigma_sq,0.5)), eps))

        latent_loss = mult * (-0.5) * tf.reduce_sum(1 + z_log_sigma_sq
                                       - tf.square(z_mean)
                                       - tf.exp(z_log_sigma_sq), 1)

        latent_loss = tf.reduce_mean(latent_loss)

        return input_seq_enc, enc_rnn_outputs, enc_rnn_states, latent_emb, latent_loss


########################
# DECODER
########################

def decoder(hidden_units_dec, latent_emb, input_seq_enc):

    with tf.variable_scope("decoder") as scope:

        W_out_dec = tf.Variable(tf.truncated_normal([hidden_units_dec,num_features]))
        b_out_dec = tf.Variable(tf.truncated_normal([num_features]))

        dec_inputs = tf.zeros(tf.shape(input_seq_enc))

        #use latent embedding as inputs
        #dec_inputs = tf.layers.dense(latent_emb, latent_emb.shape[1].value, activation=tf.nn.tanh)
        #dec_inputs = tf.tile(dec_inputs, [1, seq_length])
        #dec_inputs = tf.reshape(dec_inputs, [batch_size, seq_length, latent_emb.shape[1].value]) 

        dec_initial_state = tf.layers.dense(latent_emb, hidden_units_dec, activation=tf.nn.tanh)

        #init_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, hidden_units_dec]),tf.zeros([batch_size, hidden_units_dec]))
        #init_state = tf.contrib.rnn.LSTMStateTuple(dec_initial_state, dec_initial_state)

        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_units_dec)
        dec_rnn_outputs, dec_rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            #sequence_length=[seq_length]*batch_size,
            initial_state=dec_initial_state,
            inputs=dec_inputs)
        rnn_outputs_2d = tf.reshape(dec_rnn_outputs, [-1, hidden_units_dec])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_dec) + b_out_dec
        output_3d = tf.reshape(logits_2d, [-1, seq_length, num_features])
        #output_3d = tf.tanh(output_3d)
        reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(output_3d,input_seq_enc)))
        #reconstruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_3d, labels=input_seq_enc))

    return reconstruction_loss, output_3d



########################
# TRAINING
########################


learning_rate = [args.learning_rate]
delta_error = [0]
optimizer_str = [args.optimizer_str]
hidden_units_dec = [args.hidden_units_dec]
hidden_units_enc = [args.hidden_units_enc]
emb_dim = [args.emb_dim]
mult = [args.mult]

configs = itertools.product(learning_rate, delta_error, optimizer_str, hidden_units_dec, hidden_units_enc, emb_dim, mult)
config_keys = ['learning_rate', 'delta_error', 'optimizer_str', 'hidden_units_dec', 'hidden_units_enc', 'emb_dim', 'mult']
verbose = 3

max_epochs=10000
patience=5
minibatch_size_train=batch_size
minibatch_size_validation=batch_size
minibatch_size_test=batch_size


for config in configs:

    num_mini_batches_train = int(math.ceil(len(inputs_train) / float(minibatch_size_train)))
    num_mini_batches_validation = int(math.ceil(len(inputs_validation) / float(minibatch_size_validation)))
    num_mini_batches_test = int(math.ceil(len(inputs_test) / float(minibatch_size_test)))

    experiment_random_id = str(int(np.random.rand(1)*1000000))
    config_id = str(config).replace(", ", "_").replace("'", "")[1:-1] + "_" + experiment_random_id
    if verbose > 0:
        print(config_id)

    tf.reset_default_graph()
    
    learning_rate = config[config_keys.index('learning_rate')]
    delta_error = config[config_keys.index('delta_error')]
    optimizer_str = config[config_keys.index('optimizer_str')]
    

    with tf.variable_scope("trainer"):

        hidden_units_enc = config[config_keys.index('hidden_units_enc')]
        hidden_units_dec = config[config_keys.index('hidden_units_dec')]
        emb_dim = config[config_keys.index('emb_dim')]
        mult = config[config_keys.index('mult')]

        input_seq_enc, enc_rnn_outputs, enc_rnn_states, latent_emb, latent_loss = encoder(hidden_units_enc, emb_dim, mult)
        # when the network has the same length enc_rnn_states[1] == enc_rnn_outputs[:,-1,:]
        #in LSTM enc_rnn_states[0] is c; enc_rnn_states[1] is h
        #latent_emb = enc_rnn_states[1]
        reconstruction_loss, output_3d_pred = decoder(hidden_units_dec, latent_emb, input_seq_enc)
        cost = reconstruction_loss + latent_loss

        global_step = tf.Variable(np.int64(0), name='global_step', trainable=False)

        if (optimizer_str == "adagrad_epochs") or (optimizer_str == "adagrad_minibatch_iterations"):
            train = tf.train.AdagradDAOptimizer(learning_rate, global_step).minimize(cost)
        elif optimizer_str == "rmsprop":
            train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        elif optimizer_str == "adam":
            train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)        
        elif optimizer_str == "sgd":
            train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #train
    global_steps_count = 0
    keep_training = True
    epoch_counter = 0
    patience_counter = 0
    best_val_cost = 9999999999

    costs_train = []
    costs_val = []
    costs_test = []
    saved = False

    #Initial costs

    # start training
    weighted_cost_sum = 0
    for mbi in range(num_mini_batches_train):
        input_ = get_batch(inputs_train, mbi, minibatch_size_train)
        # FIX THIS! deal with last samples available in the set
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
            res = sess.run([cost], feed_dict=feed_dict)
            weighted_cost_sum += res[0]*len(input_)
    cost_train = weighted_cost_sum / len(inputs_train)
    costs_train.append(cost_train)
    if verbose > 1:
        print(cost_train)

    # validation cost
    weighted_cost_sum = 0
    for mbi in range(num_mini_batches_validation):
        input_ = get_batch(inputs_validation, mbi, minibatch_size_validation)
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
            res = sess.run([cost], feed_dict=feed_dict)
            weighted_cost_sum += res[0]*len(input_)
    cost_val = weighted_cost_sum / len(inputs_validation)
    costs_val.append(cost_val)
    if verbose > 1:
        print(cost_val)
    
    # compute test cost
    # this should be optional since it is not needed in every epoch
    # I am doing it to get the learning curve
    weighted_cost_sum = 0
    for mbi in range(num_mini_batches_test):
        input_ = get_batch(inputs_test, mbi, minibatch_size_test)
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
            res = sess.run([cost], feed_dict=feed_dict)
            weighted_cost_sum += res[0]*len(input_)
    cost_test = weighted_cost_sum / len(inputs_test)
    costs_test.append(cost_test)
    if verbose > 1:
        print(cost_test)

    print("++++++++++++++++++++++++++++++++++")


    # start training

    while keep_training:
        time_start = time.time()
        #shuffle data
        np.random.shuffle(inputs_train)

        weighted_cost_sum = 0
        for mbi in range(num_mini_batches_train):
            input_ = get_batch(inputs_train, mbi, minibatch_size_train)
            # FIX THIS! deal with last samples available in the set
            if len(input_) == batch_size:
                feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
                res = sess.run([train, cost, reconstruction_loss, latent_loss], feed_dict=feed_dict)
                if config[2] == "adagrad_epochs":
                    global_steps_count += 1
                # since last minibatch can have different lenth, we compute the mean cost as a
                # weighted mean
                weighted_cost_sum += res[1]*len(input_)
        cost_train = weighted_cost_sum / len(inputs_train)
        costs_train.append(cost_train)
        if verbose > 1:
            print(res[2])
            print(res[3])
            print(cost_train)
            print("-")

        # validation cost
        weighted_cost_sum = 0
        for mbi in range(num_mini_batches_validation):
            input_ = get_batch(inputs_validation, mbi, minibatch_size_validation)
            if len(input_) == batch_size:
                feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
                res = sess.run([cost], feed_dict=feed_dict)
                weighted_cost_sum += res[0]*len(input_)
        cost_val = weighted_cost_sum / len(inputs_validation)
        costs_val.append(cost_val)
        if verbose > 1:
            print(cost_val)
        
        # compute test cost
        # this should be optional since it is not needed in every epoch
        # I am doing it to get the learning curve
        weighted_cost_sum = 0
        for mbi in range(num_mini_batches_test):
            input_ = get_batch(inputs_test, mbi, minibatch_size_test)
            if len(input_) == batch_size:
                feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
                res = sess.run([cost], feed_dict=feed_dict)
                weighted_cost_sum += res[0]*len(input_)
        cost_test = weighted_cost_sum / len(inputs_test)
        costs_test.append(cost_test)
        
        # check patience
        if cost_val <= best_val_cost - delta_error:
            best_val_cost = cost_val
            patience_counter = 0
            save_path = saver.save(sess, "./" + experiment_id + "/" + config_id + "_model.ckpt")
            saved = True
        else:
            patience_counter += 1
            if patience_counter > patience:
                keep_training = False
                if saved:
                    saver.restore(sess, "./" + experiment_id + "/" + config_id + "_model.ckpt")            

        epoch_counter += 1
        if config[2] == "adagrad_minibatch_iterations":
            global_steps_count = epoch_counter
        if epoch_counter >= max_epochs:
            keep_training = False

        if verbose > 1:
            print(time.time() - time_start)
            print('--------------------')

        print((inputs_train[0] == inputs_train[1]).all())

    # save learning curve plots
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Learning curves')
    plt.plot(range(len(costs_train)), costs_train, label='training')
    plt.plot(range(len(costs_val)), costs_val, label='validation')
    plt.plot(range(len(costs_test)), costs_test, label='test')
    plt.legend(loc="upper right")
    plt.savefig("./" + experiment_id + "/" + config_id + "_learning_curves.png", dpi=300)

    # validation costs
    predicted_values = []
    predicted_values_not_flatten = []
    true_values = []
    true_values_not_flatten = []
    other_scores_validation = [] 
    weighted_cost_sum = 0
    for mbi in range(num_mini_batches_validation):
        input_ = get_batch(inputs_validation, mbi, minibatch_size_validation)
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
            res = sess.run([cost], feed_dict=feed_dict)
            weighted_cost_sum += res[0]*len(input_)
    cost_val = weighted_cost_sum / len(inputs_validation)
    costs_val.append(cost_val)
    other_scores_validation = {}

    # test costs
    predicted_values = []
    predicted_values_not_flatten = []
    true_values = []
    true_values_not_flatten = []
    other_scores_test = [] 
    weighted_cost_sum = 0
    for mbi in range(num_mini_batches_test):
        input_ = get_batch(inputs_test, mbi, minibatch_size_test)
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc:input_, global_step:global_steps_count}
            res = sess.run([cost], feed_dict=feed_dict)
            weighted_cost_sum += res[0]*len(input_)

    cost_test = weighted_cost_sum / len(inputs_test)
    costs_test.append(cost_test)
    other_scores_test = {}
        
    total_time = time.time() - time_start

    # need to convert values to float64 to be able to serialize them
    to_store = {'config': config, 'costs_train': costs_train, 'costs_val': costs_val, 'costs_test': costs_test,
               'best_val_cost': best_val_cost, 'total_time': total_time, 'random_seed': random_seed,
                'experiment_random_id': experiment_random_id, 'other_scores_validation': other_scores_validation,
               'other_scores_test': other_scores_test}

    with open("./" + experiment_id + "/" + config_id + ".json", "w") as f:
        json.dump(to_store, f)

    if verbose > 0:
        print("==========================")
        print(cost_val)
        print(best_val_cost)
        print(cost_test)
        print("==========================")


print((inputs_train[0] == inputs_train[1]).all())


########################
# SYNTHETIC SAMPLES GENERATION
########################

# Generate new samples
generated_samples = []
for i in range(num_mini_batches_train):
    feed_dict = {latent_emb:np.random.normal(size=(32,emb_dim))}
    res = sess.run([output_3d_pred], feed_dict=feed_dict)
    generated_samples.append(res[0])
generated_samples = np.vstack(generated_samples)

save_plot_sample(generated_samples[0:7], '_' + config_id + '_generated', 'test_RVAE', path=experiment_id)

input_ = get_batch(inputs_train, 0, minibatch_size_train)

save_plot_sample(input_[0:7], '_' + config_id + '_input', 'test_RVAE', path=experiment_id)

feed_dict = {input_seq_enc:input_}
res = sess.run([output_3d_pred], feed_dict=feed_dict)

save_plot_sample(res[0][0:7], '_' + config_id + '_reconstuction', 'test_RVAE', path=experiment_id)