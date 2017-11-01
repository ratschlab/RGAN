import data_utils
import pandas as pd
import numpy as np
import tensorflow as tf
import math, random, itertools
import pickle
import time
import json
import os
import math
from data_utils import get_eICU_with_targets
import plotting

import model

# function for getting one mini batch
def get_batch(samples, labels, batch_size, batch_idx):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos], labels[start_pos:end_pos]

# directory where the data will be saved
wd = './synthetic_eICU_datasets'
if not os.path.isdir(wd):
    os.mkdir(wd)

# runs the experiment 5 times
identifiers = ['eICU_cdgan_synthetic_dataset_r' + str(i) for i in range(2,5)]

for identifier in identifiers:

    # reset tensorflow graph
    tf.reset_default_graph()

    print ("loading data...")

    samples, labels = data_utils.eICU_task()
    train_seqs = samples['train'].reshape(-1,16,4)
    vali_seqs = samples['vali'].reshape(-1,16,4)
    test_seqs = samples['test'].reshape(-1,16,4)
    train_targets = labels['train']
    vali_targets = labels['vali']
    test_targets = labels['test']
    train_seqs, vali_seqs, test_seqs = data_utils.scale_data(train_seqs, vali_seqs, test_seqs)

    print ("data loaded.")

    #training config
    lr = 0.1
    batch_size = 28
    num_epochs = 1005
    D_rounds = 1    # number of rounds of discriminator training
    G_rounds = 3    # number of rounds of generator training
    use_time = False    # use one latent dimension as time

    print(identifier)

    seq_length = train_seqs.shape[1]
    num_generated_features = train_seqs.shape[2]
    hidden_units_d = 100
    hidden_units_g = 100
    latent_dim = 10 # dimension of the random latent space
    cond_dim = train_targets.shape[1] # dimension of the condition

    CG = tf.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
    CD = tf.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    W_out_G = tf.Variable(tf.truncated_normal([hidden_units_g, num_generated_features]))
    b_out_G = tf.Variable(tf.truncated_normal([num_generated_features]))

    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    W_out_D = tf.Variable(tf.truncated_normal([hidden_units_d,1]))
    b_out_D = tf.Variable(tf.truncated_normal([1]))


    def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
        sample = np.random.normal(size=[batch_size, seq_length, latent_dim])
        if use_noisy_time or use_time:
            # time grid is time_grid_mult times larger than seq_length
            time_grid_mult = 5
            time_grid = (np.arange(seq_length*time_grid_mult)/((seq_length*time_grid_mult)/2)) - 1
            time_axes = []
            for i in range(batch_size):
                # randomly chose a starting point in the time grid
                starting_point = random.choice(np.arange(len(time_grid))[:-seq_length])
                time_axis = time_grid[starting_point:starting_point+seq_length]
                if use_noisy_time:
                    time_axis += np.random.normal(scale=2.0/len(time_axis), size=len(time_axis))
                time_axes.append(time_axis)
            sample[:,:,0] = time_axes
        return sample

        
    def generator(z, c):
        with tf.variable_scope("generator") as scope:
            
            # each step of the generator takes a random seed + the conditional embedding
            repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1],
                                                               cond_dim])        
            generator_input = tf.concat([repeated_encoding, z], 2)

            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_g, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=[seq_length]*batch_size,
                inputs=generator_input)
            rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
            logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
            output_2d = tf.nn.tanh(logits_2d)
            output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
        return output_3d


    def discriminator(x, c, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # correct?
            if reuse:
                scope.reuse_variables()
             
            # each step of the generator takes one time step of the signal to evaluate + 
            # its conditional embedding  
            repeated_encoding = tf.tile(c, [1, tf.shape(x)[1]])
            repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(x)[0], tf.shape(x)[1],
                                                               cond_dim])
            decoder_input = tf.concat([repeated_encoding, x], 2)
            
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=decoder_input)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_g])
            logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
            output = tf.nn.sigmoid(logits)
        return output, logits


    G_sample = generator(Z, CG)
    D_real, D_logit_real = discriminator(X, CD)
    D_fake, D_logit_fake = discriminator(G_sample, CG, reuse=True)

    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                         labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                         labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

    D_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(D_loss, var_list=discriminator_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #plot the ouput from the same seed
    vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    X_mb_vis, Y_mb_vis = get_batch(train_seqs, train_targets, batch_size, 0)
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_z, CG:Y_mb_vis})
    plotting.vis_eICU_patients_downsampled(vis_sample, seq_length, 
                identifier=identifier, idx=0)


    # visualise some real samples
    vis_real = np.float32(vali_seqs[np.random.choice(len(vali_seqs), size=batch_size), :, :])
    plotting.vis_eICU_patients_downsampled(vis_real, seq_length, 
                identifier=identifier + '_real', idx=0)


    trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
    trace.write('epoch D_loss G_loss time\n')
    print('epoch\tD_loss\tG_loss\ttime\n')
    t0 = time.time()


    def train_generator(batch_idx, offset):
        # update the generator
        for g in range(G_rounds):
            X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx + g + offset)
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={CG:Y_mb,
                                                 Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})
        return G_loss_curr


    def train_discriminator(batch_idx, offset):
        # update the discriminator
        for d in range(D_rounds):
            # using same input sequence for both the synthetic data and the real one,
            # probably it is not a good idea...
            X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx + d + offset)
            _, D_loss_curr = sess.run([D_solver, D_loss],
                                      feed_dict={CD:Y_mb, CG:Y_mb, X:X_mb, 
                                                 Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})

        return D_loss_curr


    for num_epoch in range(num_epochs):
        # we use D_rounds + G_rounds batches in each iteration
        for batch_idx in range(0, int(len(train_seqs) / batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds):
            # we should shuffle the data instead
            if num_epoch % 2 == 0:
                G_loss_curr = train_generator(batch_idx, 0)
                D_loss_curr = train_discriminator(batch_idx, G_rounds)
            else:
                D_loss_curr = train_discriminator(batch_idx, 0)           
                G_loss_curr = train_generator(batch_idx, D_rounds)

        t = time.time() - t0
        print(num_epoch,'\t', D_loss_curr, '\t', G_loss_curr, '\t', t)
       
        # record/visualise
        trace.write(str(num_epoch) + ' ' + str(D_loss_curr) + ' ' + str(G_loss_curr) + ' ' + str(t) + '\n')
        if num_epoch % 10 == 0:
            trace.flush()

        vis_sample = sess.run(G_sample, feed_dict={Z: vis_z, CG:Y_mb_vis})
        plotting.vis_eICU_patients_downsampled(vis_sample, seq_length, identifier=identifier, idx=num_epoch+1)

        # save synthetic data
        if num_epoch % 50 == 0:
            # generate synthetic dataset
            gen_samples = []
            labels_gen_samples = []
            print(int(len(train_seqs) / batch_size))
            for batch_idx in range(int(len(train_seqs) / batch_size)):
                X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx)
                z_ = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG:Y_mb})
                gen_samples.append(gen_samples_mb)
                labels_gen_samples.append(Y_mb)
                print (batch_idx)


            for batch_idx in range(int(len(vali_seqs) / batch_size)):
                X_mb, Y_mb = get_batch(vali_seqs, vali_targets, batch_size, batch_idx)
                z_ = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG:Y_mb})
                gen_samples.append(gen_samples_mb)
                labels_gen_samples.append(Y_mb)

            gen_samples = np.vstack(gen_samples)
            labels_gen_samples = np.vstack(labels_gen_samples)

            with open(wd + '/samples_' + identifier + '_' + str(num_epoch) + '.pk', 'wb') as f:
                pickle.dump(file=f, obj=gen_samples)

            with open(wd + '/labels_' + identifier + '_' + str(num_epoch) + '.pk', 'wb') as f:
                pickle.dump(file=f, obj=labels_gen_samples)

            # save the model used to generate this dataset
            model.dump_parameters(identifier + '_' + str(num_epoch), sess)
