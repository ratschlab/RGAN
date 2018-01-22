import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb
from time import time
from matplotlib.colors import hsv_to_rgb
from pandas import read_table, read_hdf

import paths
from data_utils import scale_data

def visualise_at_epoch(vis_sample, data, predict_labels, one_hot, epoch,
        identifier, num_epochs, resample_rate_in_min, multivariate_mnist,
        seq_length, labels):
    # TODO: what's with all these arguments
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
            labs = labels
            samps = vis_sample
        if multivariate_mnist:
            save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), epoch, identifier, n_samples=6, labels=labs)
        else:
            save_mnist_plot_sample(samps, epoch, identifier, n_samples=6, labels=labs)
    elif 'eICU' in data:
        vis_eICU_patients_downsampled(vis_sample[:6, :, :],
                resample_rate_in_min,
                identifier=identifier,
                idx=epoch)
    else:
        save_plot_sample(vis_sample, epoch, identifier, n_samples=6,
                num_epochs=num_epochs)

    return True

def save_plot_sample(samples, idx, identifier, n_samples=6, num_epochs=None, ncol=2):
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
    fig.savefig("./experiments/plots/" + identifier + "_epoch" + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return

def save_plot_interpolate(input_samples, samples, idx, identifier,  num_epochs=None, distances=None, sigma=1):
    """ very boilerplate, unsure how to make nicer """
    n_samples = samples.shape[0]
    sample_length = samples.shape[1]

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)
    if distances is None:
        nrow = n_samples
    else:
        nrow = n_samples + 1
    ncol = 1
    fig, axarr = plt.subplots(nrow, ncol, figsize=(3, 9))
    if distances is None:
        startat = 0
    else:
        startat = 1
        axarr[0].plot(distances.dA, color='green', label='distance from A', linestyle='--', marker='o', markersize=4)
        axarr[0].plot(distances.dB, color='orange', label='distance from B', linestyle='dotted', marker='o', markersize=4)
        axarr[0].get_xaxis().set_visible(False)
        axarr[0].set_title('distance from endpoints')
    for m in range(startat, nrow):
        sample = samples[m-startat, :, 0]
        axarr[m].plot(x_points, sample, color=col)
    for m in range(startat, nrow):
        axarr[m].set_ylim(-1.1, 1.1)
        axarr[m].set_xlim(0, sample_length)
        axarr[m].spines["top"].set_visible(False)
        axarr[m].spines["bottom"].set_visible(False)
        axarr[m].spines["right"].set_visible(False)
        axarr[m].spines["left"].set_visible(False)
        axarr[m].tick_params(bottom='off', left='off')
        axarr[m].get_xaxis().set_visible(False)
        axarr[m].get_yaxis().set_visible(False)
        axarr[m].set_facecolor((0.96, 0.96, 0.96))
    if not input_samples is None:
        # now do the real samples
        axarr[startat].plot(x_points, input_samples[0], color='green', linestyle='--')
        axarr[-1].plot(x_points, input_samples[1], color='green', linestyle='--')

    axarr[-1].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.2)
    fig.savefig("./experiments/plots/" + identifier + "_interpolate.png")
    fig.savefig("./experiments/plots/" + identifier + "_interpolate.pdf")
    plt.clf()
    plt.close()
    return

def reconstruction_errors(identifier, train_errors, vali_errors,
                          generated_errors, random_errors):
    """
    Plot two histogram of the reconstruction errors.
    """
    print(identifier)
    fig, axarr = plt.subplots(4, 1, sharex=True, figsize=(4, 8))
    axarr[0].hist(train_errors, normed=1, color='green', bins=50)
    axarr[0].set_title("train reconstruction errors")
    axarr[1].hist(vali_errors, normed=1, color='blue', bins=50)
    axarr[1].set_title('vali reconstruction errors')
    axarr[2].hist(generated_errors, normed=1, color='pink', bins=50)
    axarr[2].set_title('generated reconstruction errors')
    axarr[3].hist(random_errors, normed=1, color='grey', bins=50)
    axarr[3].set_title('random reconstruction errors')
    for ax in axarr:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom='off', left='off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    axarr[3].set_xlim(0, 0.05)
    plt.tight_layout()
    plt.savefig('./experiments/plots/' + identifier + '_reconstruction_errors.png')
    return True

def save_plot_reconstruct(real_samples, model_samples, identifier):
    assert real_samples.shape == model_samples.shape
    sample_length = real_samples.shape[1]
    x_points = np.arange(sample_length)
    nrow = real_samples.shape[0]
    ncol = 2
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        real_sample = real_samples[m, :, 0]
        model_sample = model_samples[m, :, 0]
        axarr[m, 0].plot(x_points, real_sample, color='green')
        axarr[m, 1].plot(x_points, model_sample, color='red')
    axarr[-1, 0].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    axarr[-1, 1].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    axarr[0, 0].set_title('real')
    axarr[0, 1].set_title('reconstructed')
    fig.subplots_adjust(hspace = 0.15)
    fig.savefig("./experiments/plots/" + identifier + "_reconstruct.png")
    plt.clf()
    plt.close()
    return

def save_plot_vary_dimension(samples_list, idx, identifier, n_dim):
    """
    """
    assert len(samples_list) == n_dim
    sample_length = samples_list[0].shape[1]

    x_points = np.arange(sample_length)

    nrow = samples_list[0].shape[0]
    sidelength = n_dim*1.5
    fig, axarr = plt.subplots(nrow, n_dim, sharex=True, sharey=True, figsize=(sidelength, sidelength))
    for dim in range(n_dim):
        sample_dim = samples_list[dim]
        axarr[0, dim].set_title(dim)
        h = dim*1.0/n_dim       # hue
        for n in range(nrow):
            sample = sample_dim[n, :, 0]
            axarr[n, dim].plot(x_points, sample, color='black')
            axarr[n, dim].spines["top"].set_visible(False)
            axarr[n, dim].spines["bottom"].set_visible(False)
            axarr[n, dim].spines["right"].set_visible(False)
            axarr[n, dim].spines["left"].set_visible(False)
            axarr[n, dim].tick_params(bottom='off', left='off')
            axarr[n, dim].get_xaxis().set_visible(False)
            axarr[n, dim].set_facecolor(hsv_to_rgb((h, 0 + 0.25*n/nrow, 0.96)))
        axarr[-1, dim].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.11, wspace=0.11)
    fig.savefig("./experiments/plots/" + identifier + "_epoch" + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return True

def interpolate(sampleA, sampleB=None, n_steps=6):
    """
    Plot the linear interpolation between two latent space points.
    """
    weights = np.linspace(0, 1, n_steps)
    if sampleB is None:
        # do it "close by"
        sampleB = sampleA + np.random.normal(size=sampleA.shape, scale=0.05)
    samples = np.array([w*sampleB + (1-w)*sampleA for w in weights])
    return samples

def vary_latent_dimension(sample, dimension, n_steps=6):
    """
    """
    assert dimension <= sample.shape[1]
    scale = np.mean(np.abs(sample[:, dimension]))
    deviations = np.linspace(0, 2*scale, n_steps)
    samples = np.array([sample[:, :]]*n_steps)
    for n in range(n_steps):
        samples[n, :, dimension] += deviations[n]
    return samples

def plot_sine_evaluation(real_samples, fake_samples, idx, identifier):
    """
    Create histogram of fake (generated) samples frequency, amplitude distribution.
    Also for real samples.
    """
    ### frequency
    seq_length = len(real_samples[0])    # assumes samples are all the same length
    frate = seq_length
    freqs_hz = np.fft.rfftfreq(seq_length)*frate      # this is for labelling the plot
    # TODO, just taking axis 0 for now...
    w_real = np.mean(np.abs(np.fft.rfft(real_samples[:, :, 0])), axis=0)
    w_fake = np.mean(np.abs(np.fft.rfft(fake_samples[:, :, 0])), axis=0)
    ### amplitude
    A_real = np.max(np.abs(real_samples[:, :, 0]), axis=1)
    A_fake = np.max(np.abs(fake_samples[:, :, 0]), axis=1)
    ### now plot
    nrow = 2
    ncol = 2
    fig, axarr = plt.subplots(nrow, ncol, sharex='col', figsize=(6, 6))
    # freq
    axarr[0, 0].vlines(freqs_hz, ymin=np.minimum(np.zeros_like(w_real), w_real), ymax=np.maximum(np.zeros_like(w_real), w_real), color='#30ba50')
    axarr[0, 0].set_title("frequency", fontsize=16)
    axarr[0, 0].set_ylabel("real", fontsize=16)
    axarr[1, 0].vlines(freqs_hz, ymin=np.minimum(np.zeros_like(w_fake), w_fake), ymax=np.maximum(np.zeros_like(w_fake), w_fake), color='#ba4730')
    axarr[1, 0].set_ylabel("generated", fontsize=16)
    # amplitude
    axarr[0, 1].hist(A_real, normed=True, color='#30ba50', bins=30)
    axarr[0, 1].set_title("amplitude", fontsize=16)
    axarr[1, 1].hist(A_fake, normed=True, color='#ba4730', bins=30)

    fig.savefig('./experiments/plots/' + identifier + '_eval' + str(idx).zfill(4) +'.png')
    plt.clf()
    plt.close()
    return True

def plot_trace(identifier, xmax=250, final=False, dp=False):
    """
    """

    trace_path = './experiments/traces/' + identifier + '.trace.txt'
    da = read_table(trace_path, sep=' ')
    nrow = 3
    if dp:
        trace_dp_path = './experiments/traces/' + identifier + '.dptrace.txt'
        da_dp = read_table(trace_dp_path, sep=' ')
        nrow += 1

    ncol=1
    fig, axarr = plt.subplots(nrow, ncol, sharex='col', figsize=(6, 6))

    # D_loss
    d_handle,  = axarr[0].plot(da.epoch, da.D_loss, color='red', label='discriminator')
    axarr[0].set_ylabel('D loss')
#    axarr[0].set_ylim(0.9, 1.6)
    if final:
        #D_ticks = [1.0, 1.2, 1.5]
        D_ticks = [0.5, 1.0, 1.5]
        axarr[0].get_yaxis().set_ticks(D_ticks)
        for tick in D_ticks:
            axarr[0].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)
    # G loss
    ax_G = axarr[0].twinx()
    g_handle,  = ax_G.plot(da.epoch, da.G_loss, color='green', ls='dashed', label='generator')
    ax_G.set_ylabel('G loss')
    if final:
        G_ticks = [2.5, 5]
        ax_G.get_yaxis().set_ticks(G_ticks)
#        for tick in G_ticks:
#            axarr[0].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='green', alpha=1.0, zorder=0)

    ax_G.spines["top"].set_visible(False)
    ax_G.spines["bottom"].set_visible(False)
    ax_G.spines["right"].set_visible(False)
    ax_G.spines["left"].set_visible(False)
    ax_G.tick_params(bottom='off', right='off')
    axarr[0].legend(handles=[d_handle, g_handle], labels=['discriminator', 'generator'])

    # mmd
    da_mmd = da.loc[:, ['epoch', 'mmd2']].dropna()
    axarr[1].plot(da_mmd.epoch, da_mmd.mmd2, color='purple')
    axarr[1].set_ylabel('MMD$^2$')
    #axarr[1].set_ylim(0.0, 0.04)

    #ax_that = axarr[1].twinx()
    #ax_that.plot(da.that)
    #ax_that.set_ylabel('$\hat{t}$')
    #ax_that.set_ylim(0, 50)
    if final:
        mmd_ticks = [0.01, 0.02, 0.03]
        axarr[1].get_yaxis().set_ticks(mmd_ticks)
        for tick in mmd_ticks:
            axarr[1].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)

    # log likelihood
    da_ll = da.loc[:, ['epoch', 'll', 'real_ll']].dropna()
    axarr[2].plot(da_ll.epoch, da_ll.ll, color='orange')
    axarr[2].plot(da_ll.epoch, da_ll.real_ll, color='orange', alpha=0.5)
    axarr[2].set_ylabel('likelihood')
    axarr[2].set_xlabel('epoch')
    axarr[2].set_ylim(-750, 100)
    #axarr[2].set_ylim(-10000000, 500)
    if final:
#        ll_ticks = [-1.0*1e7, -0.5*1e7, 0]
        ll_ticks = [-500 ,-250, 0]
        axarr[2].get_yaxis().set_ticks(ll_ticks)
        for tick in ll_ticks:
            axarr[2].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)

    if dp:
        assert da_dp.columns[0] == 'epoch'
        epochs = da_dp['epoch']
        eps_values = da_dp.columns[1:]
        for eps_string in eps_values:
            if 'eps' in eps_string:
                eps = eps_string[3:]
            else:
                eps = eps_string
            deltas = da_dp[eps_string]
            axarr[3].plot(epochs, deltas, label=eps)
            axarr[3].set_ylabel('delta')
            axarr[3].set_xlabel('epoch')
        axarr[3].legend()

    # beautify
    for ax in axarr:
        #ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color((0, 0, 0, 0.3))
        #ax.spines["bottom"].set_visible(False)
        ax.spines["bottom"].set_color((0, 0, 0, 0.3))
        #ax.spines["right"].set_visible(False)
        ax.spines["right"].set_color((0, 0, 0, 0.3))
        #ax.spines["left"].set_visible(False)
        ax.spines["left"].set_color((0, 0, 0, 0.3))
        ax.tick_params(bottom='off', left='off')
        # make background grey
    #    ax.set_facecolor((0.96, 0.96, 0.96))
        ymin, ymax = ax.get_ylim()
        for x in np.arange(0, xmax+10, 10):
            ax.plot((x, x), (ymin, ymax), ls='dotted', lw=0.5, color='black', alpha=0.40, zorder=0)
        ax.set_xlim(-5, xmax)
        ax.get_yaxis().set_label_coords(-0.11,0.5)

    # bottom one

    fig.savefig('./experiments/traces/' + identifier + '_trace.png')
    fig.savefig('./experiments/traces/' + identifier + '_trace.pdf')
    plt.clf()
    plt.close()
    return True

### scripts for eICU

def vis_eICU_patients(patients, upto=None, identifier=None):
    """
    Given a list of patientIDs, visualise the chosen variables.
    (if only one patient given, only vis one patient)
    """
    patients = list(set(patients))
    print('Plotting traces of', len(patients), 'patients.')
    eICU_dir = 'REDACTED'
    variables = ['temperature', 'heartrate', 'respiration', 'systemicmean']
    # set up the plot
    fig, axarr = plt.subplots(len(variables), 1, sharex=True, figsize=(6.5, 9))
    for patient in patients:
        pat_df = read_hdf(eICU_dir + '/vitalPeriodic.h5', where='patientunitstayid = ' + str(patient), columns=['observationoffset'] + variables, mode='r')
        pat_df.set_index('observationoffset', inplace=True)
        pat_df.sort_index(inplace=True)
        if not upto is None:
            # restrict to first "upto" minutes
            pat_df = pat_df.loc[0:upto*60]
        for variable, ax in zip(variables, axarr):
            ax.plot(pat_df.index/60, pat_df[variable], alpha=0.5)
    # aesthetics
    xmin, xmax = axarr[0].get_xlim()
    for variable, ax in zip(variables, axarr):
        ax.set_ylabel(variable)
        ax.get_yaxis().set_label_coords(-0.15,0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom='off')
        #ax.set_facecolor((0.96, 0.96, 0.96))
        ymin, ymax = ax.get_ylim()
        # expand the ylim ever so slightly
        yrange = np.abs(ymax - ymin)
        ybuffer = yrange*0.08
        ymin_new = ymin - ybuffer
        ymax_new = ymax + ybuffer
        for x in np.linspace(xmin, xmax - (xmax - xmin)*0.005, num=10):
            ax.plot((x, x), (ymin_new, ymax_new), ls='dotted', lw=0.5, color='black', alpha=0.25, zorder=0)
        ax.set_ylim(ymin_new, ymax_new)
    axarr[-1].set_xlabel("time since admision (hours)")
    axarr[-1].get_xaxis().tick_bottom()
    if not identifier is None:
        plt.suptitle(identifier)
        fig.savefig('./plots/' + identifier + '.png', bbox_inches='tight')
    else:
        fig.savefig('./plots/eICU_patients.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def save_mnist_plot_sample(samples, idx, identifier, n_samples, labels=None):
    """
    Generates a grid showing mnist digits.

    """
    assert n_samples <= samples.shape[0]
    if not labels is None:
        assert n_samples <= len(labels)
        if len(labels.shape) > 1 and not labels.shape[1] == 1:
            # one-hot
            label_titles = np.argmax(labels, axis=1)
        else:
            label_titles = labels
    else:
        label_titles = ['NA']*n_samples
    assert n_samples % 2 == 0
    img_size = int(np.sqrt(samples.shape[1]))

    nrow = int(n_samples/2)
    ncol = 2
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(8, 8))
    for m in range(nrow):
        # first column
        sample = samples[m, :, 0]
        axarr[m, 0].imshow(sample.reshape([img_size,img_size]), cmap='gray')
        axarr[m, 0].set_title(str(label_titles[m]))
        # second column
        sample = samples[nrow + m, :, 0]
        axarr[m, 1].imshow(sample.reshape([img_size,img_size]), cmap='gray')
        axarr[m, 1].set_title(str(label_titles[m + nrow]))
    fig.suptitle(idx)
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)
    fig.savefig("./experiments/plots/" + identifier + "_epoch" + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return

def visualise_latent(Z, identifier):
    """
    visualise a SINGLE point in the latent space
    """
    seq_length = Z.shape[0]
    latent_dim = Z.shape[1]
    if latent_dim > 2:
        print('WARNING: Only visualising first two dimensions of latent space.')
    h = np.random.random()
    colours = np.array([hsv_to_rgb((h, i/seq_length, 0.96)) for i in range(seq_length)])
#    plt.plot(Z[:, 0], Z[:, 1], c='grey', alpha=0.5)
    for i in range(seq_length):
        plt.scatter(Z[i, 0], Z[i, 1], marker='o', c=colours[i])
    plt.savefig('./experiments/plots/' + identifier + '_Z.png')
    plt.clf()
    plt.close()
    return True



# --- to do with the model --- #
def plot_parameters(parameters, identifier):
    """
    visualise the parameters of a GAN
    """
    generator_out = parameters['generator/W_out_G:0']
    generator_weights = parameters['generator/rnn/lstm_cell/weights:0'] # split this into four
    generator_matrices = np.split(generator_weights, 4, 1)
    fig, axarr = plt.subplots(5, 1, sharex=True,
            gridspec_kw = {'height_ratios':[0.2, 1, 1, 1, 1]}, figsize=(3,13))

    axarr[0].matshow(generator_out.T, extent=[0,100,0,100])
    axarr[0].set_title('W_out_G')
    axarr[1].matshow(generator_matrices[0])
    axarr[1].set_title('LSTM weights (1)')
    axarr[2].matshow(generator_matrices[1])
    axarr[2].set_title('LSTM weights (2)')
    axarr[3].matshow(generator_matrices[2])
    axarr[3].set_title('LSTM weights (3)')
    axarr[4].matshow(generator_matrices[3])
    axarr[4].set_title('LSTM weights (4)')
    for a in axarr:
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.spines["top"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
#        a.tick_params(bottom='off', left='off', top='off')
    plt.tight_layout()
    plt.savefig('./experiments/plots/' + identifier + '_weights.png')
    return True


def vis_eICU_patients_downsampled(pat_arrs, time_step, time_steps_to_plot=None,
        variable_names=['sao2', 'heartrate', 'respiration', 'systemicmean'],
        identifier=None, idx=0):
    """
    Given a list of patient dataframes, visualise the chosen variables.
    (if only one patient given, only vis one patient)
    """
    # set up the plot
    fig, axarr = plt.subplots(len(variable_names), 1, sharex=True, figsize=(6.5, 9))
    # fix the same colour for each patient for each axis
    n_patients = pat_arrs.shape[0]
    colours = [hsv_to_rgb((i/n_patients, 0.8, 0.8)) for i in range(n_patients)]
    for (i, pat_arr) in enumerate(pat_arrs):
        if not time_steps_to_plot is None:
            pat_arr = pat_arr[0:time_steps_to_plot]
        for col, ax in zip(range(pat_arr.shape[1]), axarr):
            ax.plot(range(0, len(pat_arr)*time_step, time_step), pat_arr[:,col], alpha=0.5, color=colours[i])
    # aesthetics
    xmin, xmax = axarr[0].get_xlim()
    for variable, ax in zip(variable_names, axarr):
        ax.set_ylabel(variable)
        ax.get_yaxis().set_label_coords(-0.15,0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom='off')
        #ax.set_facecolor((0.96, 0.96, 0.96))
        ymin, ymax = ax.get_ylim()
        # expand the ylim ever so slightly
        yrange = np.abs(ymax - ymin)
        ybuffer = yrange*0.08
        ymin_new = ymin - ybuffer
        ymax_new = ymax + ybuffer
        for x in np.linspace(xmin, xmax - (xmax - xmin)*0.005, num=10):
            ax.plot((x, x), (ymin_new, ymax_new), ls='dotted', lw=0.5, color='black', alpha=0.25, zorder=0)
        #ax.set_ylim(ymin_new, ymax_new)
        ax.set_ylim(-1.5, 1.5)
    axarr[-1].set_xlabel("time since admision (minutes)")
    axarr[-1].get_xaxis().tick_bottom()
    if not identifier is None:
        plt.suptitle(idx)
        fig.savefig("./experiments/plots/" + identifier + "_epoch" + str(idx).zfill(4) + ".png", bbox_inches='tight')
    else:
        fig.savefig('./experiments/plots/eICU_patients.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

### TSTR ###
def view_mnist_eval(identifier, train_X, train_Y, synth_X, synth_Y, test_X, test_Y, synth_predY, real_predY):
    """
    Basically just
    http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
    """
    # resize everything
    side_length = int(np.sqrt(train_X.shape[1]))
    train_X = train_X.reshape(-1, side_length, side_length)
    synth_X = synth_X.reshape(-1, side_length, side_length)
    test_X = test_X.reshape(-1, side_length, side_length)
    # remember, they're wrecked in the outer function thanks to python
    synth_images_and_labels = list(zip(synth_X, synth_Y))
    for index, (image, label) in enumerate(synth_images_and_labels[:4]):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('synth train: %i' % label)
        else:
            plt.title('%i' % label)
    train_images_and_labels = list(zip(train_X, train_Y))
    for index, (image, label) in enumerate(train_images_and_labels[:4]):
        plt.subplot(4, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('real train: %i' % label)
        else:
            plt.title('%i' % label)
    images_and_synthpreds = list(zip(test_X, synth_predY))
    for index, (image, prediction) in enumerate(images_and_synthpreds[:4]):
        plt.subplot(4, 4, index + 9)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('synth pred: %i' % prediction)
        else:
            plt.title('%i' % prediction)
    images_and_realpreds = list(zip(test_X, real_predY))
    for index, (image, prediction) in enumerate(images_and_realpreds[:4]):
        plt.subplot(4, 4, index + 13)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        if index == 0:
            plt.title('real pred: %i' % prediction)
        else:
            plt.title('%i' % prediction)
    plt.tight_layout()
    plt.title(identifier)
    plt.savefig('./experiments/tstr/' + identifier + '_preds.png')
    return True

def view_marginals_raw(data, label=''):
    """
    Sort of a duplication with 'view_marginals_cristobal', this doesn't attempt to compare distributions or anything.
    """
    variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']

    num_gradations = 25
    # for cutoff in the gradations, what fraction of samples (at a given time point) fall into that cutoff bracket?
    grid = np.zeros(shape=(16, num_gradations, 4))
    grid = np.zeros(shape=(16, num_gradations, 4))
    assert data.shape[-1] == 4
    ranges = []
    for var in range(4):
        # allow for a different range per variable (if zoom)
        low = np.min(data[:, :, var])
        high = np.max(data[:, :, var])
        ranges.append([low, high])
        gradations = np.linspace(low, high, num_gradations)
        for (i, cutoff) in enumerate(gradations):
            # take the mean over samples
            frac = ((data[:, :, var] > low) & (data[:, :, var] <= cutoff)).mean(axis=0)
            low = cutoff
            grid[:, i, var] = frac

    fig, axarr = plt.subplots(nrows=4, ncols=1, sharex=True)
    axarr[0].imshow(grid[:, :, 0].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[1].imshow(grid[:, :, 1].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[2].imshow(grid[:, :, 2].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[3].imshow(grid[:, :, 3].T, origin='lower', aspect=0.5, cmap='magma_r')

    for (var, ax) in enumerate(axarr):
        labels = np.round(np.linspace(ranges[var][0], ranges[var][1], num_gradations)[1::4], 0)
        ax.set_yticks(np.arange(num_gradations)[1::4])
        ax.set_yticklabels(labels)
        ax.set_ylabel(variables[var])
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_adjustable('box-forced')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(b=True, color='black', alpha=0.2, linestyle='--')

    axarr[-1].set_xticks(np.arange(16)[::2])

    plt.tight_layout(pad=0.0, w_pad=-5.0, h_pad=0.1)
    plt.savefig("./experiments/eval/eICU_marginals_" + label + ".png")

    return True

def view_marginals_cristobal(rep=0, epoch=300, zoom=False):
    """
    View marginals of the synthetic data (compare to real data), from the data Cristobal generated.
    """
    samples_path = paths.eICU_synthetic_dir + 'samples_eICU_cdgan_synthetic_dataset_r' + str(rep) + '_' + str(epoch) + '.pk'
    samples = np.load(samples_path)
    labels_path = paths.eICU_synthetic_dir + 'labels_eICU_cdgan_synthetic_dataset_r' + str(rep) + '_' + str(epoch) + '.pk'
    labels = np.load(labels_path)
    real_path = paths.eICU_task_data
    raw_real_train = np.load(real_path).item()['X_train'].reshape(-1, 16, 4)
    real_test = np.load(real_path).item()['X_test'].reshape(-1, 16, 4)
    real_vali = np.load(real_path).item()['X_vali'].reshape(-1, 16, 4)
    # discard vali, test
    real, scaled_vali, scaled_test = scale_data(raw_real_train, real_vali, real_test)
    real = raw_real_train
    view_marginals_raw(raw_real_train, label='raw_real_train')
    view_marginals_raw(real, label='real_train')
    view_marginals_raw(samples, label='synthetic')

    variables = ['sao2', 'heartrate', 'respiration', 'systemicmean']

    # get the scaling factors
    scaling_factors = {'a': np.zeros(shape=(16, 4)), 'b': np.zeros(shape=(16, 4))}
    ranges = []
    for var in range(4):
        var_min = 100
        var_max = 0
        for timestep in range(16):
            min_val = np.min([np.min(raw_real_train[:, timestep, var]), np.min(real_vali[:, timestep, var])])
            max_val = np.max([np.max(raw_real_train[:, timestep, var]), np.max(real_vali[:, timestep, var])])
            if min_val < var_min:
                var_min = min_val
            if max_val > var_max:
                var_max = max_val
            a = (max_val - min_val)/2
            b = (max_val + min_val)/2
            scaling_factors['a'][timestep, var] = a
            scaling_factors['b'][timestep, var] = b
        ranges.append([var_min, var_max])

    # now, scale the synthetic data manually
    samples_scaled = np.zeros_like(samples)
    for var in range(4):
        for timestep in range(16):
            samples_scaled[:, timestep, var] = samples[:, timestep, var]*scaling_factors['a'][timestep, var] + scaling_factors['b'][timestep, var]

    if zoom:
        # use modes, skip for now
        modes = False
        if modes:
            # get rough region of interest, then zoom in on it afterwards!
            num_gradations = 5
            gradations = np.linspace(-1, 1, num_gradations)
            # for cutoff in the gradations, what fraction of samples (at a given time point) fall into that cutoff bracket?
            lower = 0
            real_grid = np.zeros(shape=(16, num_gradations, 4))
            for (i, cutoff) in enumerate(gradations):
                # take the mean over samples
                real_frac = ((real > lower) & (real <= cutoff)).mean(axis=0)
                lower = cutoff
                real_grid[:, i, :] = real_frac
            time_averaged_grid = np.mean(real_grid, axis=0)
            # get the most populated part of the grid for each variable
            grid_modes = np.argmax(time_averaged_grid, axis=0)
            lower = 0
            ranges = []
            for i in grid_modes:
                lower = gradations[i-1]
                upper = gradations[i]
                ranges.append([lower, upper])
        else:
            # hand-crafted ranges
            ranges = [[88, 100], [30, 130], [7, 60], [35, 135]]

    num_gradations = 25
    # for cutoff in the gradations, what fraction of samples (at a given time point) fall into that cutoff bracket?
    grid = np.zeros(shape=(16, num_gradations, 4))
    real_grid = np.zeros(shape=(16, num_gradations, 4))
    assert samples.shape[-1] == 4
    for var in range(4):
        # allow for a different range per variable (if zoom)
        low = ranges[var][0]
        high = ranges[var][1]
        gradations = np.linspace(low, high, num_gradations)
        for (i, cutoff) in enumerate(gradations):
            # take the mean over samples
            frac = ((samples_scaled[:, :, var] > low) & (samples_scaled[:, :, var] <= cutoff)).mean(axis=0)
            real_frac = ((real[:, :, var] > low) & (real[:, :, var] <= cutoff)).mean(axis=0)
            low = cutoff
            grid[:, i, var] = frac
            real_grid[:, i, var] = real_frac

    # now plot this as an image
    fig, axarr = plt.subplots(nrows=4, ncols=2, sharey='row', sharex=True)
    axarr[0, 0].imshow(grid[:, :, 0].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[1, 0].imshow(grid[:, :, 1].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[2, 0].imshow(grid[:, :, 2].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[3, 0].imshow(grid[:, :, 3].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[0, 1].imshow(real_grid[:, :, 0].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[1, 1].imshow(real_grid[:, :, 1].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[2, 1].imshow(real_grid[:, :, 2].T, origin='lower', aspect=0.5, cmap='magma_r')
    axarr[3, 1].imshow(real_grid[:, :, 3].T, origin='lower', aspect=0.5, cmap='magma_r')

    axarr[0, 0].set_title("synthetic")
    axarr[0, 1].set_title("real")
    for var in range(4):
        low, high = ranges[var]
        labels = np.linspace(low, high, num_gradations)[1::4]
        labels = np.round(labels, 0)
        axarr[var, 0].set_yticklabels(labels)
        axarr[var, 0].set_yticks(np.arange(num_gradations)[1::4])
        axarr[var, 0].set_ylabel(variables[var])
        for ax in axarr[var, :]:
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            ax.set_adjustable('box-forced')
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(b=True, color='black', alpha=0.2, linestyle='--')

    axarr[-1, 0].set_xticks(np.arange(16)[::2])
    axarr[-1, 1].set_xticks(np.arange(16)[::2])

    if zoom:
        plt.suptitle('(zoomed)')

    plt.tight_layout(pad=0.0, w_pad=-5.0, h_pad=0.1)
    plt.savefig("./experiments/eval/eICU_cristobal_marginals_r" + str(rep) + "_epoch" + str(epoch) + ".png")

    # now make the histograms
    fig, axarr = plt.subplots(nrows=1, ncols=4)
    axarr[0].set_ylabel("density")
    axarr[0].hist(real[:, :, 0].flatten(), normed=True, color='black', alpha=0.8, range=ranges[0], bins=min(50, (ranges[0][1] - ranges[0][0])), label='real')
    axarr[1].hist(real[:, :, 1].flatten(), normed=True, color='black', alpha=0.8, range=ranges[1], bins=50)
    axarr[2].hist(real[:, :, 2].flatten(), normed=True, color='black', alpha=0.8, range=ranges[2], bins=50)
    axarr[3].hist(real[:, :, 3].flatten(), normed=True, color='black', alpha=0.8, range=ranges[3], bins=50)
    axarr[0].hist(samples_scaled[:, :, 0].flatten(), normed=True, alpha=0.6, range=ranges[0], bins=min(50, (ranges[0][1] - ranges[0][0])), label='synthetic')
    axarr[0].legend()
    axarr[1].hist(samples_scaled[:, :, 1].flatten(), normed=True, alpha=0.6, range=ranges[1], bins=50)
    axarr[2].hist(samples_scaled[:, :, 2].flatten(), normed=True, alpha=0.6, range=ranges[2], bins=50)
    axarr[3].hist(samples_scaled[:, :, 3].flatten(), normed=True, alpha=0.6, range=ranges[3], bins=50)
    for (var, ax) in enumerate(axarr):
        ax.set_xlabel(variables[var])
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(b=True, color='black', alpha=0.2, linestyle='--')

    plt.gcf().subplots_adjust(bottom=0.2)
    fig.set_size_inches(10, 3)
    plt.savefig("./experiments/eval/eICU_cristobal_hist_r" + str(rep) + "_epoch" + str(epoch) + ".png")

    return True


# --- nips !!! --- #
def nips_plot_rbf(sample, index, which='train'):
    if which == 'train':
#        col = '#167ea0'
        col = '#13af5f'
    else:
        col = 'black'
    sample_length = len(sample)
    sample = sample.reshape(sample_length)
    x_points = np.arange(sample_length)
    fig, axarr = plt.subplots(1, 1, figsize=(2, 2))
    axarr.set_facecolor((0.95, 0.96, 0.96))
    axarr.plot(x_points, sample, color=col)
    axarr.set_ylim(-1.5, 1.5)
    axarr.get_xaxis().set_visible(False)
    axarr.get_yaxis().set_visible(False)
    axarr.spines["top"].set_visible(False)
    axarr.spines["bottom"].set_visible(False)
    axarr.spines["right"].set_visible(False)
    axarr.spines["left"].set_visible(False)
    axarr.tick_params(bottom='off', left='off')
    plt.savefig('./plots/NIPS_rbf_' + which + '_' + str(index) + '.png')
    plt.savefig('./plots/NIPS_rbf_' + which + '_' + str(index) + '.pdf')
    plt.clf()
    plt.close()
    return True

def nips_plot_sine(sample, index, which='train'):
    if which == 'train':
        #col = '#167ea0'
        #col = '#13af5f'
        col = '#1188ad'
    else:
        col = 'black'
    sample_length = len(sample)
    sample = sample.reshape(sample_length)
    sample_length = len(sample)
    sample = sample.reshape(sample_length)
    x_points = np.arange(sample_length)
    fig, axarr = plt.subplots(1, 1, figsize=(2, 2))
    axarr.set_facecolor((0.95, 0.96, 0.96))
    axarr.plot(x_points, sample, color=col)
    axarr.set_ylim(-1.1, 1.1)
    axarr.get_xaxis().set_visible(False)
    axarr.get_yaxis().set_visible(False)
    axarr.spines["top"].set_visible(False)
    axarr.spines["bottom"].set_visible(False)
    axarr.spines["right"].set_visible(False)
    axarr.spines["left"].set_visible(False)
    axarr.tick_params(bottom='off', left='off')
    plt.savefig('./plots/NIPS_sine_' + which + '_' + str(index) + '.png')
    plt.savefig('./plots/NIPS_sine_' + which + '_' + str(index) + '.pdf')
    plt.clf()
    plt.close()
    return True

def nips_plot_mnist(sample, index, which='train'):
    plt.axis('off')
    plt.imshow(sample, cmap=plt.cm.gray, interpolation='nearest')
    plt.savefig('./plots/NIPS_mnist_' + which + '_' + str(index) + '.png')
    plt.savefig('./plots/NIPS_mnist_' + which + '_' + str(index) + '.pdf')
    plt.clf()
    plt.close()
    return True
