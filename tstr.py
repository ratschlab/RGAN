#!/usr/bin/env ipython
# Run TSTR on a trained model.

import model

# grab labels (from eICU, mnist)

# generate synthetic data
synth_samples = model.sample_trained_model(settings_file, epoch, num_samples,
        C_samples=train_labels)

# grab test/vali data (from eICU, mnist)

# train model on synth data

# test model on test data
