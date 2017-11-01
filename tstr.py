#!/usr/bin/env ipython
# Run TSTR on a trained model. (helper script)

import sys
import glob
import numpy as np
import pdb

from eval import TSTR_mnist, TSTR_eICU

assert len(sys.argv) >= 2
identifier = sys.argv[1]
print(identifier)

model = sys.argv[2]
if model == 'CNN':
    CNN = True
    print('Using CNN')
else:
    CNN = False
    print('Using RF')

task = sys.argv[3]
if task == 'mnist':
    mnist = True
    print('testing on mnist')
else:
    mnist = False
    print('testing on eicu')

params_dir = 'REDACTED'
params = glob.glob(params_dir + identifier + '_*.npy')
print(params)
epochs = [int(p.split('_')[-1].strip('.npy')) for p in params]

# (I write F1 here but we're not actually reporting the F1, sorry :/)
epoch_f1 = np.zeros(len(epochs))
print('Running TSTR on validation set across all epochs for which parameters are available')
for (i, e) in enumerate(epochs):
    if mnist:
        synth_f1, real_f1 = TSTR_mnist(identifier, e, generate=True, vali=True, CNN=CNN)
    else:
        print('testing eicu')
        synth_f1 = TSTR_eICU(identifier, e, generate=True, vali=True, CNN=CNN)
    epoch_f1[i] = synth_f1

best_epoch_index = np.argmax(epoch_f1)
best_epoch = epochs[best_epoch_index]

print('Running TSTR on', identifier, 'at epoch', best_epoch, '(validation f1 was', epoch_f1[best_epoch_index], ')')
if mnist:
    TSTR_mnist(identifier, best_epoch, generate=True, vali=False, CNN=CNN)
    # also run TRTS at that epoch
    TSTR_mnist(identifier, best_epoch, generate=True, vali=False, CNN=CNN, reverse=True)
else:
    TSTR_eICU(identifier, best_epoch, generate=True, vali=False, CNN=CNN)
    # also run TRTS at that epoch
    TSTR_eICU(identifier, best_epoch, generate=True, vali=False, CNN=CNN, reverse=True)
