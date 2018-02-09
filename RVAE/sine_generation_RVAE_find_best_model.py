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

experiment_id = './experiments_test_RVAE_sine_SRNN_new_RVAE_HS_short_callable'
 
########################
# SELECT RVAE WITH LOWER COST
########################

# Find the model that performed best in the validation set
experiments_directory = "./" + experiment_id + "/"

files = glob.glob(experiments_directory + '/*.json')
rows = []
for fi in files:
    with open(fi, 'r') as f:
        r = json.load(f)
        r['final_train_cost'] = r['costs_train'][-1]
        r['final_validation_cost'] = r['costs_val'][-1]
        r['final_test_cost'] = r['costs_test'][-1]
        r['filename'] = fi
    rows.append(r)
    
df = pd.DataFrame(rows)
to_delete = [col for col in df.columns if ('costs' in col) or ('other' in col)]
df.drop(to_delete, axis=1, inplace=True)
best_model_filename = df.ix[df.sort_values(by='final_validation_cost').index[0]]["filename"]
print(best_model_filename.replace(".json", ""))
print("learning_rate, delta_error, optimizer_str, hidden_units_dec, hidden_units_enc, emb_dim, mult")
print(df.ix[df.sort_values(by='final_validation_cost').index[0]]["config"])