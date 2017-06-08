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
import data_utils
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import copy
from scipy.stats import sem

print ("Starting TSTR experiment.")
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

# iterate over all dataset versions generated after running the GAN for 5 times

aurocs_all_runs = []
auprcs_all_runs = []
for oo in range(5):

	print (oo)

	# find the best "dataset epoch", meaning the GAN epoch that generated the dataset
	# validation is only done in some of the tasks, and the others are considered unknown
	# (use validation set to pick best GAN epoch, then get result on test set)

	vali_seqs_r = vali_seqs.reshape((vali_seqs.shape[0], -1))
	test_seqs_r = test_seqs.reshape((test_seqs.shape[0], -1))

	all_aurocs_exp = []
	all_auprcs_exp = []
	for nn in np.arange(50,1050,50):

		with open('./synthetic_eICU_datasets/samples_eICU_cdgan_synthetic_dataset_r' + str(oo) + '_' + str(nn) + '.pk', 'rb') as f:
		    synth_data = pickle.load(file=f)
		with open('./synthetic_eICU_datasets/labels_eICU_cdgan_synthetic_dataset_r' + str(oo) + '_' + str(nn) + '.pk', 'rb') as f:
		    synth_labels = pickle.load(file=f)

		train_seqs = synth_data
		train_targets = synth_labels
		train_seqs_r = train_seqs.reshape((train_seqs.shape[0], -1))

		all_aurocs = []
		all_auprcs = []

		# in case we want to train each random forest multiple times with each dataset
		for exp_num in range(1):
			accuracies = []
			precisions = []
			recalls = []
			aurocs = []
			auprcs = []
			for col_num in range(train_targets.shape[1]):
				estimator = RandomForestClassifier(n_estimators=100)
				estimator.fit(train_seqs_r, train_targets[:,col_num])
				accuracies.append(estimator.score(vali_seqs_r, vali_targets[:,col_num]))
				preds = estimator.predict(vali_seqs_r)
				precisions.append(precision_score(y_pred=preds, y_true=vali_targets[:,col_num]))
				recalls.append(recall_score(y_pred=preds, y_true=vali_targets[:,col_num]))
				preds = estimator.predict_proba(vali_seqs_r)
				fpr, tpr, thresholds = roc_curve(vali_targets[:,col_num], preds[:,1])
				aurocs.append(auc(fpr, tpr))
				precision, recall, thresholds = precision_recall_curve(vali_targets[:,col_num], preds[:,1])
				auprcs.append(auc(recall, precision))

			all_aurocs.append(aurocs)
			all_auprcs.append(auprcs)

		all_aurocs_exp.append(all_aurocs)
		all_auprcs_exp.append(all_auprcs)


	#with open('all_aurocs_exp_r' + str(oo) + '.pk', 'wb') as f:
	#	pickle.dump(file=f, obj=all_aurocs_exp)

	#with open('all_auprcs_exp_r' + str(oo) + '.pk', 'wb') as f:
	#	pickle.dump(file=f, obj=all_auprcs_exp)

	best_idx = np.argmax(np.array(all_aurocs_exp).sum(axis=1)[:,[0,2,4]].sum(axis=1) + np.array(all_auprcs_exp).sum(axis=1)[:,[0,2,4]].sum(axis=1))
	best = np.arange(50,1050,50)[best_idx]

	with open('./synthetic_eICU_datasets/samples_eICU_cdgan_synthetic_dataset_r' + str(oo) + '_' + str(best) + '.pk', 'rb') as f:
	    synth_data = pickle.load(file=f)
	with open('./synthetic_eICU_datasets/labels_eICU_cdgan_synthetic_dataset_r' + str(oo) + '_' + str(best) + '.pk', 'rb') as f:
	    synth_labels = pickle.load(file=f)


	train_seqs = synth_data
	train_targets = synth_labels
	train_seqs_r = train_seqs.reshape((train_seqs.shape[0], -1))

	accuracies = []
	precisions = []
	recalls = []
	aurocs = []
	auprcs = []
	for col_num in range(train_targets.shape[1]):
		estimator = RandomForestClassifier(n_estimators=100)
		estimator.fit(train_seqs_r, train_targets[:,col_num])
		accuracies.append(estimator.score(test_seqs_r, test_targets[:,col_num]))
		preds = estimator.predict(test_seqs_r)
		precisions.append(precision_score(y_pred=preds, y_true=test_targets[:,col_num]))
		recalls.append(recall_score(y_pred=preds, y_true=test_targets[:,col_num]))
		preds = estimator.predict_proba(test_seqs_r)
		fpr, tpr, thresholds = roc_curve(test_targets[:,col_num], preds[:,1])
		aurocs.append(auc(fpr, tpr))
		precision, recall, thresholds = precision_recall_curve(test_targets[:,col_num], preds[:,1])
		auprcs.append(auc(recall, precision))
	print(accuracies)
	print(precisions)
	print(recalls)
	print(aurocs)
	print(auprcs)
	print ("----------------------------")

	aurocs_all_runs.append(aurocs)
	auprcs_all_runs.append(auprcs)


allr = np.vstack(aurocs_all_runs)
allp = np.vstack(auprcs_all_runs)

tstr_aurocs_mean = allr.mean(axis=0)
tstr_aurocs_sem = sem(allr, axis=0)
tstr_auprcs_mean = allp.mean(axis=0)
tstr_auprcs_sem = sem(allp, axis=0)


# get AUROC/AUPRC for real, random data

print ("Experiment with real data.")
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

train_seqs_r = train_seqs.reshape((train_seqs.shape[0], -1))
vali_seqs_r = vali_seqs.reshape((vali_seqs.shape[0], -1))
test_seqs_r = test_seqs.reshape((test_seqs.shape[0], -1))

aurocs_all = []
auprcs_all = []
for i in range(5):
	accuracies = []
	precisions = []
	recalls = []
	aurocs = []
	auprcs = []
	for col_num in range(train_targets.shape[1]):
		estimator = RandomForestClassifier(n_estimators=100)
		estimator.fit(train_seqs_r, train_targets[:,col_num])
		accuracies.append(estimator.score(test_seqs_r, test_targets[:,col_num]))
		preds = estimator.predict(test_seqs_r)
		precisions.append(precision_score(y_pred=preds, y_true=test_targets[:,col_num]))
		recalls.append(recall_score(y_pred=preds, y_true=test_targets[:,col_num]))
		preds = estimator.predict_proba(test_seqs_r)
		fpr, tpr, thresholds = roc_curve(test_targets[:,col_num], preds[:,1])
		aurocs.append(auc(fpr, tpr))
		precision, recall, thresholds = precision_recall_curve(test_targets[:,col_num], preds[:,1])
		auprcs.append(auc(recall, precision))
	print(accuracies)
	print(precisions)
	print(recalls)
	print(aurocs)
	print(auprcs)
	aurocs_all.append(aurocs)
	auprcs_all.append(auprcs)

real_aurocs_mean = np.array(aurocs_all).mean(axis=0)
real_aurocs_sem = sem(aurocs_all, axis=0)
real_auprcs_mean = np.array(auprcs_all).mean(axis=0)
real_auprcs_sem = sem(auprcs_all, axis=0)


print ("Experiment with random predictions.")

#random score
test_targets_random = copy.deepcopy(test_targets)
random.shuffle(test_targets_random)
accuracies = []
precisions = []
recalls = []
aurocs = []
auprcs = []
for col_num in range(train_targets.shape[1]):
	accuracies.append(accuracy_score(y_pred=test_targets_random[:,col_num], y_true=test_targets[:,col_num]))
	precisions.append(precision_score(y_pred=test_targets_random[:,col_num], y_true=test_targets[:,col_num]))
	recalls.append(recall_score(y_pred=test_targets_random[:,col_num], y_true=test_targets[:,col_num]))
	preds = np.random.rand(len(test_targets[:,col_num]))
	fpr, tpr, thresholds = roc_curve(test_targets[:,col_num], preds)
	aurocs.append(auc(fpr, tpr))
	precision, recall, thresholds = precision_recall_curve(test_targets[:,col_num], preds)
	auprcs.append(auc(recall, precision))
print(accuracies)
print(precisions)
print(recalls)
print(aurocs)
print(auprcs)

random_aurocs = aurocs
random_auprcs = auprcs

print("Results")
print("------------")
print("------------")
print("TSTR")
print(tstr_aurocs_mean)
print(tstr_aurocs_sem)
print(tstr_auprcs_mean)
print(tstr_auprcs_sem)
print("------------")
print("Real")
print(real_aurocs_mean)
print(real_aurocs_sem)
print(real_auprcs_mean)
print(real_auprcs_sem)
print("------------")
print("Random")
print(random_aurocs)
print(random_auprcs)