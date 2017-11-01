# RGAN

This repository contains code for the paper, _[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)_, by Stephanie L. Hyland* ([@corcra](https://github.com/corcra)), Cristóbal Esteban* ([@cresteban](https://github.com/cresteban)), and Gunnar Rätsch ([@ratsch](https://github.com/ratsch)), from the Ratschlab, also known as the [Biomedical Informatics](http://bmi.inf.ethz.ch/) Group at ETH Zurich.

*Contributed equally, can't decide on name ordering

## Paper Overview

Idea: Use generative adversarial networks (GANs) to generate real-valued time series, for medical purposes. As the title suggests. 
The GAN is **R**GAN because it uses recurrent neural networks for both encoder and decoder (specifically LSTMs). 

#### What does this have to do with medicine?
We aim to generate time series from ICU patients, using the open-access [eICU dataset](http://eicu-crd.mit.edu/about/eicu/). However, we also generate some non-medical time-series, like sine waves and smooth functions sampled from Gaussian Processes, and MNIST digits (imagined as a time series).

#### Why generating data at all?
Sharing medical data is hard, because it comes from real people, and is naturally highly sensitive (not to mention legally protected). One workaround for this difficultly would be to create _sufficiently realistic_ synthetic data. This synthetic data could then be used to reproducibly develop and train machine learning models, enabling better science, and ultimately better models for medicine.

#### When is data 'sufficiently realistic'?
We claim in this paper, that synthetic data is useful when it can be used to train a model which can perform well on real data. So, we use the performance of a classifier _trained_ on the synthetic data, then _tested_ on real data as a measure of the quality of the data. We call this the "**TSTR** score". This is a way of evaluating the output of a GAN without relying on human perceptual judgements of individual samples.

#### Differential privacy

We also include the case where the GAN is trained in a differentially private manner, to provide stronger privacy guarantees for the training data. We mostly just use the differentially private SGD optimiser and the moments accountant from [here](https://github.com/tensorflow/models/tree/master/research/differential_privacy) (with some minor modifications).

## Code Quickstart

Primary dependencies: `tensorflow`, `scipy`, `numpy`, `pandas`

Note: This code is written in Python3!

Simplest route to running code (Linux/Mac):
```
git clone git@github.com:ratschlab/RGAN.git
cd RGAN
python experiment.py --settings_file test
```

Note: the `test` settings file is a dummy to demonstrate which options exist, and may not produce reasonable looking output.

## Expected Directory Structure

See the directories in this folder: https://github.com/ratschlab/RGAN/tree/master/experiments 

## Files in this Repository

The main script is `experiment.py` - this parses many options, loads and preprocesses data as needed, trains a model, and does evaluation. It does this by calling on some helper scripts:
- `data_utils.py`: utilities pertaining to data: generating toy data (e.g. sine waves, GP samples), loading MNIST and eICU data, doing test/train split, normalising data, generating synthetic data to use in TSTR experiments
- `model.py`: functions for defining ML models, i.e. the tensorflow meat, defines the generator and discriminator, the update steps, and functions for sampling from the model and 'inverting' points to find their latent-space representations
- `plotting.py`: visualisation scripts using matplotlib
- `mmd.py`: for maximum-mean discrepancy calculations, mostly taken from https://github.com/dougalsutherland/opt-mmd

Other scripts in the repo:
- `eICU_synthetic_dataset_generation.py`: essentially self-contained script for training the RCGAN to generate synthetic eICU data
- `eICU_task.py`: script to help identify a doable task in eICU, and generating the training data - feel free to experiment with different, harder tasks!
- `eICU_tstr_evaluation.py`: for running the TSTR evaluation using pre-generated synthetic dataset
- `eugenium_mmd.py`: code for doing MMD 3-sample tests, from https://github.com/eugenium/mmd
- `eval.py`: functions for evaluating the RGAN/generated data, like testing if the RGAN has memorised the training data, comparing two models, getting reconstruction errors, and generating data for visualistions of things like varying the latent dimensions, interpolating between input samples 
- `mod_core_rnn_cell_impl.py`: this is a modification of the same script from TensorFlow, modified to allow us to initialise the bias in the LSTM (required for saving/loading models)
- `kernel.py`: some playing around with kernels on time series
- `tf_ops.py`: required by `eugenium_mmd.py`

There are plenty of functions in many of these files that weren't used for the manuscript.

## Command line options

TODO

## Data sources

### MNIST

Get MNIST as CSVs here: https://pjreddie.com/projects/mnist-in-csv/

### eICU

eICU is access-restricted, and must be applied for. For more information: http://eicu-crd.mit.edu/about/eicu/

TODO: describe how we preprocess eICU/upload script for doing it
