# Code for CHIL 2021 paper
### Self-supervised transfer learning of physiological representations from free-living wearable data

This repository contains the necessary code to reproduce and investigate the experiments conducted for our paper. Unfortunately we cannot share the dataset due to strong privacy limitations that safeguard health data, however the pre-processing of the data is described thoroughly on the paper (section Evaluation). The main input is a chest ECG which measured heart rate and uniaxial acceleration in 15-second intervals along with the wrist device which recorded 60 Hz triaxial acceleration. 

Our pre-processing scripts align, window, and produce numpy vectors with all participants. Then, the models are trained on these vectors. Last, we extract embeddings from these models and do transfer learning with linear classifiers.

## Requirements
The code is written in python 3.6.0. The main libraries needed to execute our code are as follows:

 - tensorflow 1.4.0
 - keras 2.2.0
 - matplotlib 3.2.2
 - pandas 1.0.5
 - scikit-learn 0.23.1
 - numpy 1.19.0
 
You might also need some extra helper libraries like `tqdm` (prettier for-loops) but they are not mandatory.

## Data 
We use data from the Fenland Study (https://www.mrc-epid.cam.ac.uk/research/studies/fenland/). We cannot publicly share this data but it is available from the MRC Epidemiology Unit at the University of Cambridge upon reasonable request. To facilitate easier testing of our code, we provide small samples with the same vectors and naming conventions. See data/feature_names for the features and their order and data/fitness_test for the lab test data sample and the dictionary. Sensor windows from a randomly selected participant are provided in /data.

 
# Run
All experiments are streamlined and automated in bash files. The hyperparameter tuning was done in a high-performance computing SLURM cluster, and the contribution of the hyperparams was evaluated on the validation set. 

To train all the deep learning models (proposed and baselines) run:

    bash "run_experiments.sh"

To evaluate the feature-based pre-training baseline run:

    python3 "06_extract_features_for_xgb.py"
    python3 "07_xgboost_baseline_train_evaluate.py"

Last, to evaluate the embeddings with transfer learning run: 

    python3 "04_extract_embeddings.py"
    python3 "05_transfer_learning.py"

## Pre-trained models

We provide the best model and its weights [Step2Heart (A/R/T)] in /models/20200115-041958. This model can be used directly to extract embeddings with 04_extract_embeddings.py. 

## How to cite our paper 


