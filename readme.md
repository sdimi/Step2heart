# _Step2Heart_ 🏃‍♂️🤍 — Generalizable physiological representations
![header image](https://github.com/sdimi/Step2heart/blob/main/data/architecture_overview.png)


### 📖 Self-supervised transfer learning of physiological representations from free-living wearable data

<details><summary>Abstract (click to expand)</summary>
<p>

Wearable devices such as smartwatches are becoming increasingly popular tools for objectively monitoring physical activity in free-living conditions. To date, research has primarily focused on the purely supervised task of human activity recognition, demonstrating limited success in inferring high-level health outcomes from low-level signals. Here, we present a novel self-supervised representation learning method using activity and heart rate (HR) signals without semantic labels. With a deep neural network, we set HR responses as the supervisory signal for the activity data, leveraging their underlying physiological relationship. In addition, we propose a custom quantile loss function that accounts for the long-tailed HR distribution present in the general population.

We evaluate our model in the largest free-living combined-sensing dataset (comprising >280k hours of wrist accelerometer & wearable ECG data). Our contributions are two-fold: i) the pre-training task creates a model that can accurately forecast HR based only on cheap activity sensors, and ii) we leverage the information captured through this task by proposing a simple method to aggregate the learnt latent representations (embeddings) from the window-level to user-level. Notably, we show that the embeddings can generalize in various downstream tasks through transfer learning with linear classifiers, capturing physiologically meaningful, personalized information. For instance, they can be used to predict variables associated with individuals’ health, fitness and demographic characteristics (AUC >70), outperforming unsupervised autoencoders and common bio-markers. Overall, we propose the first multimodal self-supervised method for behavioral and physiological data with implications for large-scale health and lifestyle monitoring.

</p>
</details>

**This repository**. We provide the necessary code to reproduce the experiments of our paper [1]. Unfortunately we cannot share the entire dataset due to privacy limitations that safeguard health data, however we provide some samples (see below). The main pre-training input is a wrist device which recorded 60 Hz triaxial acceleration and the output is a chest ECG wearable device which measured heart rate in 15-second intervals. For the downstream tasks, the most important outcome is VO2max (cardiorespiratory fitness), measured with a treadmil test. We have pre-processed the data by aligning and windowing all sensors in order to produce numpy vectors. Then, we train the proposed models on these vectors and extract embeddings for transfer learning with linear classifiers.

## 🛠️ Requirements
The code is written in python 3.6.0. The main libraries needed to execute our code are as follows:

 - tensorflow 1.4.0
 - keras 2.2.0
 - matplotlib 3.2.2
 - pandas 1.0.5
 - scikit-learn 0.23.1
 - numpy 1.19.0
 
You might also need some extra helper libraries like `tqdm` (prettier for-loops) but they are not mandatory.

## 🗂️ Data 
We use data from the [Fenland Study](https://www.mrc-epid.cam.ac.uk/research/studies/fenland/). We cannot publicly share this data but it is available from the MRC Epidemiology Unit at the University of Cambridge upon reasonable request. To facilitate easier testing of our code, we provide small samples with the same vectors and naming conventions. See ``data/feature_names`` for the features and their order and ``data/fitness_test`` for the laboratory treadmill data sample and the data dictionary. Sensor windows from a randomly selected participant are provided in ``/data``.

The input vector of activity for the pre-training task is a 3D tensor of dimensions [samples, timesteps, features] while the output heart rate is an 1D vector of [samples]. In particular, in ``/data`` we provide X = ``[1, 512, 34]`` and y = ``[1]``. Essentially, every 512-long input window corresponds to a single future heart rate. The outcomes for transfer learning in ``data/fitness_test`` are in an 2D vector of [1, features].

 
# ▶️ Run
All experiments are streamlined in bash files. The hyperparameter tuning was done in a SLURM cluster, and the contribution of the hyperparams was evaluated on the validation set. 

To train all pre-training neural networks (proposed and baselines), run:

    bash "run_experiments.sh"

To extract embeddings and perform transfer learning, run: 

    python3 "04_extract_embeddings.py"
    python3 "05_transfer_learning.py"

Last, to evaluate the feature-based pre-training baseline, run:

    python3 "06_extract_features_for_xgb.py"
    python3 "07_xgboost_baseline_train_evaluate.py"


## Pre-trained models

We provide the best model and its weights (called _Step2Heart<sub>A/R/T</sub>_ in the paper) in the folder ``/models/20200115-105719``. This model can be used directly to extract embeddings with ``04_extract_embeddings.py`` and subsequently perform transfer learning. 

## How to cite our paper 

Please consider citing our papers if you use code or ideas from this project:

> [1]  Dimitris Spathis, Ignacio Perez-Pozuelo, Soren Brage, Nicholas J. Wareham, Cecilia Mascolo. ["Self-supervised transfer learning of physiological representations from free-living wearable data."](https://dl.acm.org/doi/10.1145/3450439.3451863) In Proceedings of ACM Conference on Health, Inference, and Learning (CHIL), USA, 2021. (to appear)

> [2] Dimitris Spathis, Ignacio Perez-Pozuelo, Soren Brage, Nicholas J. Wareham, Cecilia Mascolo. ["Learning Generalizable Physiological Representations from Large-scale Wearable Data."](https://arxiv.org/pdf/2011.04601.pdf) In NeurIPS Machine Learning for Mobile Health workshop, Vancouver, Canada, 2020.

## License

This code is licensed under the terms and conditions of GPLv3 unless otherwise stated. The actual paper is governed by a separate license and the paper authors retain their respective copyrights.



