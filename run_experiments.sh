#Author: Dimitris Spathis (ds806@cl.cam.ac.uk)

#pre-process vectors for deep learning
python3 01_data_preprocessing.py
python3 02_data_normalization.py

#train the networks 3 times each 

#modality numbers correspond to Step2Heart (A), (A/R), (A/T),  (A/R/T), Convolutional Autoencoder
#(m=modality [1,2,3,4,5] check 03_training.py and utils.py for details)
#(l=loss [mse, quantile] check 03_training.py and utils.py for details)
python3 03_training.py -m 1 -l "quantile"
python3 03_training.py -m 1 -l "quantile"
python3 03_training.py -m 1 -l "quantile"

python3 03_training.py -m 1 -l "mse"
python3 03_training.py -m 1 -l "mse"
python3 03_training.py -m 1 -l "mse"

python3 03_training.py -m 2 -l "quantile"
python3 03_training.py -m 2 -l "quantile"
python3 03_training.py -m 2 -l "quantile"

python3 03_training.py -m 2 -l "mse"
python3 03_training.py -m 2 -l "mse"
python3 03_training.py -m 2 -l "mse"

python3 03_training.py -m 3 -l "quantile"
python3 03_training.py -m 3 -l "quantile"
python3 03_training.py -m 3 -l "quantile"

python3 03_training.py -m 3 -l "mse"
python3 03_training.py -m 3 -l "mse"
python3 03_training.py -m 3 -l "mse"

python3 03_training.py -m 4 -l "quantile"
python3 03_training.py -m 4 -l "quantile"
python3 03_training.py -m 4 -l "quantile"

python3 03_training.py -m 4 -l "mse"
python3 03_training.py -m 4 -l "mse"
python3 03_training.py -m 4 -l "mse"

#autoencoder baseline (no quantile here, only MSE)
python3 03_training.py -m 5 -l "mse"
python3 03_training.py -m 5 -l "mse"
python3 03_training.py -m 5 -l "mse"
