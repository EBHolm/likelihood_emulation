import tensorflow as tf 
import numpy as np 
import argparse
import os 
import pandas as pd 
import sys
from utils import load_data, symlog, normalize, normalize_inv, loglkl_norm, loglkl_norm_inv

"""
    Trains a neural network to emulate a likelihood function.
    How to use:
        python train.py -d DATA

    -> DATA: Should point to a folder with data files; the script will then scroll through all data with file endings .cl**, where
             ** can be either tt, te, tp, or so on
             -> These files are formatted as follows: (-loglkl, Cl1, Cl2, ...) and so on. The ell values corresponding to the Cls
                can be found in the .clell file. Only the ells computed internally by CLASS are used in order to compress data.

    The hyperparameters for the training process (e.g. optimizer, learning rate, batch sizes etc.) are set manually below.

    The resulting trained neural network will be stored in the /trained_networks/ subfolder. 

"""

# HYPERPARAMETERS FOR THE TRAINING PROCESS
network_name = 'planck_highl_tt_lite'

train_validation_test_split = (0.90, 0.05, 0.05)


batch_size = 500
epochs = 25

learning_rate = 0.001 #0.001 is default

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

loss_function = tf.keras.losses.MeanSquaredError

architecture = 'standard' # situated in the /architecture/ folder



# ----- AUTOMATIC FROM HERE ON -----
parser = argparse.ArgumentParser(description='For further argument specifications, do python train.py --help')
parser.add_argument('-d', metavar='DATA', type=str, help="The folder containing the data", required=True)
args = parser.parse_args()

# ----- DATA HANDLING -----
data_full, loglkl, data_indices, ell = load_data(args.d)
data_full, data_mean, data_std = normalize(data_full)
loglkl_normed, loglkl_mean, loglkl_sigma = loglkl_norm(loglkl)

data_full = tf.convert_to_tensor(data_full)

N_data = data_full.shape[0]
N_train, N_val = int(np.ceil(N_data*train_validation_test_split[0])), int(np.floor(N_data*train_validation_test_split[1]))
N_test = N_data - N_train - N_val
print(f"\nTraining with:\nN_train = {N_train}\nN_val = {N_val}\nN_test = {N_test}\n")

dataset = tf.data.Dataset.from_tensor_slices((data_full, loglkl_normed))
dataset = dataset.shuffle(buffer_size=N_data)
data_train = dataset.take(N_train).batch(batch_size)
data_val = dataset.skip(N_train).take(N_val).batch(batch_size)
data_test = dataset.skip(N_train + N_val).take(N_test).batch(batch_size)

# load architecture
if architecture == 'standard':
    from architecture import standard
    model = standard.Standard(N_data, data_indices)
elif architecture == 'simple':
    from architecture import simple
    model = simple.Simple(N_data, data_indices)

model.compile(optimizer=optimizer, loss=loss_function(), metrics=['MeanAbsoluteError'])
history = model.fit(data_train, validation_data=data_val, epochs=epochs)

test_loss = model.evaluate(data_test)

print(test_loss)
for inp, label in iter(data_test):
    print(loglkl_norm_inv(model.call(inp), loglkl_mean, loglkl_sigma), loglkl_norm_inv(label, loglkl_mean, loglkl_sigma))
    exit()

# save the model 
save_path = 'trained_networks'
if os.path.isdir(save_path):
    n = 0
    while os.path.isdir(f"{save_path}/{network_name}_{n}"):
        n += 1
    model.save(f"{save_path}/{network_name}_{n}")



















#############################################
# OLD CODE

"""
# OLD NORMALIZATION CODE 
# normalize each spectrum individually
N_spectra = data_full.shape[1]
for i in range(N_spectra):
    print(f"\n\n\nPreprocessing of spectrum {spectrum_types[i]}")
    print(f"BEFORE: \n{data_full[0, i, :]}")
    #data_full[:, i, :] = np.log(data_full[:, i, :])
    
    symlog_thresh = 1e-40
    data_full[:, i, :] = symlog(data_full[:, i, :], symlog_thresh)

    
    data_full[:, i, :] = (data_full[:, i, :] - np.mean(data_full[:, i, :]))/np.std(data_full[:, i, :])
    #print(np.mean(data_full[:, i, :]))
    print(f"AFTER: \n{data_full[0, i, :]}")


print(data_full.shape)
"""



# CODE FOR FLATTENED DATA
#cls = np.swapaxes(cls, 0, 1) # shape (N_data_points, N_spectra, N_ell)
#cls = np.reshape(cls, [cls.shape[0], cls.shape[1]*cls.shape[2]]) # shape (N_data_points, N_spectra*N_ell), ordered such that e.g. all clTT's come first

# split data into training, validation and training sets 
#N_data = data_full.shape[0]
#N_train, N_val = int(np.ceil(N_data*train_validation_test_split[0])), int(np.floor(N_data*train_validation_test_split[1]))
#N_test = N_data - N_train - N_val
#print(f"\nTraining with:\nN_train = {N_train}\nN_val = {N_val}\nN_test = {N_test}\n")

#data_train = tf.data.Dataset.from_tensor_slices((data_full[0:N_train, :], loglkl[0:N_train]))
#data_val   = tf.data.Dataset.from_tensor_slices((data_full[N_train:N_train + N_val, :], loglkl[N_train:N_train + N_val]))
#data_test  = tf.data.Dataset.from_tensor_slices((data_full[N_train + N_val:, :], loglkl[N_train + N_val:]))

# shuffle data into batches

#input_size = data_full.shape[1]


# data_full has shape (N_spectra, N_data_points, N_ell)
# change into shape (N_data_points, N_spectra, N_ell)
#print(data_full)

# CODE FOR INDIVIDUAL NORMALIZATINO OF SPECTRA
#N_spectra = data_full.shape[1]
#normalizers = []
#for i in range(N_spectra):
    #normalizer = tf.keras.layers.Normalization()
    #normalizer.adapt(data_full[:, i, 2:])
    #normalizer.adapt(data_full)
    #print(data_full[0, i, 2:])
    #print(normalizer(data_full[0, i, 2:]))
    #normalizers.append(normalizer)




# CODE FOR UNFLATTENED DATA
# data_full has shape (N_spectra, N_data_points, N_ell)
#data_full = tf.convert_to_tensor(np.array(cls)) 
# change into shape (N_data_points, N_spectra, N_ell)
#data_full = tf.transpose(data_full, [1, 0, 2])

#data_train = tf.data.Dataset.from_tensor_slices((data_full[0:N_train, :, :], loglkl[0:N_train]))
#data_val   = tf.data.Dataset.from_tensor_slices((data_full[N_train:N_train + N_val, :, :], loglkl[N_train:N_train + N_val]))
#data_test  = tf.data.Dataset.from_tensor_slices((data_full[N_train + N_val:, :, :], loglkl[N_train + N_val:]))

#input_size = data_full.shape[1]*data_full.shape[2]