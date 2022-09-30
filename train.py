import tensorflow as tf 
import numpy as np
import argparse
import os 
from utils import load_data

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


batch_size = 1000
epochs = 15

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
data_full = tf.convert_to_tensor(data_full)

input_norm = [tf.cast(tf.math.reduce_mean(data_full, axis=0), tf.float32), tf.cast(tf.math.reduce_std(data_full, axis=0), tf.float32)]
output_norm = [np.mean(loglkl), np.std(loglkl)]

N_data = data_full.shape[0]
N_train, N_val = int(np.ceil(N_data*train_validation_test_split[0])), int(np.floor(N_data*train_validation_test_split[1]))
N_test = N_data - N_train - N_val
print(f"\nTraining with:\nN_train = {N_train}\nN_val = {N_val}\nN_test = {N_test}\n")

dataset = tf.data.Dataset.from_tensor_slices((data_full, loglkl))
dataset = dataset.shuffle(buffer_size=N_data)
data_train = dataset.take(N_train).batch(batch_size)
data_val = dataset.skip(N_train).take(N_val).batch(batch_size)
data_test = dataset.skip(N_train + N_val).take(N_test).batch(batch_size)

# load architecture
if architecture == 'standard':
    from architecture import standard
    model = standard.Standard(data_indices, input_norm, output_norm)
elif architecture == 'simple':
    from architecture import simple
    model = simple.Simple(data_indices, input_norm, output_norm)

model.compile(optimizer=optimizer, loss=loss_function(), metrics=['MeanAbsoluteError'])
history = model.fit(data_train, validation_data=data_val, epochs=epochs)

test_loss = model.evaluate(data_test)

print(test_loss)
for inp, label in iter(data_test):
    print(model.call(inp), label)

# save the model 
save_path = 'trained_networks'
if os.path.isdir(save_path):
    n = 0
    while os.path.isdir(f"{save_path}/{network_name}_{n}"):
        n += 1
    model.save(f"{save_path}/{network_name}_{n}")
