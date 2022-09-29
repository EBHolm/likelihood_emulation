import tensorflow as tf
import numpy as np 

class Standard(tf.keras.Model):
    def __init__(self, N_spectra, spectrum_size):
        super(Standard, self).__init__()
        
        #   Standard, fully-connected network

        self.num_individual_layers = 7 # one for each spectrum, applied individually
        self.N_individual_nodes = spectrum_size*3

        self.N_spectra = N_spectra
        self.individual_layers = []
        for idx_spectra in range(self.N_spectra):
            layers = []
            for idx_layer in range(self.num_individual_layers):
                layers.append(tf.keras.layers.Dense(self.N_individual_nodes, activation=tf.keras.activations.relu))
            self.individual_layers.append(layers)
        self.input_layer = tf.keras.layers.InputLayer()
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x, **kwargs):
        x = self.input_layer(x)
        print(spectra)
        spectra = tf.split(x, num_or_size_splits=1, axis=0)
        print(spectra)
        # x = tf.split(x, num_or_size_splits=N_ell)
        for idx_spectra, spectrum in enumerate(spectra):
            #new_spectrum = x[idx_spectra]
            #print(new_spectrum)
            for idx_layer in range(self.num_individual_layers):
             #   new_spectrum = self.individual_layers[idx_spectra][idx_layer](new_spectrum)
                spectrum = self.individual_layers[idx_spectra][idx_layer](spectrum)
            #spectra.append(new_spectrum)
        x = tf.concat([spectrum for spectrum in spectra], axis=0)
        print(x)
        x = self.output_layer(x)
        return x
