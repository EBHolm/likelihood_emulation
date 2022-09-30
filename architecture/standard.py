import tensorflow as tf
import numpy as np 

class Standard(tf.keras.Model):
    def __init__(self, N_data, data_indices):
        super(Standard, self).__init__()
        
        #   Standard, fully-connected network

        self.data_indices = data_indices
        self.num_spectra = len(data_indices.keys()) - 1
        self.num_spectrum_layers = 3 # one for each spectrum, applied individually
        self.num_nuisance_layers = 3
        self.num_global_layers = 3

        self.num_spectrum_nodes = 500
        self.num_nuisance_nodes = 100
        self.num_global_nodes   = 500

        self.spectrum_layers, self.nuisance_layers, self.global_layers = [], [], []
        for idx_spectrum in range(self.num_spectra):
            layers = []
            for idx_layer in range(self.num_spectrum_layers):
                layers.append(tf.keras.layers.Dense(self.num_spectrum_nodes, 
                                                    activation=tf.keras.activations.relu))
            self.spectrum_layers.append(layers)
        for idx_layer in range(self.num_nuisance_layers):
            self.nuisance_layers.append(tf.keras.layers.Dense(self.num_nuisance_nodes, 
                                                              activation=tf.keras.activations.relu))
        for idx_layer in range(self.num_global_layers):
            self.global_layers.append(tf.keras.layers.Dense(self.num_global_nodes, 
                                                              activation=tf.keras.activations.relu))
        
        self.input_layer = tf.keras.layers.InputLayer()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, **kwargs):
        x = self.input_layer(x)
        
        split_indices = [val for key, val in self.data_indices.items() if val != 0]
        split_indices.append(x.shape[1] - max(split_indices))
        spectra = tf.split(x, split_indices, axis=1)

        for idx_spectrum in range(self.num_spectra):
            for spectrum_layer in self.spectrum_layers[idx_spectrum]:
                spectra[idx_spectrum] = spectrum_layer(spectra[idx_spectrum])

        for nuisance_layer in self.nuisance_layers:
            spectra[-1] = nuisance_layer(spectra[-1])

        x = tf.concat(spectra, axis=1)

        for layer in self.global_layers:
            x = layer(x)
            
        x = self.output_layer(x)                
        return x