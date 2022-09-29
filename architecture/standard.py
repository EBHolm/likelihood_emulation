import tensorflow as tf
import numpy as np 

class Standard(tf.keras.Model):
    def __init__(self, N_spectra, spectrum_size):
        super(Standard, self).__init__()
        
        #   Standard, fully-connected network

        self.num_individual_layers = 4 # one for each spectrum, applied individually
        self.N_individual_nodes = spectrum_size*4

        self.N_spectra = N_spectra
        self.individual_layers = []
        for idx_layer in range(self.num_individual_layers):
            self.individual_layers.append(tf.keras.layers.Dense(self.N_individual_nodes, 
                                                    activation=tf.keras.activations.relu))
        #for idx_spectra in range(self.N_spectra):
        #    layers = []
        #    for idx_layer in range(self.num_individual_layers):
        #        layers.append(tf.keras.layers.Dense(self.N_individual_nodes, 
        #                                            activation=tf.keras.activations.relu))
                                                    #activation=tf.keras.activations.sigmoid))
        #    self.individual_layers.append(layers)
        self.input_layer = tf.keras.layers.InputLayer()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, **kwargs):
        #print("PRINTING")
        #print(x)
        x = self.input_layer(x)
        #print(x)
        #x = x*0.0 + 1.0
        #tf.print(x)
        #tf.print(x)
        #x = self.test_layer(x)
        #tf.print(x)
        #spectra = tf.split(x, num_or_size_splits=1, axis=0)
        #print(spectra)
        # x = tf.split(x, num_or_size_splits=N_ell)
        #for idx_spectra, spectrum in enumerate(spectra):
            #new_spectrum = x[idx_spectra]
            #print(new_spectrum)
            #for idx_layer in range(self.num_individual_layers):
             #   new_spectrum = self.individual_layers[idx_spectra][idx_layer](new_spectrum)
         #       print(idx_spectra, idx_layer)
          #      spectrum = self.individual_layers[idx_spectra][idx_layer](spectrum)
             #   print("doing nothing")
            #spectra.append(new_spectrum)
        #tf.print("PRINTING")
        #tf.print(x)
        for idx, layer in enumerate(self.individual_layers):
            #if idx == yooo:
                
            x = layer(x)
            #if idx == yooo:
        #tf.print(x)
        #x = tf.concat([spectrum for spectrum in spectra], axis=0)
        #tf.print(x)
        #print(x)
        
        x = self.output_layer(x)
        
        return x
