import tensorflow as tf
import numpy as np 

class Standard(tf.keras.Model):
    def __init__(self, N_data, data_indices):
        super(Standard, self).__init__()
        
        #   Standard, fully-connected network

        self.num_individual_layers = 4 # one for each spectrum, applied individually
        #self.N_individual_nodes = int(np.ceil(N_data/len(data_indices.keys())))
        self.N_individual_nodes = 500
        print(f"Architecture has {self.num_individual_layers} layers of {self.N_individual_nodes} nodes.")

        self.individual_layers = []
        for idx_layer in range(self.num_individual_layers):
            self.individual_layers.append(tf.keras.layers.Dense(self.N_individual_nodes, 
                                                    activation=tf.keras.activations.relu))
        
        self.input_layer = tf.keras.layers.InputLayer()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, **kwargs):
        x = self.input_layer(x)
        for idx, layer in enumerate(self.individual_layers):
            x = layer(x)
        x = self.output_layer(x)
                
        return x