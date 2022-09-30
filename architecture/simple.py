import tensorflow as tf

class Simple(tf.keras.Model):
    def __init__(self, input_norm_params, output_norm_params):
        super(Simple, self).__init__()

        self.mean, self.sigma = input_norm_params
        self.loglkl_mean, self.loglkl_sigma = output_norm_params

        self.num_individual_layers = 4 # one for each spectrum, applied individually
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
        x = self.normalize_input(x)
        for idx, layer in enumerate(self.individual_layers):
            x = layer(x)
        x = self.output_layer(x)
        x = self.normalize_output_inverse(x)        
        return x

    def normalize_input(self, x):
        return tf.divide(tf.add(x, -self.mean), self.sigma)

    def normalize_output_inverse(self, x):
        return self.loglkl_sigma*x + self.loglkl_mean