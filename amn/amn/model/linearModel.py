import keras
from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from amn.model.aMNModel import AMNModel
from amn.tools import custom_loss


class LinearModel(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)


    def build_model(self):

        tf.random.set_seed(10)

        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((keras_input_dim,))

        layer = tf.keras.layers.Dense(self.S.shape[1])
        V = layer(inputs)

        # Inputs are used to compute the loss, to do that we return inputs in
        # the output
        outputs = tf.keras.layers.Concatenate(1)([V, inputs])

        # Compile
        model = keras.models.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=custom_loss(self.S, self.P_out, self.P_in),
                      optimizer=opt,
                      metrics=[custom_loss(self.S, self.P_out, self.P_in)],
                      run_eagerly=False)
        return model


    def printout_by_type(self):
        print('dataset file:', self.dataset_file)
        print('model type:', "AMNWt")
        print('model medium bound:', self.medium_bound)
        print('timestep:', self.timestep)
        print('training set size', self.X.shape, self.Y.shape)

        if self.n_hidden > 0:
            print('nbr hidden layer:', self.n_hidden)
            print('hidden layer size:', self.hidden_dim)
            print('activation function:', self.activation)


    



