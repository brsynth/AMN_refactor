import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from aMNModel import AMNModel
from tools import custom_loss


class AMRNNModel(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)


    def build_model(self):

        tf.random.set_seed(10)

        with CustomObjectScope({'RNNCell': RNNCell}):
            rnn = RNN(RNNCell(self.S, 
                              self.V2M,
                              self.P_in,
                              self.M2V_norm,
                              self.medium_bound,
                              self.hidden_dim))


        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((keras_input_dim))

        # hidden layer + back to the same size layer 
        # layer_1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        # layer_2 = tf.keras.layers.Dense(keras_input_dim, activation='relu')
        # z = layer_1(inputs)
        # y = layer_2(z)

        # one input layer
        layer_1 = tf.keras.layers.Dense(keras_input_dim, activation='relu')
        y = layer_1(inputs)


        # no layer
        # y = inputs



        # Add dimension by concatenate several copy of inputs data to use in
        # the RNNCell
        x = tf.expand_dims(y, axis =1)
        x_n = tf.concat([x for _ in range(self.timestep)], axis=1)

        V = rnn(x_n)
        # Inputs are used to compute the loss, to do that we return inputs in
        # the output
        outputs = tf.concat([V, inputs],1)

        # Compile
        model = keras.models.Model(inputs, outputs)
        model.compile(loss=custom_loss(self.S, self.P_out, self.P_in),
                      optimizer='adam',
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

        if self.epochs > 0:
            print('training epochs:', self.epochs)
            print('training regression:', self.regression)
            print('training batch size:', self.batch_size)
            print('training validation iter:', self.n_iter)
            print('training early stopping:', self.early_stopping)
    



class RNNCell(keras.layers.Layer):
    def __init__(self, S, V2M, P_in,M2V, medium_bound, hidden_dim, **kwargs):

        super(RNNCell, self).__init__(**kwargs)

        # Precise np array type since save and load can convert matrix to list
        # Could probably do better code by defining type in config methods
        self.S  = np.float32(S)
        self.V2M = np.float32(V2M)
        self.P_in =np.float32(P_in)
        self.M2V = np.float32(M2V)
        self.medium_bound = medium_bound
        self.hidden_dim = hidden_dim

        self.meta_dim = self.S.shape[0]
        self.flux_dim = self.S.shape[1]
        self.state_size = self.S.shape[1]
        self.input_size = self.P_in.shape[0]
    
    def build(self, input_shape):
        # weighs to compute V for both input (i) and recurrent cell (r)
        # if self.medium_bound == 'UB': # no kernel_Vh and kernel_Vi for EB
        self.wr_V = self.add_weight(shape=(self.flux_dim, self.meta_dim),
                                           name='kernel_Vr',
                                           trainable=True)

        self.br_V  = self.add_weight(shape=(self.flux_dim,),
                                            initializer='random_normal',
                                            name='bias_Vr',
                                            trainable=True)
        self.built = True


    def call(self, inputs, states):
        # At steady state we have :
        # M = V2M V and V = (M2V x W) M + V0

        V0 = tf.linalg.matmul(inputs, self.P_in) 
        # a = V0<0
        # a = tf.cast(a, tf.float32)
        # print(tf.keras.backend.sum(a))


        V = states[0]
        M = tf.linalg.matmul(V,tf.transpose(self.V2M))
        W = tf.math.multiply(self.M2V,self.wr_V)
        V = tf.linalg.matmul(M,tf.transpose(W))
        V = V + V0 + self.br_V
        return V, [V]
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            "S": self.S,
            "V2M" : self.V2M,
            "P_in" : self.P_in,
            "M2V" : self.M2V,
            "medium_bound" : self.medium_bound,
            "hidden_dim" : self.hidden_dim,
        }

        base_config.update(config)
        return base_config
    



