import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from amn.model.aMNModel import AMNModel
from amn.tools import custom_loss


class AMNWtModel(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)


    def build_model(self):

        tf.random.set_seed(10)

        with CustomObjectScope({'RNNCell': RNNCell}):
            rnn = RNN(RNNCell(S=self.S, 
                              V2M=self.V2M, 
                              P_uptake=self.P_uptake,
                              M2V=self.M2V_norm, 
                              medium_bound=self.medium_bound, 
                              hidden_dim=self.hidden_dim,
                              input_size=self.X.shape[1]
                              ))


        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((keras_input_dim))

        # Add dimension by concatenate several copy of inputs data to use in
        # the RNNCell
        x = tf.expand_dims(inputs, axis =1)
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

    



class RNNCell(keras.layers.Layer):
    def __init__(self, S, V2M, P_uptake, M2V, medium_bound, hidden_dim, input_size, **kwargs):

        super(RNNCell, self).__init__(**kwargs)

        # Precise np array type since save and load can convert matrix to list
        # Could probably do better code by defining type in config methods
        self.S  = np.float32(S)
        self.V2M = np.float32(V2M)
        self.P_uptake =np.float32(P_uptake)
        self.M2V = np.float32(M2V)
        self.medium_bound = medium_bound
        self.hidden_dim = hidden_dim

        self.meta_dim = self.S.shape[0]
        self.flux_dim = self.S.shape[1]
        self.state_size = self.S.shape[1]
        self.input_size = input_size

    
    def build(self, input_shape):

        uptake_size = self.P_uptake.shape[0]
        # weighs to compute V for both input (i) and recurrent cell (r)
        if self.medium_bound == 'UB': # no kernel_Vh and kernel_Vi for EB
            if self.hidden_dim > 0: # plug an hidden layer upstream of Winput
                self.wh_V = self.add_weight(shape=(self.input_size,self.hidden_dim), 
                                            name='kernel_Vh', 
                                            trainable=True)
                self.wi_V = self.add_weight(shape=(self.hidden_dim, uptake_size), 
                                            name='kernel_Vi',
                                            trainable=True)


                
            else:
                self.wi_V = self.add_weight(shape=(self.input_size, self.input_size), 
                                            name='kernel_Vi',
                                            trainable=True)

            self.bi_V  = self.add_weight(shape=(uptake_size,),
                                        initializer='random_normal',
                                        name='bias_Vi',
                                        trainable=True)
        

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
        if self.medium_bound == 'UB':
            if self.hidden_dim > 0:
                VH = keras.backend.dot(inputs, self.wh_V)
                V0 = keras.backend.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = keras.backend.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs # EB case

        V0 = tf.linalg.matmul(V0, self.P_uptake) 
    
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
            "P_uptake" : self.P_uptake,
            "M2V" : self.M2V,
            "medium_bound" : self.medium_bound,
            "hidden_dim" : self.hidden_dim,
            "input_size" : self.input_size,
        }

        base_config.update(config)
        return base_config
    



