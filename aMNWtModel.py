import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from aMNModel import AMNModel
from loss import SV_loss, V_in_loss, V_pos_loss
from tools import my_mse


class AMNWtModel(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)
    

    def output_AMNWt(self, V, Vin, verbose=False):
        """
        This method return a concatenation of different type of information.
        First, it returns the predicted reference fluxes, P_outV. Then it
        returns the loss computed for the SV, P_inV and V_pos. Then it returns
        the prediction on all fluxes plus some loss.
        All theses information are given to easily construct the loss on the
        model.
        """
        P_out     = tf.convert_to_tensor(np.float32(self.P_out))
        P_outV    = tf.linalg.matmul(V, tf.transpose(P_out))
        SV        = SV_loss(V, self.S) # SV const
        P_inV = V_in_loss(V, self.P_in, Vin, self.medium_bound) # P_in const
        V_pos = V_pos_loss(V) # V ≥ 0 const
        outputs = concatenate([P_outV, SV, P_inV, V_pos, V], axis=1)
        if verbose:
            print('AMN output shapes for P_outV, SV, P_inV, Vpos, V, outputs', \
                  P_outV.shape, SV.shape, P_inV.shape, V_pos.shape,\
                  V.shape, outputs.shape)
        return outputs
    
    

    def build_model(self):
        """
        Build and return an AM
        N using an RNN cell
        input : medium vector in parameter
        # output: experimental steady state fluxes
        """

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

        # Add dimension by concatenate several copy of inputs data to use in
        # the RNNCell
        x = tf.expand_dims(inputs, axis =1)
        x_n = tf.concat([x for _ in range(self.timestep)], axis=1)

        V = rnn(x_n)
        outputs = tf.concat([self.output_AMNWt(V, inputs),inputs],1)

        # Compile
        model = keras.models.Model(inputs, outputs)
        model.compile(loss=my_mse,optimizer='adam',metrics=[my_mse])
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
        if self.medium_bound == 'UB': # no kernel_Vh and kernel_Vi for EB
            if self.hidden_dim > 0: # plug an hidden layer upstream of Winput
                self.wh_V = self.add_weight(shape=(self.input_size,self.hidden_dim), 
                                            name='kernel_Vh', 
                                            trainable=True)
                self.wi_V = self.add_weight(shape=(self.hidden_dim, self.input_size), 
                                            name='kernel_Vi',
                                            trainable=True)
            else:
                self.wi_V = self.add_weight(shape=(self.input_size, self.input_size), 
                                            name='kernel_Vi',
                                            trainable=True)

        self.wr_V = self.add_weight(shape=(self.flux_dim, self.meta_dim),
                                           name='kernel_Vr',
                                           trainable=True)
        self.bi_V  = self.add_weight(shape=(self.input_size,),
                                            initializer='random_normal',
                                            name='bias_Vi',
                                            trainable=True)
        self.br_V  = self.add_weight(shape=(self.flux_dim,),
                                            initializer='random_normal',
                                            name='bias_Vr',
                                            trainable=True)
        self.built = True

    
    def call(self, inputs, states):
        # At steady state we have :
        # M = V2M V and V = (M2V x W) M + V0
        V = states[0]
        if self.medium_bound == 'UB':
            if self.hidden_dim > 0:
                VH = keras.backend.dot(inputs, self.wh_V)
                V0 = keras.backend.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = keras.backend.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs # EB case
            
        V0 = tf.linalg.matmul(V0, self.P_in) ## here !
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
    



