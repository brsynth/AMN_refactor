import os
import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from aMNModel import AMNModel
from loss import SV_loss, V_in_loss, V_pos_loss

class AMNWtModel(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)


    def model_input_by_type(self, X, Y):
        """
        We copy several time the dataset X to give to the RNN model a 
        sequential dataset. The number of copy is given by the timestep attribute.
        The shape of X is then transform from (a,b) to (a,timestep,b).
        """
        X_new = np.zeros((len(X), self.timestep, X.shape[1]))
        for i in range(len(X)):
            for j in range(self.timestep):
                X_new[i][j] = X[i]

        return X_new, Y


    def set_model(self, verbose=False):
        """
        Build and return an AMN using an RNN cell
        input : medium vector in parameter
        # output: experimental steady state fluxes
        """
        def Wt_layers(inputs, parameter, verbose=False):
            # Build and return AMN layers using an RNN cell
            with CustomObjectScope({'RNNCell': RNNCell}):
                rnn = RNN(RNNCell(parameter))
            V = rnn(inputs)
            Vin = inputs[:,0,:]
            return self.output_AMNWt(V, Vin,  verbose=verbose)


        def mse_(y_true, y_pred):
            # Custom loss function
            end = y_true.shape[1]
            return keras.losses.mean_squared_error(y_true, y_pred[:,:end])


        # Model
        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((None,keras_input_dim))
        outputs = Wt_layers(inputs,self)

        # Compile
        model = keras.models.Model(inputs, outputs)
        model.compile(loss=mse_,optimizer='adam',metrics=[mse_])

        if verbose == 2: 
            print(model.summary())

        self.model = model


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
        P_outV    = tf.linalg.matmul(V, tf.transpose(P_out), b_is_sparse=True)
        SV        = SV_loss(V, self.S) # SV const
        P_inV = V_in_loss(V, self.P_in, Vin, self.medium_bound) # P_in const
        V_pos = V_pos_loss(V) # V ≥ 0 const
        outputs = concatenate([P_outV, SV, P_inV, V_pos, V], axis=1)
        if verbose:
            print('AMN output shapes for P_outV, SV, P_inV, Vpos, V, outputs', \
                  P_outV.shape, SV.shape, P_inV.shape, V_pos.shape,\
                  V.shape, outputs.shape)
        return outputs

    def get_V_in(self,x):
        return x[:,0,:]
    
    def nb_columns_pred(self):
        return self.Y.shape[1] + 3 + self.S.shape[1] 
    
    
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
            print('training droP_out:', self.droP_out)
            print('training batch size:', self.batch_size)
            print('training validation iter:', self.n_iter)
            print('training xfold:', self.xfold)
            print('training early stopping:', self.early_stopping)
    

    def build_model(self):
        """
        Build and return an AMN using an RNN cell
        input : medium vector in parameter
        # output: experimental steady state fluxes
        """

        seed = 10
        np.random.seed(seed=seed)  
        tf.random.set_seed(seed)

        def Wt_layers(inputs, parameter, verbose=False):
            # Build and return AMN layers using an RNN cell
            with CustomObjectScope({'RNNCell': RNNCell}):
                rnn = RNN(RNNCell(parameter))
            V = rnn(inputs)
            Vin = inputs[:,0,:]
            return self.output_AMNWt(V, Vin,  verbose=verbose)


        def mse(y_true, y_pred):
            # Custom loss function
            end = y_true.shape[1]
            return keras.losses.mean_squared_error(y_true, y_pred[:,:end])
        
        # Model
        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((None,keras_input_dim))


        # outputs = Wt_layers(inputs,self)
        V_in = self.get_V_in(inputs) 
        outputs = tf.concat([Wt_layers(inputs,self),V_in],1)


        # Compile
        model = keras.models.Model(inputs, outputs)
        model.compile(loss=mse,optimizer='adam',metrics=[mse])
        return model
    


class RNNCell(keras.layers.Layer):
    def __init__(self, parameter):


        self.meta_dim = parameter.S.shape[0]
        self.flux_dim = parameter.S.shape[1]
        self.state_size = parameter.S.shape[1]
        self.input_size = parameter.P_in.shape[0]
        
        self.medium_bound = parameter.medium_bound
        self.hidden_dim = parameter.hidden_dim

        self.S  = tf.convert_to_tensor(np.float32(parameter.S))
        self.V2M = tf.convert_to_tensor(np.float32(parameter.V2M))
        self.P_in = tf.convert_to_tensor(np.float32(parameter.P_in))


        ## Is this the right place for that ?
        ## Is matrix M2V is used in other models ?
        # Normalize M2V
        M2V = parameter.M2V
        for i in range(self.flux_dim):
            if np.count_nonzero(M2V[i]) > 0:
                M2V[i] = M2V[i] / np.count_nonzero(M2V[i])

        self.M2V  = tf.convert_to_tensor(np.float32(M2V))
        self.droP_out = parameter.droP_out
        super(RNNCell, self).__init__(True)

    # Build need an input_shape argument.
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
            ## Is this possible to have hidden_dim == 0 ?
            if self.hidden_dim > 0:
                VH = keras.backend.dot(inputs, self.wh_V)
                V0 = keras.backend.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = keras.backend.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs # EB case
            
        V0 = tf.linalg.matmul(V0, self.P_in, b_is_sparse=True)
        M = tf.linalg.matmul(V,tf.transpose(self.V2M),b_is_sparse=True)
        W = tf.math.multiply(self.M2V,self.wr_V)
        V = tf.linalg.matmul(M,tf.transpose(W),b_is_sparse=True)
        V = V + V0 + self.br_V
        return V, [V]
    
    ## ????
    def get_config(self): # override tf.get_config to save RNN model
        # The code below does not work !! anyone to debug?
        config = super().get_config().copy()
        #config.update({'parameter': self.parameter.__dict__})
        return config
    