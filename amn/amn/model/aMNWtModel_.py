import keras
from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.layers import concatenate,RNN
from keras.utils.generic_utils import CustomObjectScope
from amn.model.aMNModel import AMNModel
from amn.tools import custom_loss


class AMNWtModel_(AMNModel):
    def __init__(self, **kwargs):
        AMNModel.__init__(self, **kwargs)


    def build_model(self):

        tf.random.set_seed(10)

        keras_input_dim = self.X.shape[1]
        inputs = keras.Input((keras_input_dim,))

        pre_layer = tf.keras.layers.Dense(keras_input_dim)
        V0 = pre_layer(inputs)

        # Simple injection or dense layer :
        V0 = tf.linalg.matmul(V0, self.P_uptake)
        # or
        # layer_dense = tf.keras.layers.Dense(self.S.shape[1])
        # V0 = layer_dense(V0)

        # Use Wt layer or not :
        Wt_layer = WtLayer(self.M2V_norm, self.V2M,self.S.shape[1], self.S.shape[0],self.timestep)
        outputs = tf.keras.layers.Concatenate(1)([Wt_layer(V0), inputs])
        # or 
        # outputs = tf.keras.layers.Concatenate(1)([V0, inputs])

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

    
class WtLayer(keras.layers.Layer):

    def __init__(self,M2V,V2M,input_dim,units,timestep=4):
        super().__init__()

        self.V2M = np.float32(V2M)
        self.M2V = np.float32(M2V)
        self.input_dim = input_dim
        self.timestep = timestep

        self.wr_V = self.add_weight(shape=(input_dim, units), 
                                           name='kernel_Vr',
                                           trainable=True)
        

        self.br_V  = self.add_weight(shape=(input_dim,), 
                                            initializer='random_normal',
                                            name='bias_Vr',
                                            trainable=True)

    def call(self, inputs):
        A = tf.linalg.matmul(tf.transpose(self.V2M),tf.transpose((tf.math.multiply(self.M2V,self.wr_V))))
        A_2 = tf.linalg.matmul(A, A)
        A_3 = tf.linalg.matmul(A_2, A)
        return tf.linalg.matmul(inputs, tf.cast(tf.identity(self.input_dim),tf.float32) + A + A_2 + A_3) + self.br_V

    
    def call(self, inputs):
        A = tf.linalg.matmul(tf.transpose(self.V2M),tf.transpose((tf.math.multiply(self.M2V,self.wr_V))))

        geom_series_A = tf.cast(tf.identity(self.input_dim),tf.float32)
        new_power = A

        for _ in range(self.timestep -1):
            geom_series_A += new_power
            new_power = tf.linalg.matmul(new_power,A)

        return tf.linalg.matmul(inputs, geom_series_A) + self.br_V    

