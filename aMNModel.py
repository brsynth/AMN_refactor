import sklearn
import numpy as np
import tensorflow as tf
from neuralModel import NeuralModel
from loss import SV_loss, V_in_loss, V_pos_loss

class AMNModel(NeuralModel):

    def __init__(self, **kwargs):
        NeuralModel.__init__(self, **kwargs)

        
    def model_input(self, X, Y, verbose=False):
        """
        This method add three zero columns to Y making the loss to minimize
        SV, P_in and V≥0 constraint easier to compute. 
        """
        Y = np.concatenate((Y, np.zeros((len(Y),3))), axis=1)
        return X,Y
    

    def preprocessing_for_specific_model(self):
        self.X_train, self.Y_train = self.model_input(self.X_train, self.Y_train)
        self.X_test, self.Y_test = self.model_input(self.X_test, self.Y_test)


    def mse(self,y_true, y_pred):

        V = y_pred[:,:self.S.shape[1]]
        V_in = y_pred[:,self.S.shape[1]:]

        P_out     = tf.convert_to_tensor(np.float32(self.P_out))        
        L = tf.concat([tf.linalg.matmul(V, tf.transpose(P_out)),
                       SV_loss(V, self.S), 
                       V_in_loss(V, self.P_in, V_in, self.medium_bound),
                       V_pos_loss(V)], 
                       axis=1)

        return sklearn.metrics.mean_squared_error(y_true,L)
    
    def R2(self,y_true, y_pred):
        
        V = y_pred[:,:self.S.shape[1]]
        V_in = y_pred[:,self.S.shape[1]:]


        P_out     = tf.convert_to_tensor(np.float32(self.P_out))        
        L = tf.concat([tf.linalg.matmul(V, tf.transpose(P_out)),
                       SV_loss(V, self.S), 
                       V_in_loss(V, self.P_in, V_in, self.medium_bound),
                       V_pos_loss(V)], 
                       axis=1)
    
        return sklearn.metrics.r2_score(y_true, L, multioutput='variance_weighted')
    

    def loss_constraint(self, y_true, y_pred):

        V = y_pred[:,:self.S.shape[1]]
        V_in = y_pred[:,self.S.shape[1]:]

        L = tf.concat([SV_loss(V, self.S), 
                   V_in_loss(V, self.P_in, V_in, self.medium_bound),
                   V_pos_loss(V)], axis=1)
        
        L = tf.math.square(L)
        L = tf.math.reduce_sum(L, axis=1)
        L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))     
        loss = np.mean(L.numpy())
        return loss
    



    


                            