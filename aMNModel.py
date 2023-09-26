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
        # Custom loss function
        end = y_true.shape[1]
        return sklearn.metrics.mean_squared_error(y_true, y_pred[:,:end])
    

    def loss_constraint(self, y_true, y_pred):

            # Get all predicted fluxes
            V = y_pred[:,y_true.shape[1]:y_true.shape[1]+self.S.shape[1]]
            V_in = y_pred[:,y_true.shape[1]+self.S.shape[1]:]   

            L = tf.concat([SV_loss(V, self.S), 
                       V_in_loss(V, self.P_in, V_in, self.medium_bound),
                       V_pos_loss(V)], axis=1)
            
            L = tf.math.square(L)
            L = tf.math.reduce_sum(L, axis=1)
            L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))     

            loss = np.mean(L.numpy())
            return loss
    
    
    def R2(self,y_true, y_pred):
        # Custom loss function
        end = y_true.shape[1]
        return sklearn.metrics.r2_score(y_true, y_pred[:,:end], multioutput='variance_weighted')

                            