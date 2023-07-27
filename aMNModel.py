import numpy as np
import tensorflow as tf
from tools import MaxScaler
from neuralModel import NeuralModel
from loss import SV_loss, V_in_loss, V_pos_loss




class AMNModel(NeuralModel):

    def __init__(self, **kwargs):
        NeuralModel.__init__(self, **kwargs)

        
    def model_input(self, X, Y, verbose=False):
        """
        This method normalize X if scaler attribute is True. Then it add three
        zero columns to Y making the loss to minimize SV, P_in and V≥0
        constraint easier to compute. 
        Finally, it call the model_type_input method to prepare X and Y as
        input for the model, depending on its type.
        """

        if self.scaler != 0: 
            X, self.scaler = MaxScaler(X) 
        if verbose: print('AMN scaler', self.scaler)

        Y = np.concatenate((Y, np.zeros((len(Y),3))), axis=1)

        # Preparing X and Y according to the model type.
        X,Y = self.model_input_by_type(X,Y)
        if verbose: print(self.model_type + ' shape', X.shape, Y.shape)        

        return X,Y
    

    def model_input_by_type(self, X, Y):
        """
        This method prepare X and Y depending on the type of model used.
        """
        raise NotImplementedError
    
    
    def compute_loss(self, x, y_true, y_pred, verbose=False):
        """
        This method compute a loss on constraint on y_pred. Remind that y_pred
        is a concatenation of P_out.V_pred, S.V_pred, P_in.V_pred and ReLU(V_pred).
        This is not necessarily the loss used in the model optimization.
        """

        # Get all predicted fluxes
        V_final = y_pred[:,y_true.shape[1]:y_true.shape[1]+self.S.shape[1]]
        V_in = self.get_V_in(x)        
        if verbose: ## Not functional
            print_loss_evaluate(y_true, y_pred, V_in, self)               
        loss = self.constraint_loss(V_final, V_in)
        loss = np.mean(loss.numpy())
        return loss
    
    def constraint_loss(self,V,V_in):
        # mean squared sum L2+L3+L4
        L2 = SV_loss(V, self.S)
        L3 = V_in_loss(V, self.P_in, V_in,
                           self.medium_bound)
        L4 = V_pos_loss(V)

        # square sum of L2, L3, L4
        L2 = tf.math.square(L2)
        L3 = tf.math.square(L3)
        L4 = tf.math.square(L4)
        L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
        # divide by 3 
        L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))

        return L
    


    ## must be obtained from somewhere else self.Y.shape...
    def get_V_in(self, x):
        """This method depend on the model type."""
        raise NotImplementedError


def print_loss_evaluate(filemodel,truc,compile=False):
    """This is a fake function to not get error above. Waiting for the load model
    issue to be explore."""
    pass

                            