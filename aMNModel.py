import numpy as np
import tensorflow as tf
from tools import MaxScaler
from neuralModel import NeuralModel
from loss import loss_SV, loss_Vin, loss_Vpos



class AMNModel(NeuralModel):

    def __init__(self, **kwargs):
        NeuralModel.__init__(self, **kwargs)

        
    def model_input(self, X, Y, verbose=False):
        """
        This method normalize X if scaler attribute is True. Then it add three
        zeros columns to Y making the loss to minimize SV, P_in and V≥0
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
    
    


    ## TO CHANGE !!!
    def get_loss_evaluate(self, x, y_true, y_pred, verbose=False):
        """" 
        This method compute a loss on constraint on y_pred. Remind that if 
        V_pred are all the fluxes predicted by the model, y_pred is a 
        concatenation of P_out.V_pred, S.V_pred, P_in.V_pred and ReLU(V_pred).
        This is not necessarily the loss used in the model optimization.
        """


        V_final = y_pred[:,y_true.shape[1]:y_true.shape[1]+self.S.shape[1]]
        V_in = self.get_V_in(x)        
        if verbose: ## Have to rewrite the function print_loss_evaluate
            print_loss_evaluate(y_true, y_pred, V_in, self)               
        loss, _ = self.Loss_constraint(V_final, V_in)
        loss = np.mean(loss.numpy())
        return loss

    def Loss_constraint(self, V, Vin, gradient=False):
        # mean squared sum L2+L3+L4
        L2, dL2 = loss_SV(V, self.S, gradient=gradient)
        L3, dL3 = loss_Vin(V, self.P_in, Vin,
                           self.medium_bound, gradient=gradient)
        L4, dL4 = loss_Vpos(V, gradient=gradient)

        # square sum of L2, L3, L4
        L2 = tf.math.square(L2)
        L3 = tf.math.square(L3)
        L4 = tf.math.square(L4)
        L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
        # divide by 3 
        L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))

        return L, dL2+dL3+dL4






    ## must be obtained from somewhere else self.Y.shape...
    def get_V_in(self, x):
        """This method depend on the model type."""
        raise NotImplementedError










## Fake function area !!!
def print_loss_evaluate(filemodel,truc,compile=False):
    """This is a fake function to not get error above. Waiting for the load model
    issue to be explore."""
    pass

                            