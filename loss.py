import tensorflow as tf
import numpy as np

def loss_SV(V, S, gradient=False):
    # Gradient for SV constraint
    # Loss = ||SV||
    # dLoss =  ∂([SV]^2)/∂V = S^T SV
    S  = tf.convert_to_tensor(np.float32(S))
    Loss = tf.linalg.matmul(V, tf.transpose(S), b_is_sparse=True) 
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/S.shape[0] # rescaled
    if gradient:
        dLoss = tf.linalg.matmul(Loss, S, b_is_sparse=True) # derivate
        dLoss = dLoss / (S.shape[0]*S.shape[0])  # rescaling
        dLoss = dLoss / 2
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def loss_Vin(V, P_in, Vin, bound, gradient=False):
    # Gradient for input boundary constraint
    # Loss = ReLU(P_in . V - Vin)
    # dLoss = ∂(ReLU(P_in . V - Vin)^2/∂V
    # Input: Cf. Gradient_Descent
    P_in  = tf.convert_to_tensor(np.float32(P_in))
    Loss = tf.linalg.matmul(V, tf.transpose(P_in), b_is_sparse=True) - Vin
    Loss = tf.keras.activations.relu(Loss) if bound == 'UB' else Loss 
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/P_in.shape[0] # rescaled
    if gradient:
        dLoss = tf.math.divide_no_nan(Loss, Loss) # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss) # !!!
        dLoss = tf.linalg.matmul(dLoss, P_in, b_is_sparse=True) # resizing
        dLoss = dLoss / (P_in.shape[0] * P_in.shape[0])   # rescaling
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def loss_Vpos(V, gradient=False):
    # Gradient for V ≥ 0 constraint
    # Loss = ReLU(-V)
    # dLoss = ∂(ReLU(-V)^2/∂V
    Loss = tf.keras.activations.relu(-V)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/V.shape[1] # rescaled
    if gradient:
        dLoss = - tf.math.divide_no_nan(Loss, Loss) # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss) # !!!
        dLoss = dLoss / (V.shape[1] * V.shape[1]) # rescaling
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss