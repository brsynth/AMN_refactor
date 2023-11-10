import tensorflow as tf
import numpy as np


def SV_loss(V,S):
    S  = tf.convert_to_tensor(np.float32(S))
    loss = tf.linalg.matmul(V, tf.transpose(S))
    loss_norm = tf.norm(loss, axis=1, keepdims=True)/S.shape[0] # rescaled
    return loss_norm


def V_in_loss(V, P_in, V_in, bound):
    Pin  = tf.convert_to_tensor(np.float32(P_in))
    loss = tf.linalg.matmul(V, tf.transpose(Pin)) - V_in
    loss = tf.keras.activations.relu(loss) if bound == 'UB' else loss 
    loss_norm = tf.norm(loss, axis=1, keepdims=True)/P_in.shape[0] # rescaled
    return loss_norm


def V_pos_loss(V):
    loss = tf.keras.activations.relu(-V)
    loss_norm = tf.norm(loss, axis=1, keepdims=True)/V.shape[1] # rescaled
    return loss_norm

