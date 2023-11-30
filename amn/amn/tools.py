import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import keras
import tensorflow as tf
from amn.loss import SV_loss, V_in_loss, V_pos_loss


class MaxScaler(BaseEstimator,TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.max = None
    def fit(self,X,y=None):
        self.max = np.max(X)
        return self
    def transform(self,X,y=None):
        return X/self.max


def compute_V2M_M2V(S):
    n, m = S.shape[1], S.shape[0]
    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                ## Where is the 1/z_i ?
                M2V[i][j] = -1/S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)
    return V2M,M2V


def compute_M2V(S):
    n, m = S.shape[1], S.shape[0]
    _, M2V = compute_V2M_M2V(S)
    return M2V


def compute_V2M(S):
    n, m = S.shape[1], S.shape[0]
    V2M, _ = compute_V2M_M2V(S)
    return V2M


def compute_P_in(S, medium, model_reactions):
    n, n_in = S.shape[1], len(medium)
    P_in = np.zeros((n_in,n))
    i = 0
    for rid in medium:
        j = model_reactions.index(rid)
        P_in[i][j] = 1
        i = i+1
    return P_in


def compute_P_out(S, measure, model_reactions):
    n_out, n = len(measure), S.shape[1]
    P_out = np.zeros((n_out,n))
    for i, rid in enumerate(measure):
        j = model_reactions.index(rid)
        P_out[i][j] = 1
    return P_out  


def printout(filename, time, obj, loss):
    print('Stats for %s CPU-time %.4f' % (filename, time))
    print('R2 = %.4f Constraint = %.4f' % \
              (obj, 
               loss))


def custom_loss(S, P_out, P_in):
    def my_mse(y_true, y_pred):

        S_ = S
        V = y_pred[:,:S_.shape[1]]
        V_in = y_pred[:,S_.shape[1]:]

           
        Pout = tf.convert_to_tensor(np.float32(P_out))        
 
        L = tf.concat([tf.linalg.matmul(V, tf.transpose(Pout)),
                       SV_loss(V, S), 
                       V_in_loss(V, P_in, V_in, "UB"),
                       V_pos_loss(V)], 
                       axis=1)
        
        return keras.losses.mean_squared_error(y_true,L)
    return my_mse 


def threshold_percentage_max(X,percentage):
    """This function return a binary vector from X by threshold a certain
    percentage of the maximum value of X."""
    max = X.max()
    X_threshold = X >= max*percentage
    return X_threshold.astype(int)


import os
import pathlib 
import zipfile
from pathlib import Path


def unzip_folders(main_directory):
    """Extract all the files from all the subdirectory of 
    the given directory."""
    sub_dir = os.listdir(main_directory)
    for d in sub_dir:
        if zipfile.is_zipfile(main_directory + d):
            unzip_folder(main_directory, d)

def unzip_folder(path_directory, folder_name):
    """Extract all files from the given folder of the given
    directory into this directory."""
    file = path_directory + folder_name
    print("Extracting",file)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(path_directory)


def zip_models(model_directory):
    """Transform every file with ".keras" suffix in the given model
    directory into a compressed archive."""
    for file_name in os.listdir(model_directory):
        if pathlib.Path(model_directory + file_name).suffix == ".keras":
            zip_file(model_directory, file_name)

def zip_file(path_directory, file_name):
    """Transform a given file into a compressed archive in the same given
    directory."""
    file = path_directory + file_name
    archive_path = path_directory + Path(file_name).stem + ".zip"
    print("Create archive",archive_path)
    zf = zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED)
    zf.write(file, file_name)