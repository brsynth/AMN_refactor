import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import keras
import tensorflow as tf
from loss import SV_loss, V_in_loss, V_pos_loss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns



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

def plot_regression(pred, true, pred_label, true_label, title, saving_file=None):
    plt.title(title)
    sns.set(font='arial', palette="colorblind", style="whitegrid", font_scale=2.5, rc={'figure.figsize':(11,11)})
    sns.regplot(x=true, y=pred, fit_reg=0, marker='+', color='black', scatter_kws={'s':40, 'linewidths':0.7})
    plt.xlabel(true_label)
    plt.ylabel(pred_label)
    plt.xlim(min(true)-0.01, max(true)+0.05)
    plt.ylim(min(pred)-0.02, max(pred)+0.05)

    if saving_file:
        plt.savefig(saving_file, format="png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close("all")

def plot_classification(pred, true, saving_file=None):

    roc = roc_curve(true, pred)
    auc = roc_auc_score(true, pred)
    plt.title('AUC = %.2f'% auc)

    sns.set(font='arial', palette="colorblind", style="whitegrid", font_scale=2.5, rc={'figure.figsize':(11,11)})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.xticks([0.2,0.4,0.6,0.8,1.0])
    sns.lineplot(x=roc[0], y=roc[1])

    if saving_file:
        plt.savefig(saving_file, format="png", dpi=600, bbox_inches='tight')

    plt.show()
    plt.close("all")
