import numpy as np

def MaxScaler(data, Max_Scaler = 1.0e12):
    # Max standardize np array data
    if Max_Scaler == 1.0e12: # Scale
        Max_Scaler = np.max(data)
        data = data/Max_Scaler
    else: # Descale
        data = data * Max_Scaler
        Max_Scaler = 1.0e12      
    return data, Max_Scaler

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
    _, M2V = compute_V2M_M2V
    return M2V

def compute_V2M(S):
    n, m = S.shape[1], S.shape[0]
    V2M, _ = compute_V2M_M2V
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
    n, n_out = S.shape[1],len(measure)
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