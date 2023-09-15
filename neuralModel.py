import sys
# import copy
# import keras
import numpy as np
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split 
# from sklearn.model_selection import KFold
# from returnStats import ReturnStats
from metabolicDataset import MetabolicDataset
from tools import  compute_P_out





class NeuralModel:
    """
    This class gives the general structure to run hybrid models. Hybrid model
    here refers to the article: ##ref.
    This class contains information on dataset we use. The dataset
    is load from two file : <dataset_file>.xml and <dataset_file>.npz.
    This class contains a tensorflow model, stored in the model attribute.
    This class NeuralModel manage the preprocessing on dataset, train, test
    and evaluate model. Those methods will depend on the model type. 
    Different types of model are coded in different child classes.
    """

    def __init__(self,
                 dataset_file,
                 objective=None,
                #  scaler=False,
                 n_hidden=0, 
                 hidden_dim=0, # default no hidden layer
                 activation='relu', 
                 timestep=0, 
                 regression=True, 
                 epochs=0, 
                 batch_size=5,
                 n_iter=0, 
                #  xfold=5, # Cross validation LOO does not work ##?
                 early_stopping=False,
                 verbose=False,
                ):
        
        dataset = MetabolicDataset(dataset_file=dataset_file)

        # data
        self.X = dataset.X
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        # Keras model architecture
        self.model = None 
        self.medium_bound = dataset.medium_bound
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.timestep = timestep
        self.activation = activation

        # Training parameters
        self.epochs = epochs
        self.regression = regression
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.early_stopping = early_stopping

        # metabolic data
        self.dataset_file = dataset_file
        self.S = dataset.S
        self.P_in = dataset.P_in 
        self.V2M = dataset.V2M 
        self.M2V = dataset.M2V
        self.M2V_norm = self.norm_M2V(dataset.M2V)


        objective_ = objective if objective else dataset.measure
        self.P_out = compute_P_out(dataset.S, objective_, list(dataset.reactions))

        if dataset.method_generation == "SIMULATED":
            self.Y = np.matmul(dataset.Y,np.transpose(self.P_out))
        elif dataset.method_generation == "EXPERIMENTAL":
            self.Y = dataset.Y

        if verbose:
            print('number of metabolites: ', dataset.S.shape[0])
            print('filtered measurements size: ',self.Y.shape[1])



    def norm_M2V(self, M2V):
        norm_M2V = M2V.copy()
        for i,row in enumerate(M2V):
            if np.count_nonzero(row) > 0:
                norm_M2V[i] = row / np.count_nonzero(row)
        return norm_M2V


    def train_test_split(self, test_size, random_state):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, 
                                                                                self.Y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state)
        
    def preprocess(self,scaler):
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


    def model_input(self, X, Y, verbose=False):
        """This method depend on the model type."""
        raise NotImplementedError
    
  
    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        self.printout_by_type()
        if filename != '':
            sys.stdout.close()
        
    def printout_by_type(self):
        raise NotImplementedError
    
