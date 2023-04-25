import os
import sys
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from metabolicDataset import MetabolicDataset
from returnStats import ReturnStats


from keras.callbacks import EarlyStopping

class NeuralModel:
    """
    This class gives the general structure to run hybrid models. Hybrid model
    here refers to the article: ##ref.
    More precisely, three different concepts are used in this class:
    - First, this class contains information on dataset we use. The dataset
    could be experimental or generated by cobra. It could also be load from a
    given training_file. The MetabolicDataset class help us to get the flux 
    data and the metabolic information such as stoichiometry.
    - Then, this class contains a tensorflow model, given by the model type. 
    This model will be stored in the model attribute.
    - Finally, the class NeuralModel manage the preprocessing on dataset,
    train, test and evaluate model. Those methods will depend on the model
    type. We provide a child class for each model type, where adaptation are
    made in every model type context.
    """

    def __init__(self,
                 training_file=None,
                 objective=None, 
                 model=None, 
                 model_type='', 
                 scaler=False,
                 input_dim=0, output_dim=0,
                 n_hidden=0, hidden_dim=0, # default no hidden layer
                 activation='relu', 
                 timestep=0, 
                 learn_rate=1.0, 
                 decay_rate=0.9,
                 regression=True, 
                 epochs=0, train_rate=1e-3, droP_out=0.25, batch_size=5,
                 n_iter=0, xfold=5, # Cross validation LOO does not work ##?
                 early_stopping=False,
                 verbose=False,
                ):

        
        ## Create some attribute in the appropriate child class
        # Create empty object
        if model_type == '':
            sys.exit('Please give a model_type')

        # Data
        self.training_file = training_file
        self.X = None
        self.Y = None
        self.Y_all = None # Keep all the Y in case objective is given
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        # model architecture parameters
        self.model = model
        self.model_type = model_type
        
        self.objective = objective
        self.scaler = float(scaler) # From bool to float ## Why ???
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.timestep = timestep

        # LP or QP parameters
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        
        # Training parameters
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.droP_out = droP_out
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.xfold = xfold
        self.early_stopping = early_stopping
        self.medium_bound = '' # initialization
        self.Y_all = None
        

        self.get_parameter(verbose=verbose)
    

        
    def get_parameter(self, verbose=False):
        """
        This method create a MetabolicDataset object and uses it to fill attributes
        linked to the dataset.
        """

        self.check_training_file()
        parameter = MetabolicDataset(training_file = self.training_file)

        self.medium_bound = parameter.medium_bound
        self.level_med = parameter.level_med
        self.value_medium = parameter.value_medium

        # General matrices      
        self.S = parameter.S
        self.P_in = parameter.P_in 
        self.V2M = parameter.V2M 
        self.M2V = parameter.M2V 

        # Dataset
        self.X = parameter.X
        self.Y = parameter.Y
        self.Y_all = parameter.Y_all



        ## LP, QP...? Could be put in a dedicated subclass !
        self.Q = parameter.Q 
        self.P = parameter.P 
        self.b_int = parameter.b_int
        self.b_ext = parameter.b_ext
        self.Sb = parameter.Sb
        self.c = parameter.c
        self.S_int = parameter.S_int
        self.S_ext = parameter.S_ext

        

        self.P_out, self.Y = parameter.filter_measure(self.objective, verbose=verbose)
        

    def check_training_file(self):
        if self.training_file == None:
            sys.exit('Please give a training file')
        if not os.path.isfile(self.training_file+'.npz'):
            print(self.training_file+'.npz')
            sys.exit('training file is not found')



    def train_test_split(self, test_size, random_state):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, 
                                                                                self.Y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state)


    def train_evaluate(self, verbose=False):
        """
        This method takes care of all training steps, format the input depending on 
        the model type, then it creates a model, and finally train it and return stats
        and prediction. If the cross-validation is used the model attribute will be set
        to the best model encountered, ie. best Q2/Acc on the test folds.
        """

        # Preprocessing and reshape of the data X and Y, depending on the model's type.
        X, Y = self.model_input(self.X_train, self.Y_train, verbose=verbose)


        # no cross-validation
        if self.xfold ==1 :

            # Create the mechanistic model and fill the model attribute with it.
            self.set_model(verbose=verbose)
            Y_pred, stats, history = self.train_model(X,Y,X,Y, verbose=verbose)
            return Y_pred, stats, history

        else :
            best_model = self.model
            k_fold = KFold(n_splits=self.xfold, shuffle=True)
            o_max = -np.inf
            stats = ReturnStats()

            # The shape of the prediction depends on the model type, especially the number of columns.
            Y_pred = np.zeros((len(Y), self.nb_columns_pred()))

            # Cross-validation loop
            for train, test in k_fold.split(X, Y):
                if verbose: print('-------train', train, '\n-------test ', test)
                
                # Create a new network and train it 
                Net = self.copy() #shallow copy without model attribute
                Net.set_model(verbose=verbose)
                y_test_pred, new_stats, history = Net.train_model(X[train], 
                                                                  Y[train], 
                                                                  X[test], 
                                                                  Y[test], verbose=verbose)
                stats.update(new_stats)

                # Fill the prediction set for this test fold
                for i in range(len(test)):
                    Y_pred[test[i]] = y_test_pred[i]

                # Keep the model if it's the best we encounter
                if new_stats.test_obj[0] > o_max :
                    o_max = new_stats.test_obj[0]
                    best_model = Net.model

            self.model = best_model
            return Y_pred, stats, history



    def train_model(self, X_train, Y_train, X_test, Y_test, verbose=False):
        """
        A standard function to create a model, fit, and test with early
        stopping.
        """

        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min',
                           patience=10, verbose=verbose)
        callbacks = [es] if self.early_stopping else []

        # fit
        history = self.model.fit(X_train,
                                 Y_train,
                                 validation_data=(X_test, Y_test),
                                 epochs=self.epochs,
                                 batch_size=self.batch_size, 
                                 callbacks=callbacks,
                                 verbose=verbose)

        # evaluate
        _, o_train, l_train = self.evaluate_model(X_train, Y_train, verbose=verbose)
        y_test_pred, o_test, l_test  = self.evaluate_model(X_test,Y_test, verbose=verbose)

        stats = ReturnStats(train_obj = o_train, 
                            test_obj = o_test, 
                            train_loss = l_train, 
                            test_loss = l_test)
        stats.printout_train()
    
        return y_test_pred, stats, history
    



    def evaluate_model(self, x, y_true, verbose=False):
        """
        Return y_pred and stats on the model, R2 for regression and accuracy for classification.
        """

        y_pred = self.model.predict(x)

        # The model output could have more columns than the y_true. For
        # example the model could return some information to help us to
        # compute the loss. Also it can contains all the fluxes. 
        # The columns corresponding to the y_true ones are the references
        # fluxes and the three additional loss we want to uses.
        y_p = y_pred[:,:y_true.shape[1]]
        
        if self.regression:    
            ## This is odd 
            if len(y_true) == 1: # LOO case
                print('LOO True, Pred, Q2 =', y_true, y_p, obj)
                tss = y_true**2
                rss = (y_p - y_true)**2
                if np.sum(tss)== 0:
                    obj = 1 - np.sum(rss)
                else:
                    obj = 1 - np.sum(rss) / np.sum(tss)
            else:
                obj = r2_score(y_true, y_p, multioutput='variance_weighted')
        else:
            obj_ = keras.metrics.binary_accuracy(y_true, y_p).numpy()
            obj = np.count_nonzero(obj_)/len(obj_)

        # Compute stats on constraints. The loss depends on the model type.
        loss = self.compute_loss(x, y_true, y_pred, verbose=verbose)

        return y_pred, obj, loss


    def copy(self):
        net = self.__class__(training_file = self.training_file,
                               objective= self.objective, 
                               model=None, 
                               model_type=self.model_type, 
                               scaler=self.scaler,
                               input_dim=self.input_dim, 
                               output_dim=self.output_dim, 
                               n_hidden=self.n_hidden, 
                               hidden_dim=self.hidden_dim, 
                               activation=self.activation,
                               timestep=self.timestep,
                               learn_rate=self.learn_rate,
                               decay_rate=self.decay_rate,
                               regression=self.regression,
                               epochs=self.epochs, 
                               train_rate=self.train_rate,
                               droP_out=self.droP_out,
                               batch_size=self.batch_size,
                               n_iter=self.n_iter,                         
                               xfold=self.xfold,  
                               early_stopping=self.early_stopping,
                               verbose=False)
        return net

    ## This function could fill attribute during the instantiation of a metabolic model.
    def nb_columns_pred(self):
        """This method depend on the model type."""
        raise NotImplementedError

    def model_input(self, X, Y, verbose=False):
        """This method depend on the model type."""
        raise NotImplementedError
    
    def set_model(self, verbose=False):
        """This method depend on the model type."""
        raise NotImplementedError
    
    def compute_loss(self, x, y_true, y_pred, verbose=False):
        """This method depend on the model type."""
        raise NotImplementedError



############################################################ To rewrite ############################################################ 


    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
            
        print('training file:', self.training_file)
        print('model type:', self.model_type)
        print('model scaler:', self.scaler)
        print('model input dim:', self.input_dim)
        print('model output dim:', self.output_dim)
        print('model medium bound:', self.medium_bound)
        print('timestep:', self.timestep)
        if self.training_file:
            if os.path.isfile(self.training_file+'.npz'):
                print('training set size', self.X.shape, self.Y.shape)
        else:
             print('no training set provided')
        if self.n_hidden > 0:
            print('nbr hidden layer:', self.n_hidden)
            print('hidden layer size:', self.hidden_dim)
            print('activation function:', self.activation)
        if self.model_type == 'AMN_QP' and self.timestep > 0:
            print('gradient learn rate:', self.learn_rate)
            print('gradient decay rate:', self.decay_rate)
        if self.epochs > 0:
            print('training epochs:', self.epochs)
            print('training regression:', self.regression)
            print('training learn rate:', self.train_rate)
            print('training droP_out:', self.droP_out)
            print('training batch size:', self.batch_size)
            print('training validation iter:', self.n_iter)
            print('training xfold:', self.xfold)
            print('training early stopping:', self.early_stopping)
        if filename != '':
            sys.stdout.close()
    

    ## not used
    def save(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        print(fileparam)
        filemodel = filename + "_model.h5"
        s = str(self.training_file) + ","\
                    + str(self.model_type) + ","\
                    + str(self.objective) + ","\
                    + str(self.scaler) + ","\
                    + str(self.input_dim) + ","\
                    + str(self.output_dim) + ","\
                    + str(self.n_hidden) + ","\
                    + str(self.hidden_dim) + ","\
                    + str(self.activation) + ","\
                    + str(self.timestep) + ","\
                    + str(self.learn_rate) + ","\
                    + str(self.decay_rate) + ","\
                    + str(self.epochs) + ","\
                    + str(self.regression) + ","\
                    + str(self.train_rate) + ","\
                    + str(self.droP_out) + ","\
                    + str(self.batch_size) + ","\
                    + str(self.n_iter) + ","\
                    + str(self.xfold) + ","\
                    + str(self.early_stopping)
        with open(fileparam, "w") as h:
            # print(s, file = h)
            h.write(s)
        self.model.save(filemodel)

    ## not used.
    def load(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        if not os.path.isfile(fileparam):
            print(fileparam)
            sys.exit('parameter file not found')
        if not os.path.isfile(filemodel):
            print(filemodel)
            sys.exit('model file not found')
        # First read parameter file
        with open(fileparam, 'r') as h:
            for line in h:
                K = line.rstrip().split(',')
        # model architecture
        self.training_file =  str(K[0])
        self.model_type =  str(K[1])
        self.objective =  str(K[2])
        self.scaler =  float(K[3])
        self.input_dim =  int(K[4])
        self.output_dim = int(K[5])
        self.n_hidden = int(K[6])
        self.hidden_dim = int(K[7])
        self.activation = str(K[8])
        # GD parameters
        self.timestep = int(K[9])
        self.learn_rate = float(K[10])
        self.decay_rate = float(K[11])
        # Training parameters
        self.epochs = int(K[12])
        self.regression = True if K[13] == 'True' else False
        self.train_rate = float(K[14])
        self.droP_out = float(K[15])
        self.batch_size = int(K[16])
        self.n_iter = int(K[17])
        self.xfold = int(K[18])
        self.early_stopping = True if K[19] == 'True' else False

        ## Regexp my friend :)
        # Make objective a list
        self.objective = self.objective.replace('[', '')
        self.objective = self.objective.replace(']', '')
        self.objective = self.objective.replace('\'', '')
        self.objective = self.objective.replace("\"", "")
        self.objective = self.objective.split(',')
        # Get additional parameters (matrices)
        self.get_parameter(verbose=verbose)
       


def load_model(filemodel,something,compile=False):
    """This is a fake function to not get error above. Waiting for the load model
    issue to be explore."""
    pass

def print_loss_evaluate(y_true, y_pred, Vin, parameter):
    """This is a fake function. This is not used in the part of the code we run in the notebook.
    """
    pass


