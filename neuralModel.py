import sys
import copy
import keras
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from returnStats import ReturnStats
from metabolicDataset import MetabolicDataset



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
                 dataset_file=None,
                 objective=None, 
                 scaler=False,
                 n_hidden=0, 
                 hidden_dim=0, # default no hidden layer
                 activation='relu', 
                 timestep=0, 
                 learn_rate=1.0, 
                 regression=True, 
                 epochs=0, 
                 train_rate=1e-3, 
                 droP_out=0.25, 
                 batch_size=5,
                 n_iter=0, 
                 xfold=5, # Cross validation LOO does not work ##?
                 early_stopping=False,
                 verbose=False,
                ):
        
        dataset = MetabolicDataset(dataset_file=dataset_file)

        # data
        self.Y_all = dataset.Y_all # Keep all the Y in case objective is given
        self.X = dataset.X
        self.Y = dataset.Y
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_all = None ## ?

        # keras model architecture
        self.model = None 
        self.medium_bound = dataset.medium_bound
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.timestep = timestep
        self.activation = activation

        # preprocessing
        self.scaler = float(scaler) # From bool to float ## Why ???

        # Training parameters
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.droP_out = droP_out
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.xfold = xfold
        self.early_stopping = early_stopping

        # metabolic data
        self.dataset_file = dataset_file
        self.objective = objective
        self.S = dataset.S
        self.P_in = dataset.P_in 
        self.V2M = dataset.V2M 
        self.M2V = dataset.M2V 

        ## issue with the Y overwriting !
        self.P_out, self.Y = dataset.filter_measure(self.objective, verbose=verbose)

 

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

            # Create the model and fill the model attribute with it.
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
                Net = copy.deepcopy(self)
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
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
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

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        self.printout_by_type()
        if filename != '':
            sys.stdout.close()
        
    def printout_by_type(self):
        raise NotImplementedError
