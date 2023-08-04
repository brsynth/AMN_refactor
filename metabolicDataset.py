import os
import sys
import cobra
import numpy as np
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V




class MetabolicDataset:
    """
    This class manage the dataset with all useful information coming from the
    metabolic model that are used in the NeuralNetwork class. Child class 
    provide ways to populate the data. The dataset can also be save and 
    loaded from two files : one for the cobra model and one for the X,Y,
    and metabolic information such as medium, stoichiometry.
    """

    def __init__(self,
                 dataset_file='',
                 input_cobra_file='',
                 medium_file='', 
                 medium_bound='EB', 
                 objective=[], 
                 method='FBA',
                 measure=[], 
                 verbose=False):

        # Load a preexisting dataset from file
        if dataset_file !='':
            self.load(dataset_file)
            return

        # Need the cobra model to create the dataset
        self.model = cobra.io.read_sbml_model(input_cobra_file)
        self.reactions = [r.id for r in list(self.model.reactions)]
        self.objective = objective if objective else [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        self.measure = measure if measure else self.reactions.copy()
        self.medium_file = medium_file
        self.medium_bound = medium_bound # EB or UB
        self.method = method
        self.verbose=verbose
        
        if verbose:
            print('medium:',self.medium)
            print('level_med:',self.level_med)
            print('value_medium:',self.value_medium)
            print('ratio_medium:',self.ratio_medium)
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))


    def save(self, dataset_dir, dataset_name,verbose=False):

        self.Y_all = self.Y.copy() ## Just to make the code working, What is the purpose of Y_all anyway ???

        # save cobra model and other parameters
        filename = os.path.join(dataset_dir,dataset_name)
        self.cobra_name = filename
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        parameters = self.__dict__.copy()
        del parameters["model"]
        np.savez_compressed(filename, **parameters)
       
    
    def load(self, dataset_file):

        loaded = np.load(dataset_file)
        for key in loaded:
            setattr(self, key, loaded[key])


    def check_file_name_npz(self, file_name):
        if not os.path.isfile(file_name+'.npz'):
            print(file_name+'.npz')
            sys.exit('file not found')


    def valid_cobra_file(self, cobra_name):
        if cobra_name == '':
            sys.exit('Give a training file or a appropriate cobra_name.')
        if not os.path.isfile(cobra_name+'.xml'):
            print(cobra_name)
            sys.exit('xml cobra file not found')
        return cobra_name


    def valid_medium_file(self, medium_file):
        if medium_file == '':
            sys.exit('Give a training file or a appropriate medium_name.')
        if not os.path.isfile(medium_file+'.csv'):
            print(medium_file)
            sys.exit('medium file not found')
        return medium_file
    

    def filter_measure(self, objective, verbose=False):
        """
        This method return values of flux for given objectives (every column
        correspond to an objective) using the value of all flux matrix. 
        It also return the matrix P_out that is the projection of all fluxes
        on objective ones.
        """

        if not objective:
            P_out = compute_P_out(self.S, self.measure, list(self.reactions))
            Y = self.Y
        else:
            P_out = compute_P_out(self.S, objective, list(self.reactions))
            if self.method == "pFBA" or self.method == "FBA":
                Y = np.matmul(self.Y,np.transpose(P_out))
            else:
                Y = self.Y

        if verbose:
            print('number of reactions: ', self.S.shape[1], self.Y_all.shape[1])
            print('number of metabolites: ', self.S.shape[0])
            print('filtered measurements size: ',Y.shape[1])

        return P_out, Y 


    def printout(self):
        for k,v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) ==2:
                    print("%s : %s"% (k, v.shape))
                else:
                    print("%s : %s"% (k, v))

    