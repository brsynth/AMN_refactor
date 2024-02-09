import os
import sys
import cobra
import numpy as np
from amn.tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V




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
                 cobra_file='',
                 medium_file='', 
                 medium_bound='UB', 
                 method='FBA',
                 measure=None, 
                 verbose=False):

        # Load a preexisting dataset from file
        if dataset_file !='':
            self.load(dataset_file)
            return

        # Need the cobra model to create the dataset
        self.model = cobra.io.read_sbml_model(cobra_file)
        self.reactions = [r.id for r in list(self.model.reactions)]
        self.measure = measure if measure else self.reactions.copy()
        self.medium_file = medium_file
        self.medium_bound = medium_bound # EB or UB
        self.method = method
        self.verbose=verbose
        

    def save(self, dataset_dir, dataset_name,verbose=False):

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


    def printout(self):
        for k,v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) ==2:
                    print("%s : %s"% (k, v.shape))
                else:
                    print("%s : %s"% (k, v))    