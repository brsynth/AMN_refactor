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
                 dataset_dir='',
                 dataset_file='',
                 input_cobra_file='',
                 medium_file='', 
                 medium_bound='EB', 
                 medium_size=-1,
                 objective=[], 
                 method='FBA',
                 measure=[], 
                 verbose=False):

        # Load a preexisting dataset from file
        if dataset_file !='':
            self.load(dataset_file)
            return

        # Need the cobra model to create the dataset
        # self.cobra_name = self.valid_cobra_file(cobra_name) # model cobra file
        self.model = cobra.io.read_sbml_model(input_cobra_file)
        self.reactions = [r.id for r in list(self.model.reactions)]
        self.objective = objective if objective else [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        self.measure = measure if measure else self.reactions.copy()
        self.medium_file = medium_file
        self.medium_bound = medium_bound # EB or UB
        self.method = method
        self.verbose=verbose
        
        ## Explain reduce !
        self.reduce = False ## lol !

        if verbose:
            print('medium:',self.medium)
            print('level_med:',self.level_med)
            print('value_medium:',self.value_medium)
            print('ratio_medium:',self.ratio_medium)
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))

    def save(self, dataset_dir, dataset_name, reduce=False, verbose=False):

        filename = os.path.join(dataset_dir,dataset_name)
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose) 

        ## strange to do that here ! Is this because its not not done in the reduce and run ?
        # recompute matrices
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.Pin = compute_P_in(self.S, self.medium, self.reactions)
        self.Pout = compute_P_out(self.S, self.measure, self.reactions)
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)

        ## Not used in this code ! To remove !
        # self.S_int= 0
        # self.S_ext, self.Q, self.P, \
        # self.b_int, self.b_ext, self.Sb, self.c = 0,0,0,0,0,0,0
        # self.all_matrices=False
        self.Y_all = self.Y.copy() ## Just to make the code working, What is the purpose of Y_all anyway ???

        self.cobra_name = filename


        # save cobra model
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save other parameters
        parameters = self.__dict__.copy()
        del parameters["model"]
        np.savez_compressed(filename, **parameters)
       
    
    def load(self, dataset_file):

        # self.check_file_name_npz(file_name)
        # Load parameters from npz file
        # loaded = np.load(os.path.join(directory,file_name)+'.npz')
        loaded = np.load(dataset_file)
        for key in loaded:
            setattr(self, key, loaded[key])

        # Load cobra model from xml file
        #self.model = cobra.io.read_sbml_model(str(self.cobra_name)+'.xml')


    def printout(self):
        for k,v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) ==2:
                    print("%s : %s"% (k, v.shape))
                else:
                    print("%s : %s"% (k, v))


    def reduce_and_run(self,verbose=False):
        # reduce a model recompute matrices and rerun cobra
        # with the provided training set
        measure = [] if len(self.measure) == len(self.reactions) \
        else self.measure
        self.model = reduce_model(self.model, self.medium, measure,
                                  self.Y, verbose=verbose)
        self.measure = [r.id for r in self.reactions] \
        if measure == [] else measure

        self.get(sample_size=self.size, reduce=True, verbose=verbose)


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
            # We recompute P_out, then Y depending on the objective
            P_out = compute_P_out(self.S, objective, list(self.reactions))
            if self.method == "pFBA":
                Y = np.matmul(self.Y,np.transpose(P_out))
            else:
                Y = self.Y

        if verbose:
            print('number of reactions: ', self.S.shape[1], self.Y_all.shape[1])
            print('number of metabolites: ', self.S.shape[0])
            print('filtered measurements size: ',Y.shape[1])

        return P_out, Y 
    

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
    