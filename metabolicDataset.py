import os
import sys
import cobra
import numpy as np
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V


class MetabolicDataset:
    """
    This class manage the dataset and all useful information coming from the metabolic
    model that are used in the NeuralNetwork class. The attributes can be found in 
    different ways : from a given training file, or by extracting information from 
    given cobra_name, medium_name and method.
    """

    def __init__(self,
                 training_file='',
                 cobra_name='',
                 medium_name='', 
                 medium_bound='EB', 
                 medium_size=-1,
                 objective=[], 
                 method='FBA',
                 measure=[], 
                 verbose=False):


        if training_file !='':
            # self.load(training_file)
            
            self.load("./",training_file)

            return
        
        self.model = cobra.io.read_sbml_model(cobra_name+'.xml')

        ## correspond to exp or simulated ?
        if objective:
            self.objective = objective
        else:
            self.objective = [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        
        if measure:
            self.measure = measure
        else:
            self.measure = [r.id for r in self.model.reactions]


        
        self.medium_name = self.valid_medium_file(medium_name)
        self.medium_bound = medium_bound # EB or UB
        self.method = method
        self.verbose=verbose

        ## Explain reduce !
        self.reduce = False ## lol !

        self.cobra_name = self.valid_cobra_file(cobra_name) # model cobra file

        if verbose:
            print('medium:',self.medium)
            print('level_med:',self.level_med)
            print('value_medium:',self.value_medium)
            print('ratio_medium:',self.ratio_medium)
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))

        

        
    

    def save(self, directory, filename, reduce=False, verbose=False):

        filename = os.path.join(directory,"Dataset",filename)
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose)

        ## strange to do that here !
        # recompute matrices
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.Pin = compute_P_in(self.S, self.medium, self.model.reactions)
        self.Pout = compute_P_out(self.S, self.measure, self.model.reactions)
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)

        ## Not used in this code ! To remove !
        self.S_int= 0
        self.S_ext, self.Q, self.P, \
        self.b_int, self.b_ext, self.Sb, self.c = 0,0,0,0,0,0,0
        self.all_matrices=False
        self.Y_all = self.Y.copy() ## Just to make the code working, What is the purpose of Y_all anyway ???

        self.cobra_name = filename


        # save cobra model
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save other parameters
        parameters = self.__dict__.copy()
        del parameters["model"]
        np.savez_compressed(filename, **parameters)
       
        

    def load(self, directory, file_name):

        # self.check_file_name_npz(file_name)
        # Load parameters from npz file
        loaded = np.load(os.path.join(directory,"Dataset",file_name)+'.npz')
        for key in loaded:
            setattr(self, key, loaded[key])

        # Load cobra model from xml file
        self.model = cobra.io.read_sbml_model(str(self.cobra_name)+'.xml')


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
        measure = [] if len(self.measure) == len(self.model.reactions) \
        else self.measure
        self.model = reduce_model(self.model, self.medium, measure,
                                  self.Y, verbose=verbose)
        self.measure = [r.id for r in self.model.reactions] \
        if measure == [] else measure

        self.get(sample_size=self.size, reduce=True, verbose=verbose)




    def filter_measure(self, objective, verbose=False):
        """
        This method return values of Y and P_out depending on the given objective.
        The objective argument is a list of measured reactions flux.
        """

        if not objective:
            ## Is this the good logic to give self.measure as argument ?
            P_out = compute_P_out(self.S, self.measure, self.model.reactions)
            Y = self.Y
        else:
            # We recompute P_out, then Y depending on the objective
            P_out = compute_P_out(self.S, objective, self.model.reactions) 
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

    
    def valid_medium_file(self, medium_name):
        if medium_name == '':
            sys.exit('Give a training file or a appropriate medium_name.')
        if not os.path.isfile(medium_name+'.csv'):
            print(medium_name)
            sys.exit('medium file not found')
        return medium_name
    