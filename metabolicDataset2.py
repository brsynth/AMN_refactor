import os
import sys
import cobra
import numpy as np
import pandas as pd
from run_cobra import create_random_medium_cobra, run_cobra
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V


# import pandas

## This function must be replaced everywhere by L.index(name) !!!!
# Cobra utilities and stoichiometric derived matrices
def get_index_from_id(name,L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == name:
            return i
    return -1



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
            self.load(training_file)
            return


        
        # self.check_medium_name(medium_name)
        # self.medium_name = medium_name # medium file
        self.medium_name = self.valid_medium_file(medium_name)
        self.medium_bound = medium_bound # EB or UB
        self.method = method

        ## Explain reduce !
        self.reduce = False
        # self.all_matrices = True

        self.cobra_name = self.valid_cobra_file(cobra_name) # model cobra file
        self.model = cobra.io.read_sbml_model(cobra_name+'.xml')
    

    
    



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


    def save(self, filename, reduce=False, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose)

        ## not here !
        # Recompute matrices
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
        self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)
        
        ## Have to do a equivalent of the set matrices for LP. ,
        ## It's strange that it is always done, even if the model is not LP
        set_matrices_LP(self, self.medium_bound, self.X, self.S,
                             self.P_in, self.medium, self.objective)
        
        
        # save cobra file
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save parameters
        
        np.savez_compressed(filename, 
                            cobra_name = filename,
                            reduce = self.reduce,
                            medium_name = self.medium_name,
                            medium_bound = self.medium_bound,
                            objective =self.objective,
                            method = self.method,
                            size = self.size,
                            medium = self.medium,
                            level_med = self.level_med, 
                            value_medium = self.value_medium, 
                            ratio_medium = self.ratio_medium, 
                            measure = self.measure,
                            S = self.S,
                            P_in = self.P_in,
                            P_out = self.P_out,
                            V2M = self.V2M,
                            M2V = self.M2V,
                            X = self.X,
                            Y = self.Y,
                            S_int = self.model.S_int,
                            S_ext = selfmodel.S_ext,
                            Q = self.Q,
                            P = self.P,
                            b_int = self.b_int,
                            b_ext = self.b_ext,
                            Sb = self.Sb,
                            c = self.c)


    def save(self, filename, reduce=False, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose)

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
        # get_matrices_LP(self.model, self.medium_bound, self.X, self.S,
                            #  self.Pin, self.medium, self.objective)
        


        # save cobra file
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save parameters
        np.savez_compressed(filename, 
                            cobraname = filename,
                            reduce = self.reduce,
                            mediumname = self.medium_name,
                            mediumbound = self.medium_bound,
                            objective =self.objective,
                            method = self.method,
                            size = self.size,
                            medium = self.medium,
                            levmed = self.level_med, 
                            valmed = self.value_medium, 
                            ratmed = self.ratio_medium, 
                            measure = self.measure,
                            S = self.S,
                            Pin = self.Pin,
                            Pout = self.Pout,
                            V2M = self.V2M,
                            M2V = self.M2V,
                            X = self.X,
                            Y = self.Y,
                            S_int = self.S_int,
                            S_ext = self.S_ext,
                            Q = self.Q,
                            P = self.P,
                            b_int = self.b_int,
                            b_ext = self.b_ext,
                            Sb = self.Sb,
                            c = self.c)
        



    def load(self, file_name):
        # load parameters (npz format)
        

        self.check_file_name_npz(file_name)
        loaded = np.load(file_name+'.npz')


        ## Could do better !!!
        self.cobra_name = str(loaded['cobraname'])
        self.reduce = str(loaded['reduce'])
        self.reduce = True if self.reduce == 'True' else False
        self.medium_name = str(loaded['mediumname'])
        self.medium_bound = str(loaded['mediumbound'])
        self.objective = loaded['objective']
        self.method = str(loaded['method'])
        self.size = loaded['size']
        self.medium = loaded['medium']
        self.level_med = loaded['levmed'] #
        self.value_medium = loaded['valmed'] #
        self.ratio_medium = loaded['ratmed'] #
        self.measure = loaded['measure']
        self.S = loaded['S']
        self.P_in = loaded['Pin']
        self.P_out = loaded['Pout']
        self.V2M = loaded['V2M']
        self.M2V = loaded['M2V']
        self.X = loaded['X']
        self.Y = loaded['Y']
        self.S_int = loaded['S_int']
        self.S_ext = loaded['S_ext']
        self.Q = loaded['Q']
        self.P = loaded['P']
        self.b_int = loaded['b_int']
        self.b_ext = loaded['b_ext']
        self.Sb = loaded['Sb']
        self.c = loaded['c']
        self.all_matrices = True
        ## Could be more precise about what we need in model !
        self.model = cobra.io.read_sbml_model(self.cobra_name+'.xml')


        self.Y_all = loaded['Y']


    def check_file_name_npz(self, file_name):
        if not os.path.isfile(file_name+'.npz'):
            print(file_name+'.npz')
            sys.exit('file not found')



    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'wb')
        print('model file name:',self.cobra_name)
        print('reduced model:',self.reduce)
        print('medium file name:',self.medium_name)
        print('medium bound:',self.medium_bound)
        print('list of reactions in objective:',self.objective)
        print('method:',self.method)
        print('training size:',self.size)
        print('list of medium reactions:',len(self.medium))
        print('list of medium levels:',len(self.level_med))
        print('list of medium values:',len(self.value_medium))
        print('ratio of variable medium turned on:',self.ratio_medium)
        print('list of measured reactions:',len(self.measure))
        print('Stoichiometric matrix',self.S.shape)
        print('Boundary matrix from reactions to medium:',self.P_in.shape)
        print('Measurement matrix from reaction to measures:',self.P_out.shape)
        print('Reaction to metabolite matrix:',self.V2M.shape)
        print('Metabolite to reaction matrix:',self.M2V.shape)
        print('Training set X:',self.X.shape)
        print('Training set Y:',self.Y.shape)
        if self.all_matrices:
            print('S_int matrix', self.S_int.shape)
            print('S_ext matrix', self.S_ext.shape)
            print('Q matrix', self.Q.shape)
            print('P matrix', self.P.shape)
            print('b_int vector', self.b_int.shape)
            print('b_ext vector', self.b_ext.shape)
            print('Sb matrix', self.Sb.shape)
            print('c vector', self.c.shape)
        if filename != '':
            sys.stdout.close()
        



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
    