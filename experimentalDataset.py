from metabolicDataset2 import MetabolicDataset


import os
import sys
import cobra
import numpy as np
import pandas as pd
from run_cobra import create_random_medium_cobra, run_cobra
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V

class ExperimentalDataset(MetabolicDataset):


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
        
        MetabolicDataset.__init__(self,
                 training_file=training_file,
                 cobra_name=cobra_name,
                 medium_name=medium_name, 
                 medium_bound=medium_bound, 
                 medium_size=medium_size,
                 objective=objective, 
                 method=method,
                 measure=measure, 
                 verbose=False)
        
        # get X and Y from the medium file
        df_medium = pd.read_csv(medium_name + ".csv", header=0)
        medium_column = [c for c in df_medium.columns if "GR" not in c] ## Not satisfying ! Before it was the last columns with a given number of medium columns...
        growth_rate_column = [c for c in df_medium.columns if "GR" in c]
        self.medium = medium_column
        self.X = df_medium[medium_column].values
        self.Y= df_medium[growth_rate_column].values


        ## Quite useless variable in this context !
        self.size = self.Y.shape[0] ## What is the purpose of this parameter !!!
        self.level_med = [] ## useless here !
        self.value_medium = [] ##u seless here !
        self.ratio_medium = 0 ## useless here !



        ## correspond to exp or simulated ?
        if objective:
            self.objective = objective
        else:
            self.objective = [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        
        if measure:
            self.measure = measure
        else:
            self.measure = [r.id for r in self.model.reactions]




        if verbose:

            print('medium:',self.medium)
            print('level_med:',self.level_med)
            print('value_medium:',self.value_medium)
            print('ratio_medium:',self.ratio_medium)
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))

        # compute matrices and objective vector for AMN
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.V2M = compute_V2M(self.S) ## Could be donne in the Neural model !
        self.M2V = compute_M2V(self.S) ## Could be donne in the Neural model !
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
        self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)