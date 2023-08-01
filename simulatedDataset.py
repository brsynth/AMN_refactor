import cobra
import numpy as np
import pandas as pd
from metabolicDataset import MetabolicDataset
from run_cobra import create_random_medium_cobra, run_cobra
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V

class SimulatedDataset(MetabolicDataset):


    def __init__(self, experimental_file='', sample_size=100, **kwargs):
        MetabolicDataset.__init__(self, **kwargs)


        # get parameter for variation on medium simulation 
        df_medium = pd.read_csv(self.medium_file,index_col="name")
        self.medium = df_medium.columns.to_list()
        self.level_med = df_medium.loc["level"].values
        self.value_medium = df_medium.loc["max_value"].values
        self.ratio_medium = df_medium.loc["ratio_drawing"][0]

        if experimental_file:
            # get X and Y from the medium file
            df_medium = pd.read_csv(experimental_file, header=0)
            medium_column = [c for c in df_medium.columns if "GR" not in c] ## Not satisfying ! Before it was the last columns with a given number of medium columns...
  
            X = df_medium[medium_column].values

            # Create varmed the list of variable medium based on experimental file
            medium_variation = {}
            for i in range(X.shape[0]):
                medium_variation[i] = []
                for j in range(X.shape[1]):
                    if self.level_med[j] > 1 and X[i,j] > 0:
                        medium_variation[i].append(self.medium[j])
            medium_variation = list(medium_variation.values())

            # Get a Cobra training set constrained by varmed
            for i in range(X.shape[0]): 
                self.get_simulated_data(sample_size=20, 
                                        varmed=medium_variation[i], 
                                        add_to_existing_data = i, 
                                        verbose=True) 
        else : 
            self.get_simulated_data(sample_size=sample_size,verbose=self.verbose)

        # compute matrices and objective vector for AMN
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
        self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)
        

    def get_simulated_data(self, sample_size=100, varmed=[], add_to_existing_data =False, reduce=False,verbose=False):
        """
        Generate a training set using cobra. The training set is store in the X and Y attributes.
        """
        X,Y = [],[]
        for i in range(sample_size):
            if verbose: print('sample:',i)

            # Cobra runs on reduce model where X is already known ## EXP !!!
            if reduce:
                inf = {r.id: 0 for r in self.model.reactions}
                for j in range(len(self.medium)):
                    inf[self.medium[j]] = self.X[i,j]
            else:
                inf = create_random_medium_cobra(self.model, self.objective, 
                                         self.medium, self.medium_bound,
                                         varmed, self.level_med, self.value_medium.copy(), self.ratio_medium,
                                         method=self.method,verbose=verbose)
            
            X.append([inf[m] for m in self.medium])
            out,_ = run_cobra(self.model,self.objective,inf,method=self.method,verbose=verbose)
            Y.append(list(out.values()))

        X = np.array(X)
        Y = np.array(Y)

        # In case medium_bound is 'EB' replace X[i] by Y[i] for i in medium
        if self.medium_bound == 'EB':
            for i, reaction_id in enumerate(self.medium):
                medium_index = self.model.reactions.index(reaction_id)
                X[:,i] = Y[:,medium_index]
            
        ## old version !  
        # In case 'get' is called several times
        # if self.X.shape[0] > 0 and reduce == False:
        if add_to_existing_data:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
        else:
            self.X, self.Y = X, Y
        self.size = self.X.shape[0]

