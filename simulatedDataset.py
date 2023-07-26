import cobra
import numpy as np
import pandas as pd
from metabolicDataset import MetabolicDataset
from run_cobra import create_random_medium_cobra, run_cobra
from tools import compute_P_in, compute_P_out, compute_V2M, compute_M2V

class SimulatedDataset(MetabolicDataset):


    def __init__(self, sample_size=100, **kwargs):
        MetabolicDataset.__init__(self, **kwargs)


        # get parameter for variation on medium simulation 
        df_medium = pd.read_csv(self.medium_name + ".csv",index_col="name")
        self.medium = df_medium.columns.to_list()
        self.level_med = df_medium.loc["level"].values
        self.value_medium = df_medium.loc["max_value"].values
        self.ratio_medium = df_medium.loc["ratio_drawing"][0]


        # compute matrices and objective vector for AMN
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
        self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)

        self.get_simulated_data(sample_size=sample_size,verbose=self.verbose)



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

