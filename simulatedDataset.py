import cobra
import numpy as np
import pandas as pd
import cobra.manipulation as manip
from run_cobra import create_random_medium_cobra, run_cobra
from metabolicDataset import MetabolicDataset
from tools import compute_P_in, compute_V2M, compute_M2V

class SimulatedDataset(MetabolicDataset):


    def __init__(self, experimental_file='', sample_size=100, cobra_objective=[], **kwargs):
        MetabolicDataset.__init__(self, **kwargs)

        # get parameter for variation on medium simulation 
        df_medium = pd.read_csv(self.medium_file,index_col="name")
        self.medium = df_medium.columns.to_list()
        self.level_med = df_medium.loc["level"].values
        self.value_medium = df_medium.loc["max_value"].values
        self.ratio_medium = df_medium.loc["ratio_drawing"][0]
        self.method_generation ="SIMULATED"


        # Default objective for cobra is the biomass
        self.cobra_objective = cobra_objective if cobra_objective else\
                         [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        
        if experimental_file:
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

        # compute matrices used in AMN
        self.S = cobra.util.array.create_stoichiometric_matrix(self.model)
        self.V2M = compute_V2M(self.S)
        self.M2V = compute_M2V(self.S)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
        # self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)


    def save(self, dataset_dir, dataset_name, verbose=False, reduce=False):

        if reduce:
            self.reduce_and_run(verbose=verbose) 
            
            self.S = cobra.util.array.create_stoichiometric_matrix(self.model)
            self.V2M = compute_V2M(self.S)
            self.M2V = compute_M2V(self.S)
            self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
            # self.P_out = compute_P_out(self.S, self.measure, self.model.reactions)
        
        MetabolicDataset.save(self,dataset_dir, dataset_name, verbose=verbose)


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
                inf = create_random_medium_cobra(self.model, self.cobra_objective, 
                                         self.medium, self.medium_bound,
                                         varmed, self.level_med, self.value_medium.copy(), self.ratio_medium,
                                         method=self.method,verbose=verbose)
            
            X.append([inf[m] for m in self.medium])
            out,_ = run_cobra(self.model,self.cobra_objective,inf,method=self.method,verbose=verbose)
            Y.append(list(out.values()))

        X = np.array(X)
        Y = np.array(Y)

        # In case medium_bound is 'EB' replace X[i] by Y[i] for i in medium
        if self.medium_bound == 'EB':
            for i, reaction_id in enumerate(self.medium):
                medium_index = self.model.reactions.index(reaction_id)
                X[:,i] = Y[:,medium_index]
        
        if add_to_existing_data:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
        else:
            self.X, self.Y = X, Y
        self.size = self.X.shape[0]


    def reduce_and_run(self,verbose=False):
        # reduce a model recompute matrices and rerun cobra
        # with the provided training set
        measure = [] if len(self.measure) == len(self.reactions) \
        else self.measure

        self.model = self.reduce_model(self.model, self.medium, measure,
                                  self.Y, verbose=verbose)
        
        self.measure = [r for r in self.reactions] \
        if measure == [] else measure

        self.get_simulated_data(sample_size=self.size, reduce=True, verbose=verbose)


    def reduce_model(self, model, medium, measure, flux, verbose=False):
    # Remove all reactions not in medium having a zero flux
    # Input: the model, the medium, the flux vector (a 2D array)
    # Output: the reduce model

    # Collect reaction to be removed
        remove=[]
        for j in range(flux.shape[1]):
            if np.count_nonzero(flux[:,j]) == 0 and model.reactions[j].id not in medium and model.reactions[j].id not in measure:
                remove.append(model.reactions[j])
                
        # Actual deletion
        model.remove_reactions(remove)
        manip.delete.prune_unused_reactions(model)
        for m in model.metabolites:
            if len(m.reactions) == 0:
                model.remove_metabolites(m)
        manip.delete.prune_unused_metabolites(model)
        print('reduced numbers of metabolites and reactions:',
              len(model.metabolites), len(model.reactions))
        
        self.reactions = [r.id for r in list(model.reactions)]
        return model