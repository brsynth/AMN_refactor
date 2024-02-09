import cobra
import numpy as np
import pandas as pd
from amn.dataset import MetabolicDataset
from amn.run_cobra import create_random_medium_cobra, run_cobra
from amn.tools import compute_P_in


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


        if self.verbose:
            print('medium:',self.medium)
            print('level_med:',self.level_med)
            print('value_medium:',self.value_medium)
            print('ratio_medium:',self.ratio_medium)

        # Default objective for cobra is the biomass
        self.cobra_objective = cobra_objective if cobra_objective else\
                         [self.model.objective.to_json()["expression"]['args'][0]['args'][1]["name"]]
        
        if experimental_file:
            df_medium = pd.read_csv(experimental_file, header=0)
            medium_column = [c for c in df_medium.columns if "GR" not in c] 
            X = df_medium[medium_column].values

            for i in range(X.shape[0]): 
                medium_variation = []
                for j in range(X.shape[1]):
                    if self.level_med[j] > 1 and X[i,j] > 0:
                        medium_variation.append(self.medium[j])

                self.get_simulated_data(sample_size=sample_size,
                                        varmed=medium_variation, 
                                        add_to_existing_data = i, 
                                        verbose=self.verbose) 
        else : 
            self.get_simulated_data(sample_size=sample_size,
                                    verbose=self.verbose)

        # compute matrices used in AMN
        self.S = cobra.util.array.create_stoichiometric_matrix(self.model)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)


    def get_simulated_data(self, sample_size=100, varmed=[], add_to_existing_data =False, reduce=False,verbose=False):
        """
        Generate a training set using cobra. The training set is store in the X and Y attributes.
        """

        X,Y = [],[]
        for i in range(sample_size):
            if verbose: print('sample:',i)
            inf = create_random_medium_cobra(self.model, 
                                             self.cobra_objective, 
                                             self.medium,
                                             varmed, 
                                             self.level_med, 
                                             self.value_medium.copy(), 
                                             self.ratio_medium,
                                             method=self.method)
            X.append([inf[m] for m in self.medium])
            try:
                out,_ = run_cobra(self.model,self.cobra_objective,inf,method=self.method,verbose=verbose)
            except:
                print('Cobra cannot be run start again')
                continue
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


    def reduce_model(self):
    # Remove all reactions not in medium having a zero flux
    # Input: the model, the medium, the flux vector (a 2D array)
    # Output: the reduce model
        
        model = self.model
        medium = self.medium
        flux = self.Y
        
        measured_reaction = [] if len(self.measure) == len(self.reactions) \
        else self.measure

        # Collect reaction to be removed
        remove_reaction=[]
        remove_id = []
        for j in range(flux.shape[1]):
            if np.count_nonzero(flux[:,j]) == 0:
                # Check if reaction is not in medium of measured
                reaction_id = model.reactions[j].id
                if reaction_id not in medium and reaction_id not in measured_reaction:
                    remove_reaction.append(model.reactions[j])

                    index = [r.id for r in model.reactions].index(reaction_id)
                    remove_id.append(index)
        

        # Actual deletion
        model.remove_reactions(remove_reaction)
        cobra.manipulation.delete.prune_unused_reactions(model)


        # remove reaction columns from Y
        self.Y = np.delete(self.Y,remove_id,1)

        remove_metabolite = []
        for m in model.metabolites:
            if len(m.reactions) == 0:
                remove_metabolite.append(m)
        
        for m in remove_metabolite:
            model.remove_metabolites(m)

        cobra.manipulation.delete.prune_unused_metabolites(model)
        print('reduced numbers of metabolites and reactions:',
              len(model.metabolites), len(model.reactions))
        
        self.reactions = [r.id for r in list(model.reactions)]
        self.S = cobra.util.array.create_stoichiometric_matrix(self.model)
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)

