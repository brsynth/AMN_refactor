import cobra
import numpy as np
import pandas as pd
from metabolicDataset import MetabolicDataset
from run_cobra import create_random_medium_cobra, run_cobra
from tools import compute_P_in, compute_V2M, compute_M2V

class ExperimentalDataset(MetabolicDataset):


        
    def __init__(self, **kwargs):
        MetabolicDataset.__init__(self, **kwargs)

        # get X and Y from the medium file
        df_medium = pd.read_csv(self.medium_file, header=0)
        medium_column = [c for c in df_medium.columns if "GR" not in c] ## Not satisfying ! Before it was the last columns with a given number of medium columns...
        growth_rate_column = [c for c in df_medium.columns if "GR" in c]
        self.medium = medium_column
        self.X = df_medium[medium_column].values
        self.Y= df_medium[growth_rate_column].values
        self.method_generation ="EXPERIMENTAL"


        # compute matrices used in AMN
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.V2M = compute_V2M(self.S) ## Could be donne in the Neural model !
        self.M2V = compute_M2V(self.S) ## Could be donne in the Neural model !
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)