import cobra
import numpy as np
import pandas as pd
from amn.dataset.metabolicDataset import MetabolicDataset
from amn.run_cobra import create_random_medium_cobra, run_cobra
from amn.tools import compute_P_in, compute_V2M, compute_M2V


class ExperimentalDataset(MetabolicDataset):
        
    def __init__(self, experimental_file='',**kwargs):
        MetabolicDataset.__init__(self, **kwargs)

        self.experimental_file = experimental_file

        # get X and Y from the medium file
        df_medium = pd.read_csv(self.experimental_file, header=0)
        medium_column = [c for c in df_medium.columns if "GR" not in c] 
        growth_rate_column = [c for c in df_medium.columns if "GR" in c]
        self.medium = medium_column
        self.X = df_medium[medium_column].values
        self.Y= df_medium[growth_rate_column].values
        self.method_generation ="EXPERIMENTAL"


        # compute matrices used in AMN
        self.S = np.asarray(cobra.util.array.create_stoichiometric_matrix(self.model))
        self.P_in = compute_P_in(self.S, self.medium, self.model.reactions)
