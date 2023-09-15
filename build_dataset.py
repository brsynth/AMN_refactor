import os
import json
import numpy as np
from simulatedDataset import SimulatedDataset
from experimentalDataset import ExperimentalDataset
from metabolicDataset import MetabolicDataset

def generate_dataset(dataset,generation_parameters_file,verbose=False):
    
    with open(generation_parameters_file, 'r') as f:
        data = json.load(f)
    params = data[dataset]
    if params["method_generation"] == "simulated":
        dataset = SimulatedDataset(**params["MetabolicDataset_parameters"])
    elif params["method_generation"] ==  "experimental":
        dataset = ExperimentalDataset(**params["MetabolicDataset_parameters"])

    saving_params = params["saving_parameters"] 
    dataset.save(**saving_params)

    if verbose : 
        dataset = MetabolicDataset(dataset_file=os.path.join(saving_params["dataset_dir"],
                                                             saving_params["dataset_name"]+'.npz'))
        dataset.printout()



if __name__ == "__main__":

    seed = 10
    generation_parameters_file = './Dataset_generation_parameters.json'
    dataset = "iML1515_EXP_UB"
    dataset = "biolog_iML1515_EXP_UB"


    np.random.seed(seed=10)  
    generate_dataset(dataset,generation_parameters_file,verbose=False)

