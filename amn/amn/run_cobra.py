import cobra
import math
import numpy as np
from sklearn.utils import shuffle

import optlang
optlang.glpk_interface.Configuration()

def run_cobra(model, objective, IN, method='FBA', verbose=False,
              objective_fraction=0.75, cobra_min_flux=1.0e-8):
    # Inputs:
    # - model
    # - objective: a list of reactions (only the two first are considered)
    # - IN: Initial values for all reaction fluxes
    # - method: FBA or pFBA
    # run FBA optimization to compute reaction fluxes on the provided model
    # set the medium using values in dictionary IN.
    # When 2 objectives are given one first maximize the first objective (obj1).
    # then one set the upper and lower bounds for that objective to
    # objective_fraction * obj1 (e.g. objective_fraction = 0.75) and maximize
    # for the second objective
    # Outputs:
    # - FLUX, the reaction fluxes computed by FBA for all reactions
    # - The value for the objective

    # set the medium and objective
    medium = model.medium

    # fix solver timeout
    model.solver.configuration = optlang.glpk_interface.Configuration(timeout=5, presolve='auto', lp_method='simplex')


    medini = medium.copy()

    for k in medium.keys(): # Reset the medium
        medium[k] = 0
    for k in IN.keys(): # Additional cmpds added to medium
        if k in medium.keys():
            medium[k] = float(IN[k])

    
    model.medium = medium

    # run FBA for primal objective
    model.objective = objective[0] 
    solution = cobra.flux_analysis.pfba(model) if method == 'pFBA' else model.optimize()
    solution_val = solution.fluxes[objective[0]]
    if verbose:
        print('primal objectif =', objective, method, solution_val)

    # run FBA for second objective
    # primal objective is set to a fraction of its value
    if len(objective) > 1:
        obj = model.reactions.get_by_id(objective[0])
        obj_lb, obj_ub = obj.lower_bound, obj.upper_bound
        obj.lower_bound = objective_fraction * solution_val
        obj.upper_bound = objective_fraction * solution_val
        model.objective = objective[1]
        solution = cobra.flux_analysis.pfba(model) \
        if method == 'pFBA' else model.optimize()
        solution_val = solution.fluxes[objective[1]]
        if verbose:
            print('second objectif =', objective, method, solution_val)

        # reset bounds and objective to intial values
        obj.lower_bound, obj.upper_bound = obj_lb, obj_ub
        model.objective = objective[0]

    # get the fluxes for all model reaction
    FLUX = IN.copy()
    for x in model.reactions:
        if x.id in FLUX.keys():
            FLUX[x.id] = solution.fluxes[x.id]
            if math.fabs(float(FLUX[x.id])) < cobra_min_flux: # !!!
                FLUX[x.id] = 0

    # Reset medium
    model.medium = medini


    return FLUX, solution_val




def create_random_medium_cobra(model,
                               objective,
                               medium, 
                               medium_variation, 
                               medium_level, 
                               medium_max_value, 
                               medium_ratio,
                               method='FBA', 
                               cobra_min_objective=1.0e-3,
                               max_iteration=5):
    """
    Generate a random input for cobra. We could run cobra several times to
    make sure that the objective is not too low.

        objective: the reaction fluxes to optimize
        medium: list of reaction fluxes in medium
        medium_variation: the medium reaction fluxes allowed to change (can be empty then medium_variation are drawn at random)
        medium_level: the number of level for a flux
        medium_max_value: the maximum value the flux can take
        medium_ratio: the ration of fluxes turned on
        method: the method used by Cobra
    """
    
    initial_medium = model.medium.copy()
    influx = {}

    for r in model.reactions:
        influx[r.id] = 0


    # n_var is the number of variable medium turned ON
    if medium_variation:
        n_var = len(medium_variation)
    else:
        level_greater_than_one = (np.array(medium_level)>1).sum()
        ## this if exist because the variable covers two different ideas !
        if medium_ratio <1 :
            x = np.random.binomial(level_greater_than_one, medium_ratio)
        else:
            x =  int(medium_ratio)
        n_var = max(x,1) # at least one



    # index of medium to give to cobra, minimal or variation
    minimal_index, all_variation_index = [], []
    for i, med in enumerate(medium):
        if medium_level[i] <= 1: 
            minimal_index.append(i)
        elif not(medium_variation) or med in medium_variation: 
            all_variation_index.append(i)

    # We run several times cobra, until the objective is minimal (objective > cobra_min_objective) 
    for i in range(max_iteration):
        # create random medium choosing X fluxes in varmed at random
        influx = {k: 0 for k in influx.keys()}
        model.medium = initial_medium 
        
        # Fill the influx dictionary with minimal values
        for m_index in minimal_index:
            ## Choose who carry the information !!!
            influx[medium[m_index]] = medium_max_value[m_index]
            model.medium[medium[m_index]] =  medium_max_value[m_index]

        # subset of variation among all possible variations
        variation_index = shuffle(np.array(all_variation_index),n_samples = n_var)
        for m_index in variation_index:
            # Choose one medium to change and get the new value
            max_level = medium_level[m_index]
            new_value = (len(medium_variation)+1) * np.random.randint(1,high=max_level) * medium_max_value[m_index]/(max_level-1)

            ## Choose who carry the information !!!
            influx[medium[m_index]] = new_value
            model.medium[medium[m_index]] = new_value

        
        try :
            _, obj = run_cobra(model, objective, influx,
                                   method=method, verbose=False)
        except:
            print('Cobra cannot be run start again')
            continue

        
        if obj < cobra_min_objective:
            print("obj < cobra_min")
            continue 
        break

    model.medium = initial_medium # reset medium

    return influx