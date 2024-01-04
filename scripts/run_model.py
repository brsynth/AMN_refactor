import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from amn.model import AMNWtModel
from amn.model import LinearModel
from amn.tools import MaxScaler, threshold_percentage_max
from amn.visualize import plot_regression, plot_classification


def run_model(model_name,
              model_parameters_file,
              data_dir,
              cross_validation=False,
              save_folder="",
              verbose=0,
              add_random_state=None,
              show_figure=False):
    
    print(f'Run {add_random_state +1} AMNWt on {model_name} ')
    
    with open(model_parameters_file, 'r') as f:
        data = json.load(f)
    params = data[model_name]

    if save_folder:
        result_folder = save_folder + model_name +"/"
        Path(result_folder).mkdir(parents=True, exist_ok=True)

        with open(result_folder + "params.json", 'w') as f:
            json.dump(params, f)
    

    params["model_parameters"]["dataset_file"] = data_dir + params["model_parameters"]["dataset_file"]


    # Dataset and architecture information
    # AMN_model = AMNWtModel(**params["model_parameters"])
    AMN_model = LinearModel(**params["model_parameters"])

    random_state = params["preprocessing_parameters"]["seed"] + add_random_state

    # Preprocessing dataset
    AMN_model.train_test_split(test_size=params["preprocessing_parameters"]["test_ratio"],
                               random_state=random_state)
    if params["preprocessing_parameters"]["scaler"] == "maxscaler":
        scaler= MaxScaler()
        AMN_model.preprocess(scaler)
    AMN_model.preprocessing_for_specific_model()


    if cross_validation : 
        estimator= KerasRegressor(build_fn=AMN_model.build_model, 
                                  epochs=params["training_parameters"]["epochs"], 
                                  batch_size=params["training_parameters"]["batch_size"], 
                                  verbose=verbose)

        scoring = {"loss_constraint":make_scorer(AMN_model.loss_constraint),
                   "mse":make_scorer(AMN_model.mse),
                   "R2":make_scorer(AMN_model.R2),
                   }

        # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # fit_params = {'callbacks': [callback]}
        fit_params = {}

        ## Here we could use another seed
        kfold= KFold(n_splits=5,shuffle=True, random_state=random_state)
        results=cross_validate(estimator, 
                               AMN_model.X_train, 
                               AMN_model.Y_train, 
                               cv=kfold, 
                               n_jobs=5, 
                               scoring=scoring, 
                               fit_params=fit_params,
                               return_train_score=True)
            
        df = pd.DataFrame(results)
        print(df.describe())

        if save_folder:
            if add_random_state:
                df.describe().to_csv(result_folder + "CV_describe_" + str(add_random_state)+ ".csv")
                df.to_csv(result_folder + "CV_"+ str(add_random_state) +".csv")
            else:
                df.describe().to_csv(result_folder + "CV_describe" + ".csv")
                df.to_csv(result_folder + "CV.csv")

        
    else:
        # build and run the model
        AMNWt_model = AMN_model.build_model()
        history = AMNWt_model.fit(AMN_model.X_train, 
                                  AMN_model.Y_train, 
                                  epochs=params["training_parameters"]["epochs"], 
                                  batch_size=params["training_parameters"]["batch_size"], 
                                  verbose=verbose)
        
        pred_test =  AMNWt_model.predict(AMN_model.X_test)
        P_out = tf.convert_to_tensor(np.float32(AMN_model.P_out))   
        PRED = tf.linalg.matmul(pred_test[:,:AMN_model.S.shape[1]], tf.transpose(P_out)) 
        
        # The scores are computed on several dimension with the respect of the mechanistic constraints
        R_2 = AMN_model.R2(AMN_model.Y_train, AMNWt_model.predict(AMN_model.X_train))
        Q_2 = AMN_model.R2(AMN_model.Y_test, pred_test)

        print("R2 :", R_2)
        print("Q2 :", Q_2)

        if save_folder:
            if add_random_state:
                plot_file = result_folder + model_name + "_" + str(add_random_state) +".png"
            else:
                plot_file = result_folder + model_name +".png"
        else:
            plot_file=""
                

        if params["visualization_parameters"]["classification"]:
            TRUE = threshold_percentage_max(AMN_model.Y_test[:,0],
                                            params["visualization_parameters"]["threshold_max_percentage"]) 
            plot_classification(
                PRED, 
                TRUE,
                plot_file,
                show_figure
                )
        else:
            TRUE = AMN_model.Y_test[:,0]
            plot_regression(
            PRED, 
            TRUE, 
            "Measured growth rate (."+ r'$\mathregular{hr^{-1}}$' +")",
            "Predicted growth rate (."+ r'$\mathregular{hr^{-1}}$' +")",
            "Q²="+str(round(Q_2, 2)),
            plot_file,
            show_figure)



if __name__ == "__main__":

    model_parameters_file = "../config/run_model.json"
    data_dir = "../data"
    save_folder = "../results/"
    verbose = 2#"auto"
    cross_validation = True
    # cross_validation = False
    show_figures = False
    add_random_state = 0

    model_parameters_file = "../config/run_model_linear.json"
    save_folder = "../results_linear/"
    run_all = False
    # run_all = True

    all = ["e_coli_core_UB_100",
           "e_coli_core_UB",
           "e_coli_core_EB",
           "IJN1463_10_UB",
           "IJN1463_EXP_UB",
           "iML1515_EXP_UB", 
           "iML1515_UB",
           "biolog_iML1515_EXP_UB",
           "biolog_iML1515_medium_UB",
           "iML1515_EXP_2_UB"]

    model = "e_coli_core_UB_100"
    # model = "e_coli_core_UB"
    # model = "e_coli_core_EB"
    # model = "IJN1463_10_UB"
    # model = "IJN1463_EXP_UB"
    model = "iML1515_EXP_UB"
    # model = "iML1515_UB"
    # model = "biolog_iML1515_EXP_UB"
    # model = "biolog_iML1515_medium_UB"
    # model = "iML1515_EXP_2_UB"


    if run_all:
        for model in all:
            run_model(model,
                      model_parameters_file,
                      data_dir,
                      cross_validation,
                      save_folder,
                      verbose,
                      add_random_state,
                      show_figures)
    else:
        run_model(model,
                  model_parameters_file,
                  data_dir,
                  cross_validation,
                  save_folder,
                  verbose,
                  add_random_state,
                  show_figures)

    

    
    
    # Run the model several times, changing the seed for train/test split. (Explores influence of this seed on score for small datasets.)
    # nb_run = 1
    # for i in range(nb_run):
        # run_model(model,
        # model_parameters_file,
        # data_dir,cross_validation,
        # save_folder,
        # verbose,
        # i,
        # show_figures)







    

