import json
import pandas as pd
import tensorflow as tf

from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from aMNWtModel import AMNWtModel
from tools import MaxScaler


def run_model(model_name,model_parameters_file,cross_validation=False,verbose=False):
    
    with open(model_parameters_file, 'r') as f:
        data = json.load(f)
    params = data[model_name]

    # Dataset and architecture information
    AMN_model = AMNWtModel(**params["model_parameters"])
    
    # Preprocessing dataset
    AMN_model.train_test_split(test_size=params["preprocessing_parameters"]["test_ratio"],
                           random_state=params["preprocessing_parameters"]["seed"])
    if params["preprocessing_parameters"]["scaler"] == "maxscaler":
        scaler= MaxScaler()
        AMN_model.preprocess(scaler)
    AMN_model.preprocessing_for_specific_model()


    if cross_validation : 
        estimator= KerasRegressor(build_fn=AMN_model.build_model, 
                                  epochs=params["training_parameters"]["epochs"], 
                                  batch_size=params["training_parameters"]["batch_size"], 
                                  verbose=0)

        scoring = {"loss_constraint":make_scorer(AMN_model.loss_constraint),
                   "mse":make_scorer(AMN_model.mse),
                   "R2":make_scorer(AMN_model.R2),
                   }

        # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # fit_params = {'callbacks': [callback]}
        fit_params = {}

        ## Here we could use another seed
        kfold= KFold(n_splits=5,shuffle=True, random_state=params["preprocessing_parameters"]["seed"])
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
        # df.to_csv("Result/" + model_name)
        
    else:
        # build and run the model
        AMNWt_model = AMN_model.build_model()
        history = AMNWt_model.fit(AMN_model.X_train, 
                                  AMN_model.Y_train, 
                                  epochs=params["training_parameters"]["epochs"], 
                                  batch_size=params["training_parameters"]["batch_size"], 
                                  verbose=0)

        print("R2 :", AMN_model.R2(AMN_model.Y_train, AMNWt_model.predict(AMN_model.X_train)))
        print("Q2 :", AMN_model.R2(AMN_model.Y_test, AMNWt_model.predict(AMN_model.X_test)))



if __name__ == "__main__":

    model_parameters_file = './Model_parameters.json'
    model = "e_coli_core_UB"
    model = "iML1515_EXP_UB"
    model = "IJN1463_10_UB"
    model = "IJN1463_EXP_UB_Anne"
    model = "biolog_iML1515_EXP_UB"
    cross_validation = True

    run_model(model, model_parameters_file, cross_validation)

    

