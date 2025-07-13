import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        #report_train = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X=X_train,y=y_train)

            model.set_params(**gs.best_params_)           
            model.fit(X_train,y_train) #training the model
            
            y_pred = model.predict(X_test)
            #y_pred_train = model.predict(X_train)
            model_score = accuracy_score(y_true=y_test,y_pred=y_pred) #model evaluation
            #model_score_train = accuracy_score(y_true=y_train,y_pred=y_pred_train)

            report[list(models.keys())[i]]=model_score #storing score
            #report_train[list(models.keys())[i]]=model_score_train
        return report
    except Exception as e:
        raise CustomException(e, sys)