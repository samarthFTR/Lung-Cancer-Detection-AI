import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
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
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train) #training the model

            y_pred = model.predict(X_test)

            model_score = r2_score(y_true=y_test,y_pred=y_pred) #model evaluation
            report[list(models.keys())[i]]=model_score #storing score
        return report
    except Exception as e:
        raise CustomException(e, sys)