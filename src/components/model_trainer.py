import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('data','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                )
            models= {
                'RandomForest':RandomForestClassifier(),
                'DecisionTree':DecisionTreeClassifier(),
                'GradientBoosting':GradientBoostingClassifier(),
                'XGBoost':XGBClassifier(),
                'KNN':KNeighborsClassifier(),
                'CatBoost':CatBoostClassifier(verbose=False),
                'AdaBoost':AdaBoostClassifier(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            if best_model_score<0.75:
                raise CustomException('No Suitable Model found')
            logging.info(f"Best found mode on dataset {best_model_name}")

            best_model = models[best_model_name]
            print(model_report)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        


        except Exception as e:
            raise CustomException(e, sys)