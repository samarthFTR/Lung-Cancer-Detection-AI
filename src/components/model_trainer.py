import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
            param_grid = {
                'RandomForest': {
                    'n_estimators': [10, 20],
                    'max_depth': [2, 3, 5],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10],
                    'max_features': ['sqrt']
                },
                'DecisionTree': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 10],
                    'subsample': [0.6, 0.8, 1.0]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 10],
                    'subsample': [0.7, 1.0],
                    'colsample_bytree': [0.7, 1.0]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                'CatBoost': {
                    'depth': [4, 6, 10],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200],
                    'l2_leaf_reg': [1, 3, 5]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=param_grid)
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
            y_pred = best_model.predict(X_train)
            train_score = accuracy_score(y_pred=y_pred,y_true=y_train)
            return train_score
        

        except Exception as e:
            raise CustomException(e, sys)