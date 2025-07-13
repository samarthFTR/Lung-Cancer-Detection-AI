from dataclasses import dataclass #for function automation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer #to create pipeline
from sklearn.pipeline import Pipeline # for making pipeling
from sklearn.impute import SimpleImputer # empty data handelling
from sklearn.preprocessing import OneHotEncoder,StandardScaler #Encoding and standardiztion
from sklearn.preprocessing import LabelEncoder #To encode the target label
from sklearn.decomposition import PCA #For feature reduction
import os

import sys
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data','preprocessor.pkl')

class DataTransformation:
    """

    This function is used for data tranfromation

    """
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()

    def get_data_tranformer_object(self): #for encoding and standardization
        try:
            numerical_columns=['Age']
            categorical_columns=[
                'Gender', 'Air Pollution','Alcohol use',
                'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
                'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
                'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',
                'Weight Loss', 'Shortness of Breath', 'Wheezing',
                'Swallowing Difficulty', 'Clubbing of Finger Nails',
                'Frequent Cold','Dry Cough', 'Snoring'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scalar',StandardScaler()),
                    ('pca', PCA(n_components=0.85)),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder(sparse_output=False)),
                    ('scalar',StandardScaler()),
                    ('pca', PCA(n_components=0.85)),
                ]
            )
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')


            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
           train_df=pd.read_csv(train_path)
           test_df=pd.read_csv(test_path)

           logging.info('Reading test and train data completed')
           logging.info('Obtaining Preprocessing Object')
           
           preprocessor_obj = self.get_data_tranformer_object()
           
           target_column = 'Level'
           numerical_columns=['Age']

           input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
        #    target_feature_train_df=train_df[target_column]

           input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
        #    target_feature_test_df=test_df[target_column]

           label_encoder = LabelEncoder()
           target_feature_train_df = label_encoder.fit_transform(train_df[target_column])
           target_feature_test_df = label_encoder.transform(test_df[target_column])

           logging.info(
               f"Applying preprocessor object on training and test dataframe"
           )

           input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
           input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


           target_train_reshaped = np.array(target_feature_train_df).reshape(-1, 1)
           target_test_reshaped = np.array(target_feature_test_df).reshape(-1, 1)

           train_arr=np.concatenate((
               input_feature_train_arr, target_train_reshaped #np.array(target_feature_train_df)
           ), axis=1)
           test_arr=np.concatenate((
               input_feature_test_arr, target_test_reshaped #np.array(target_feature_test_df)
           ), axis=1)

           logging.info('saving preprocessing object.')

           save_object(
                file_path = self.DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            ) #to save pickle file

           return(
               train_arr,
               test_arr,
               self.DataTransformationConfig.preprocessor_obj_file_path,
           )
        except Exception as e:
            raise CustomException(e,sys)