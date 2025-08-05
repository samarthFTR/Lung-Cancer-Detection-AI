import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("data","train.csv")
    test_data_path: str = os.path.join("data","test.csv")
    raw_data_path: str = os.path.join("data","raw.csv")

class DataIngestion: 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
         logging.info("Entered the date ingestion method")
         try:
             df = pd.read_csv('src\\dataset\\cancer patient data sets.csv') #read dataset

             logging.info("Read the dataset as dataframe")
             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
             df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #to save raw data


             logging.info("Train test split initiated")
             train_set,test_set=train_test_split(df,test_size=0.25,random_state=50) #split data into test and train
             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
             logging.info("Ingestion of the data is completed")
             return(
                 self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path,
             )
         except Exception as e:
             raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    Data_Transformation=DataTransformation()
    train_arr,test_arr,_=Data_Transformation.initiate_data_transformation(train_data,test_data)

    modeltrain = ModelTrainer()
    print(modeltrain.initiate_model_trainer(test_array=test_arr,train_array=train_arr))