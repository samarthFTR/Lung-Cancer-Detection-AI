from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer #to create pipeline
from sklearn.pipeline import Pipeline # for making pipeling
from sklearn.impute import SimpleImputer # empty data handelling
from sklearn.preprocessing import OneHotEncoder,StandardScaler #Encoding and standardiztion
import os

from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

        def get_data_tranformer_object(self): #for encoding and standardization
            try:
                pass
            except:
                pass
            