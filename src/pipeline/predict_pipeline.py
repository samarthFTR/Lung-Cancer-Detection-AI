import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'data/model.pkl'
            preprocessor_path = 'data/preprocessor.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
        self,Gender,Air_Pollution,Alcohol_use,
        Dust_Allergy,OccuPational_Hazards,Genetic_Risk,
        chronic_Lung_Disease,Balanced_Diet,Obesity,
        Smoking,PassiveSmoker,ChestPain,CoughingofBlood,
        Fatigue,WeightLoss,ShortnessofBreath,Wheezing,
        SwallowingDifficulty,ClubbingofFingerNails,FrequentCold,
        DryCough,Snoring,Age):
        self.Gender = Gender
        self.Air_Pollution = Air_Pollution
        self.Alcohol_use = Alcohol_use
        self.Dust_Allergy = Dust_Allergy
        self.OccuPational_Hazards = OccuPational_Hazards
        self.Genetic_Risk = Genetic_Risk
        self.chronic_Lung_Disease = chronic_Lung_Disease
        self.Balanced_Diet = Balanced_Diet
        self.Obesity = Obesity
        self.Smoking = Smoking
        self.PassiveSmoker = PassiveSmoker
        self.ChestPain = ChestPain
        self.CoughingofBlood = CoughingofBlood
        self.Fatigue = Fatigue
        self.WeightLoss = WeightLoss
        self.ShortnessofBreath = ShortnessofBreath
        self.Wheezing = Wheezing
        self.SwallowingDifficulty = SwallowingDifficulty
        self.ClubbingofFingerNails = ClubbingofFingerNails
        self.FrequentCold = FrequentCold
        self.DryCough = DryCough
        self.Snoring = Snoring
        self.Age = Age

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Air Pollution": [self.Air_Pollution],
                "Alcohol use": [self.Alcohol_use],
                "Dust Allergy": [self.Dust_Allergy],
                "OccuPational Hazards": [self.OccuPational_Hazards],
                "Genetic Risk": [self.Genetic_Risk],
                "chronic Lung Disease": [self.chronic_Lung_Disease],
                "Balanced Diet": [self.Balanced_Diet],
                "Obesity": [self.Obesity],
                "Smoking": [self.Smoking],
                "Passive Smoker": [self.PassiveSmoker],
                "Chest Pain": [self.ChestPain],
                "Coughing of Blood": [self.CoughingofBlood],
                "Fatigue": [self.Fatigue],
                "Weight Loss": [self.WeightLoss],
                "Shortness of Breath": [self.ShortnessofBreath],
                "Wheezing": [self.Wheezing],
                "Swallowing Difficulty": [self.SwallowingDifficulty],
                "Clubbing of Finger Nails": [self.ClubbingofFingerNails],
                "Frequent Cold": [self.FrequentCold],
                "Dry Cough": [self.DryCough],
                "Snoring": [self.Snoring]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)