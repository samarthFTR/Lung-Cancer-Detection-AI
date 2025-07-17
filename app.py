import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)
app = application

#route for homepage

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            Age=int(request.form.get('age')),  # Numerical data
            Gender=request.form.get('gender'),  # Categorical
            Air_Pollution=request.form.get('air_pollution'),  # Categorical
            Alcohol_use=request.form.get('alcohol_use'),  # Categorical
            Dust_Allergy=request.form.get('dust_allergy'),  # Categorical
            OccuPational_Hazards=request.form.get('occupational_hazards'),  # Categorical
            Genetic_Risk=request.form.get('genetic_risk'),  # Categorical
            chronic_Lung_Disease=request.form.get('chronic_lung_disease'),  # Categorical
            Balanced_Diet=request.form.get('balanced_diet'),  # Categorical
            Obesity=request.form.get('obesity'),  # Categorical
            Smoking=request.form.get('smoking'),  # Categorical
            PassiveSmoker=request.form.get('passive_smoker'),  # Categorical
            ChestPain=request.form.get('chest_pain'),  # Categorical
            CoughingofBlood=request.form.get('coughing_of_blood'),  # Categorical
            Fatigue=request.form.get('fatigue'),  # Categorical
            WeightLoss=request.form.get('weight_loss'),  # Categorical
            ShortnessofBreath=request.form.get('shortness_of_breath'),  # Categorical
            Wheezing=request.form.get('wheezing'),  # Categorical
            SwallowingDifficulty=request.form.get('swallowing_difficulty'),  # Categorical
            ClubbingofFingerNails=request.form.get('clubbing_of_finger_nails'),  # Categorical
            FrequentCold=request.form.get('frequent_cold'),  # Categorical
            DryCough=request.form.get('dry_cough'),  # Categorical
            Snoring=request.form.get('snoring')  # Categorical
        )
        pred_df=data.get_data_as_frame()
        print(pred_df)

        Predict_Pipeline=PredictPipeline()
        results = Predict_Pipeline.predict(pred_df)
        return render_template('index.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)