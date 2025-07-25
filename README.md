---

# Lung Cancer Detection using AI/ML

A complete machine learning pipeline to assess lung cancer risk from patient data using AI. The system includes a modern web dashboard styled like iOS/Health app, and a backend powered by Flask for real-time predictions.

---

## Overview

This project covers the entire lifecycle of an ML solution:

* Data ingestion and preprocessing
* Dimensionality reduction with PCA
* Multiple model training and evaluation
* Saving the best model and preprocessor
* Web deployment via Flask with an intuitive UI

---

## Project Structure

```
Lung-Cancer-Detection-AI/
├── app.py                          # Flask backend app
├── index.html                      # Web UI (templates/index.html)
├── cancer patient data sets.csv    # Original dataset
├── data/                           # Stores raw, train/test splits, model, preprocessor
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── requirements.txt                # All dependencies
├── src/
│   ├── components/                 # Core ML pipeline scripts
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py     # For inference
│   ├── exception.py, logger.py, utils.py
├── README.md                       # You're here!
```

---

## Dataset Info

* **File:** `cancer patient data sets.csv`
* **Target Column:** `Level` (Low, Medium, High risk)
* **Features:** 23 inputs including age, gender, smoking history, environmental and genetic risk, and symptoms like chest pain, fatigue, coughing, etc.

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/components/data_ingestion.py
```

This:

* Loads the raw dataset
* Splits into train/test
* Applies preprocessing
* Trains multiple models
* Saves best model & preprocessor into `data/`

### 3. Launch the Web App

```bash
python app.py
```

Visit `http://localhost:5000` in your browser to access the dashboard.

---

## Web App Features

* Clean, modern UI (Glassmorphism & iOS-inspired design)
* Interactive sliders for severity input
* Radio buttons for gender
* Real-time prediction with immediate visual feedback
* Risk-level interpretation:

  * **High Risk:** Seek urgent medical help
  * **Medium Risk:** Book a doctor consultation
  * **Low Risk:** Maintain healthy lifestyle

---

## Model Training Details

* Algorithms used:

  * Random Forest, XGBoost, CatBoost, KNN, Gradient Boosting, AdaBoost, Decision Tree
* Hyperparameter tuning via grid search
* Label encoding for categorical targets
* PCA for feature dimensionality reduction
* Accuracy-based model selection

---

## Tech Stack

| Layer         | Tools Used                                 |
| ------------- | ------------------------------------------ |
| Frontend      | HTML5, CSS (custom), responsive UI         |
| Backend       | Python, Flask                              |
| ML Libraries  | scikit-learn, XGBoost, CatBoost, pandas    |
| Visualization | Matplotlib , Seaborn                       |
| Deployment    | Localhost, easy to deploy on Render/HF     |

---

## Future Improvements

* Add user authentication
* Host live on Render or Hugging Face Spaces
* Integrate SHAP or LIME for explainable AI
* Log user input and predictions for auditing
* Add visual result charts

---

