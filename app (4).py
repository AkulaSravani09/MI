#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and imputer
model = joblib.load("myocardial_model.pkl")
imputer = joblib.load("imputer.pkl")

# Define feature descriptions for the form
features_info = [
    ("AGE", "0-120", "Age of the patient"),
    ("SEX", "0 (Female), 1 (Male)", "Patient's gender"),
    ("SIM_GIPERT", "0 or 1", "History of hypertension"),
    ("STENOK_AN", "0 or 1", "History of angina"),
    ("FK_STENOK", "1-4", "Functional class of angina"),
    ("IBS_POST", "0 or 1", "History of past ischemic heart disease"),
    ("IBS_NASL", "0 or 1", "Family history of ischemic heart disease"),
    ("K_BLOOD", "2.5-5.5", "Potassium in blood"),
    ("L_BLOOD", "3.0-11.0", "Leukocytes in blood"),
    ("ROE", "1-50", "Erythrocyte Sedimentation Rate"),
    ("S_AD_KBRIG", "80-200", "Systolic blood pressure"),
    ("D_AD_KBRIG", "50-120", "Diastolic blood pressure"),
    ("GIPO_K", "0 or 1", "Hypokalemia presence"),
    ("GIPER_NA", "0 or 1", "Hypernatremia presence"),
]

@app.route('/')
def home():
    return render_template("home.html", features_info=features_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[feature[0]]) for feature in features_info]
        
        input_data = np.array(features).reshape(1, -1)
        input_data = imputer.transform(input_data)
        
        prediction = model.predict(input_data)[0]
        result = "High Risk of Myocardial Infarction" if prediction == 1 else "Low Risk of Myocardial Infarction"

        return render_template("result.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

