from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.logging.logger import logging

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # 1. Get inputs from form
            input_data = {
                'Age': float(request.form['age']),
                'Marital_Status': request.form['marital_status'],
                'Married_Premium_Discount': float(request.form['married_premium_discount']),
                'Prior_Insurance': request.form['prior_insurance'],
                'Claims_Severity': request.form['claims_severity'],  # Keep as string for ordinal encoder
                'Claims_Adjustment': float(request.form['claims_adjustment']),
                'Policy_Type': request.form['policy_type'],
                'Premium_Amount': float(request.form['premium_amount']),
                'Safe_Driver_Discount': float(request.form['safe_driver_discount']),
                'Multi_Policy_Discount': float(request.form['multi_policy_discount']),
                'Bundling_Discount': float(request.form['bundling_discount']),
                'Total_Discounts': float(request.form['total_discounts']),
                'Source_of_Lead': request.form['source_of_lead'],
                'Credit_Score': float(request.form['credit_score']),
                'Region': request.form['region'],
            }

            # 2. Convert to DataFrame (important: column names must match training data!)
            input_df = pd.DataFrame([input_data])

            logging.info("Raw input data: %s", input_data)

            # 3. Load preprocessor and model
            preprocessor = joblib.load(Path('final_model/preprocessor.pkl'))
            model = joblib.load(Path('final_model/model.pkl'))

            # 4. Transform input
            transformed_input = preprocessor.transform(input_df)

            logging.info("Transformed input: %s", transformed_input)

            # 5. Predict
            prediction = model.predict(transformed_input)[0]

            logging.info("Prediction: %s", model.predict(transformed_input))

            return render_template('index.html', prediction=str(prediction))

        except Exception as e:

            print('The Exception message is: ', e)
            return 'Something went wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
