from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load dataset
car = pd.read_csv("Cleaned_Car_data.csv")

# Prepare data
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Convert categorical columns to numbers
X = pd.get_dummies(X)

# Train model
model = LinearRegression()
model.fit(X, y)


# ---------------- ROUTES ---------------- #

@app.route('/')
def login():
    return render_template("login.html")


@app.route('/home', methods=['POST'])
def home():
    return render_template("home.html")


@app.route('/prediction')
def prediction():

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template(
        "predict.html",
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type
    )


@app.route('/predict', methods=['POST'])
def predict():

    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms = int(request.form.get('kilo_driven'))

    input_data = pd.DataFrame(
        [[car_model, company, year, kms, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    input_data = pd.get_dummies(input_data)

    # Align columns with training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)

    return str(round(prediction[0], 2))


# ✅ FIXED POSITION (IMPORTANT)

@app.route('/cars')
def cars():

    grouped_data = {}

    for company in sorted(car['company'].unique()):
        models = sorted(car[car['company'] == company]['name'].unique())
        grouped_data[company] = models

    return render_template("cars.html", grouped_data=grouped_data)


@app.route('/how')
def how():
    return render_template("how.html")


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)