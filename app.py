from flask import Flask, request, render_template
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load and preprocess training data
data = pd.read_csv('loan-train.csv')

# Handle missing values
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in categorical_cols + ['Property_Area', 'Loan_Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

# Split the dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)


@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy, class_report=class_report)


@app.route('/predict_test', methods=['POST'])
def predict_test():
    try:
        # Check if file is uploaded
        if 'test_file' not in request.files:
            return render_template('index.html', error="No file uploaded", accuracy=accuracy, class_report=class_report)

        file = request.files['test_file']
        if file.filename == '':
            return render_template('index.html', error="No file selected", accuracy=accuracy, class_report=class_report)

        # Load and preprocess test data
        test_data = pd.read_csv(file)

        # Handle missing values
        for col in numerical_cols:
            test_data[col] = test_data[col].fillna(test_data[col].median())
        for col in categorical_cols:
            test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

        # Encode categorical variables
        for col in categorical_cols + ['Property_Area']:
            test_data[col] = label_encoders[col].transform(test_data[col])

        # Prepare test features
        X_test_final = test_data.drop(['Loan_ID'], axis=1)

        # Make predictions
        test_predictions = model.predict(X_test_final)
        test_predictions = ['Eligible' if pred == 1 else 'Not Eligible' for pred in test_predictions]

        # Create results DataFrame
        results = pd.DataFrame({
            'Loan_ID': test_data['Loan_ID'],
            'Prediction': test_predictions
        })

        return render_template('index.html', results=results.to_dict('records'),
                               accuracy=accuracy, class_report=class_report)
    except Exception as e:
        return render_template('index.html', error=str(e), accuracy=accuracy, class_report=class_report)


@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get form data
        form_data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_amount_term']),
            'Credit_History': request.form['credit_history'],
            'Property_Area': request.form['property_area']
        }

        # Create DataFrame from form data
        input_df = pd.DataFrame([form_data])

        # Encode categorical variables
        for col in categorical_cols + ['Property_Area']:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Make prediction
        prediction = model.predict(input_df)[0]
        single_result = 'Eligible' if prediction == 1 else 'Not Eligible'

        return render_template('index.html', single_result=single_result,
                               accuracy=accuracy, class_report=class_report)
    except Exception as e:
        return render_template('index.html', single_error=str(e), accuracy=accuracy, class_report=class_report)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)