"""
This script defines a Flask application for loan eligibility prediction.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Load the trained model, scaler, and feature names
model = joblib.load('model/loan_eligibility_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

# Load the dataset for visualization
data = pd.read_csv('dataset/loan-train.csv')


def validate_input(value, expected_type, choices=None):
    """Validates input to ensure it meets expected type and optional constraints."""
    try:
        if value is None:
            return None
        if expected_type == int:
            value = int(value)
        elif expected_type == float:
            value = float(value)
        elif expected_type == str:
            value = str(value)
            if choices and value not in choices:
                raise ValueError("Invalid choice")
        return value
    except (ValueError, TypeError):
        return None


@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, processes input, and returns prediction results."""
    if request.method == 'POST':
        try:
            # Validate and retrieve input data from form
            gender = validate_input(request.form.get(
                'Gender'), str, choices=['Male', 'Female'])
            married = validate_input(request.form.get(
                'Married'), str, choices=['Yes', 'No'])
            dependents = validate_input(request.form.get(
                'Dependents'), str, choices=['0', '1', '2', '3+'])
            education = validate_input(request.form.get(
                'Education'), str, choices=['Graduate', 'Not Graduate'])
            self_employed = validate_input(request.form.get(
                'Self_Employed'), str, choices=['Yes', 'No'])
            applicant_income = validate_input(
                request.form.get('ApplicantIncome'), float)
            coapplicant_income = validate_input(
                request.form.get('CoapplicantIncome'), float)
            loan_amount = validate_input(request.form.get('LoanAmount'), float)
            loan_amount_term = validate_input(
                request.form.get('Loan_Amount_Term'), float)
            credit_history = validate_input(
                request.form.get('Credit_History'), float)
            property_area = validate_input(request.form.get(
                'Property_Area'), str, choices=['Urban', 'Semiurban', 'Rural'])

            # Check for any invalid inputs
            if None in [gender, married, dependents, education, self_employed, applicant_income,
                        coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]:
                flash("Invalid input. Please check your form entries.", "danger")
                return redirect(url_for('home'))

            # Data preprocessing as done during training
            # Create a DataFrame with all features
            input_dict = {
                'Dependents': 3 if dependents == '3+' else int(dependents),
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Gender_Male': 1 if gender == 'Male' else 0,
                'Married_Yes': 1 if married == 'Yes' else 0,
                'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
                'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
                'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
                'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
            }

            # Ensure all features are present
            for col in feature_names:
                if col not in input_dict:
                    input_dict[col] = 0  # Default value for missing features

            # Create DataFrame
            features_df = pd.DataFrame([input_dict], columns=feature_names)

            # Scale the features
            features_scaled = scaler.transform(features_df)

            # Make prediction
            prediction = model.predict(features_scaled)
            output = 'Eligible' if prediction[0] == 1 else 'Not Eligible'

            return render_template('result.html', prediction=output)

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('home'))

    # Redirect to home if not POST
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    """Generates data visualizations and renders the dashboard page."""
    try:
        # Plot 1: Histogram of Applicant Income
        img1 = io.BytesIO()
        plt.figure(figsize=(6, 4))
        sns.histplot(data['ApplicantIncome'], kde=True)
        plt.title('Distribution of Applicant Income')
        plt.savefig(img1, format='png')
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()

        # Plot 2: Bar chart of Loan Status counts
        img2 = io.BytesIO()
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Loan_Status', data=data)
        plt.title('Loan Status Counts')
        plt.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()

        # Plot 3: Heatmap of feature correlations
        img3 = io.BytesIO()
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                    annot_kws={"size": 10})  # Format annotations
        plt.title('Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.tight_layout()  # Adjust layout to fit everything
        plt.savefig(img3, format='png')
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode()

        return render_template('dashboard.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3)

    except Exception as e:
        flash(
            f"An error occurred while generating the dashboard: {str(e)}", "danger")
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
