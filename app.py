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

# Load the trained model
model = joblib.load('model/loan_eligibility_model.pkl')

# Load the dataset for visualization
data = pd.read_csv('dataset/loan-train.csv')


def validate_input(value, expected_type, choices=None):
    """Validates input to ensure it meets expected type and optional constraints."""
    try:
        if expected_type == int:
            value = int(value)
        elif expected_type == float:
            value = float(value)
        elif expected_type == str:
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
            gender = validate_input(
                request.form['Gender'], str, choices=['Male', 'Female'])
            married = validate_input(
                request.form['Married'], str, choices=['Yes', 'No'])
            dependents = validate_input(
                request.form['Dependents'], str, choices=['0', '1', '2', '3+'])
            education = validate_input(request.form['Education'], str, choices=[
                                       'Graduate', 'Not Graduate'])
            self_employed = validate_input(
                request.form['Self_Employed'], str, choices=['Yes', 'No'])
            applicant_income = validate_input(
                request.form['ApplicantIncome'], float)
            coapplicant_income = validate_input(
                request.form['CoapplicantIncome'], float)
            loan_amount = validate_input(request.form['LoanAmount'], float)
            loan_amount_term = validate_input(
                request.form['Loan_Amount_Term'], float)
            credit_history = validate_input(
                request.form['Credit_History'], float)
            property_area = validate_input(request.form['Property_Area'], str, choices=[
                                           'Urban', 'Semiurban', 'Rural'])

            # Check for any invalid inputs
            if None in [gender, married, dependents, education, self_employed, applicant_income,
                        coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]:
                flash("Invalid input. Please check your form entries.", "danger")
                return redirect(url_for('home'))

            # Data preprocessing as done during training
            if dependents == '3+':
                dependents = 3
            else:
                dependents = int(dependents)

            # Create input array for numerical features
            input_data = [
                applicant_income, coapplicant_income, loan_amount,
                loan_amount_term, credit_history, dependents
            ]

            # Encode categorical variables
            categorical_data = [0, 0, 0, 0, 0, 0]
            if gender == 'Male':
                categorical_data[0] = 1
            if married == 'Yes':
                categorical_data[1] = 1
            if education == 'Not Graduate':
                categorical_data[2] = 1
            if self_employed == 'Yes':
                categorical_data[3] = 1
            if property_area == 'Semiurban':
                categorical_data[4] = 1
            elif property_area == 'Urban':
                categorical_data[5] = 1

            # Combine all features
            features = np.array(input_data + categorical_data).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features)
            output = 'Eligible' if prediction[0] == 1 else 'Not Eligible'

            return render_template('result.html', prediction=output)

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
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
        plt.figure(figsize=(8, 6))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(img3, format='png')
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode()

        return render_template('dashboard.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3)

    except Exception as e:
        flash(
            f"An error occurred while generating the dashboard: {str(e)}", "danger")
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run()  # Turn off debug mode in production
