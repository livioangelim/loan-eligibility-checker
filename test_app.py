import unittest
from app import app


class LoanEligibilityTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_eligible_applicant(self):
        # Define input data for an eligible applicant
        data = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': '8000',
            'CoapplicantIncome': '0',
            'LoanAmount': '150',
            'Loan_Amount_Term': '360',
            'Credit_History': '1.0',
            'Property_Area': 'Urban'
        }
        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertIn(b'Eligible', response.data)
        print("Test Eligible Applicant: Passed")

    def test_not_eligible_applicant(self):
        # Define input data for a not eligible applicant
        data = {
            'Gender': 'Male',
            'Married': 'No',
            'Dependents': '2',
            'Education': 'Not Graduate',
            'Self_Employed': 'Yes',
            'ApplicantIncome': '1000',
            'CoapplicantIncome': '0',
            'LoanAmount': '500',
            'Loan_Amount_Term': '360',
            'Credit_History': '0.0',
            'Property_Area': 'Rural'
        }
        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertIn(b'Not Eligible', response.data)
        print("Test Not Eligible Applicant: Passed")

    def test_missing_input(self):
        # Test with missing input data
        data = {
            'Gender': 'Male',
            # 'Married' field is missing
            'Dependents': '1',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': '5000',
            'CoapplicantIncome': '0',
            'LoanAmount': '100',
            'Loan_Amount_Term': '360',
            'Credit_History': '1.0',
            'Property_Area': 'Urban'
        }
        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertIn(b'Invalid input', response.data)
        print("Test Missing Input: Passed")

    def test_invalid_input_type(self):
        # Test with invalid input types
        data = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': 'Two',  # Invalid input
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 'NotANumber',  # Invalid input
            'CoapplicantIncome': '0',
            'LoanAmount': '100',
            'Loan_Amount_Term': '360',
            'Credit_History': '1.0',
            'Property_Area': 'Urban'
        }
        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertIn(b'Invalid input', response.data)
        print("Test Invalid Input Type: Passed")

    def test_zero_income(self):
        # Test with zero income
        data = {
            'Gender': 'Female',
            'Married': 'No',
            'Dependents': '0',
            'Education': 'Not Graduate',
            'Self_Employed': 'Yes',
            'ApplicantIncome': '0',
            'CoapplicantIncome': '0',
            'LoanAmount': '200',
            'Loan_Amount_Term': '360',
            'Credit_History': '0.0',
            'Property_Area': 'Rural'
        }
        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertIn(b'Not Eligible', response.data)
        print("Test Zero Income: Passed")


if __name__ == '__main__':
    unittest.main()
