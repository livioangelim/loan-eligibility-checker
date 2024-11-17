# Loan Eligibility Checker

## Overview

The **Loan Eligibility Checker** is a web application that allows loan officers to assess loan eligibility quickly and accurately. Built using Python's Flask framework, the application leverages a machine learning model to predict loan eligibility based on applicant data. It also provides a data visualization dashboard to offer insights into the underlying dataset.

## Features

- **User-Friendly Interface**: A web-based form for inputting applicant personal and financial information.
- **Machine Learning Prediction**: Utilizes a trained Logistic Regression model to predict loan eligibility.
- **Data Visualization Dashboard**: Visualizes key insights from the dataset, including income distribution, loan status counts, and correlation heatmaps.
- **Input Validation**: Ensures all user inputs are valid and provides feedback on invalid entries.
- **Automated Testing**: Includes a suite of tests to verify the application's functionality under various scenarios.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Git (optional, for cloning the repository)
- pip (Python package installer)

### Installation

#### 1. Clone the Repository (Optional)

You can clone the repository using Git:

```bash
git clone https://github.com/yourusername/loan-eligibility-checker.git
cd loan-eligibility-checker
```

Alternatively, you can download the ZIP file from the repository and extract it.

#### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On macOS and Linux:

  ```bash
  source venv/bin/activate
  ```

#### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```
Flask==2.0.3
pandas==1.3.5
numpy==1.21.5
scikit-learn==1.0.2
imbalanced-learn==0.8.1
matplotlib==3.5.1
seaborn==0.11.2
joblib==1.1.0
```

#### 4. Download the Dataset

Ensure the dataset `loan-train.csv` is placed in the `dataset` directory. If it's not already there, download it from the Kaggle link provided in the Acknowledgments section.

#### 5. Train the Machine Learning Model

Run the Jupyter Notebook to prepare the data and train the model:

```bash
jupyter notebook data_preparation.ipynb
```

- Execute all cells in `data_preparation.ipynb`.
- This will preprocess the data, handle class imbalance, train the model, and save the necessary files (`loan_eligibility_model.pkl`, `scaler.pkl`, and `feature_names.pkl`) in the `model` directory.

## Running the Application

### 1. Start the Flask Application

Ensure you're in the project directory and your virtual environment is activated.

```bash
python app.py
```

By default, the application will run on `http://127.0.0.1:5000/`.

### 2. Access the Application

Open a web browser and navigate to `http://127.0.0.1:5000/`.

### 3. Using the Application

- Fill out the loan application form with the applicant's details.
- Click on **Check Eligibility** to see the prediction.
- To view data visualizations, click on **View Data Dashboard**.

## Running Tests

To run the automated tests:

```bash
python test_app.py
```

You should see output indicating that all tests have passed.

## Example Usage

Below are examples for both male and female applicants in both eligible and not eligible categories.

---

### Eligible Applicant - Male

- **Input Details:**

  - **Gender:** Male
  - **Married:** Yes
  - **Dependents:** 0
  - **Education:** Graduate
  - **Self_Employed:** No
  - **ApplicantIncome:** 8000
  - **CoapplicantIncome:** 2000
  - **LoanAmount:** 150
  - **Loan_Amount_Term:** 360
  - **Credit_History:** 1 (Has credit history)
  - **Property_Area:** Urban

- **Steps:**

  1. Open the application at `http://127.0.0.1:5000/`.
  2. Fill in the above details in the form.
  3. Click **Check Eligibility**.

- **Output:**

  - The application displays **"Eligible"**

---

### Eligible Applicant - Female

- **Input Details:**

  - **Gender:** Female
  - **Married:** Yes
  - **Dependents:** 1
  - **Education:** Graduate
  - **Self_Employed:** No
  - **ApplicantIncome:** 7500
  - **CoapplicantIncome:** 1500
  - **LoanAmount:** 120
  - **Loan_Amount_Term:** 360
  - **Credit_History:** 1 (Has credit history)
  - **Property_Area:** Semiurban

- **Steps:**

  1. Open the application at `http://127.0.0.1:5000/`.
  2. Enter the above details in the form.
  3. Click **Check Eligibility**.

- **Output:**

  - The application displays **"Eligible"**

---

### Not Eligible Applicant - Male

- **Input Details:**

  - **Gender:** Male
  - **Married:** No
  - **Dependents:** 2
  - **Education:** Not Graduate
  - **Self_Employed:** Yes
  - **ApplicantIncome:** 3000
  - **CoapplicantIncome:** 0
  - **LoanAmount:** 200
  - **Loan_Amount_Term:** 360
  - **Credit_History:** 0 (No credit history)
  - **Property_Area:** Rural

- **Steps:**

  1. Open the application at `http://127.0.0.1:5000/`.
  2. Input the above details into the form.
  3. Click **Check Eligibility**.

- **Output:**

  - The application displays **"Not Eligible"**

---

### Not Eligible Applicant - Female

- **Input Details:**

  - **Gender:** Female
  - **Married:** No
  - **Dependents:** 3+
  - **Education:** Not Graduate
  - **Self_Employed:** Yes
  - **ApplicantIncome:** 2500
  - **CoapplicantIncome:** 0
  - **LoanAmount:** 180
  - **Loan_Amount_Term:** 180
  - **Credit_History:** 0 (No credit history)
  - **Property_Area:** Rural

- **Steps:**

  1. Open the application at `http://127.0.0.1:5000/`.
  2. Enter the above details in the form.
  3. Click **Check Eligibility**.

- **Output:**

  - The application displays **"Not Eligible"**

---

## Project Structure

```
loan-eligibility-checker/
├── app.py
├── data_preparation.ipynb
├── test_app.py
├── requirements.txt
├── dataset/
│   ├── loan-train.csv
│   └── loan-test.csv
├── model/
│   ├── loan_eligibility_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
└── static/
    └── (optional static files)
```

## Dependencies

- **Flask==2.0.3**: Web framework for Python.
- **pandas==1.3.5**: Data manipulation and analysis.
- **numpy==1.21.5**: Numerical computing.
- **scikit-learn==1.0.2**: Machine learning algorithms.
- **imbalanced-learn==0.8.1**: Handling class imbalance with techniques like SMOTE.
- **matplotlib==3.5.1**: Data visualization.
- **seaborn==0.11.2**: Statistical data visualization built on top of matplotlib.
- **joblib==1.1.0**: Serialization and parallel computing.

## Important Files

- **app.py**: The main Flask application script.
- **data_preparation.ipynb**: Jupyter Notebook for data preprocessing and model training.
- **test_app.py**: Contains automated tests for the application.
- **templates/**: Directory containing HTML templates.
- **model/**: Contains the trained model and related files.
- **dataset/**: Contains the dataset used for training.

## Notes

- Ensure that the feature names and preprocessing steps in `app.py` match those used during model training in `data_preparation.ipynb`.
- The application uses the `StandardScaler` for feature scaling. The same scaler must be applied to input data during prediction.
- When adding new dependencies or making changes to the code, update the `requirements.txt` and retrain the model if necessary.

## Troubleshooting

- **Module Not Found Errors**: Ensure all dependencies are installed in your virtual environment.
- **Model Prediction Issues**: Verify that the model and scaler are correctly loaded and that the input features match those used during training.
- **Application Crashes**: Check the console for error messages and ensure all files are in the correct directories.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset obtained from [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).
- Thanks to the Flask and scikit-learn communities for their excellent tools and documentation.