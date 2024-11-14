# Loan Eligibility Checker

**Description**:  
This project is a web-based application designed to predict loan eligibility for applicants based on their financial and personal information. Leveraging machine learning, the system aims to assist financial institutions in streamlining the loan approval process and improving decision accuracy. 

**Dataset**:  
The project utilizes publicly available datasets from Kaggle, which include details such as applicant income, employment status, credit history, loan amount, and more.

**Machine Learning Algorithms**:  
The application employs Logistic Regression and Decision Tree algorithms to make predictions. Both models are evaluated on precision, recall, F1-score, and ROC-AUC to ensure optimal performance.

**Business Problem**:  
This tool addresses a core business need by automating and improving the loan eligibility assessment process, ultimately reducing manual effort and minimizing default rates.

**User Interface**:  
A user-friendly web interface allows users to input applicant data and instantly receive a loan eligibility prediction. The frontend is developed using JavaScript, HTML5, and CSS3, while the backend and machine learning components are implemented in Python with Flask.

---

### Project Development

**Analysis**:
1. **Exploratory Data Analysis (EDA)**:  
   The project includes EDA to understand data distribution and relationships. It offers statistical summaries, visualizations (e.g., histograms, bar charts, and correlation heatmaps), and insights into factors like credit history's impact on loan status.
   
2. **Model Development**:
   - Logistic Regression and Decision Tree algorithms are used to predict eligibility.
   - Techniques to handle class imbalance are applied to ensure fair model performance.
   - Models are evaluated on several metrics to ensure accuracy and reliability.

**Design and Development**:
- **Languages**: Python for backend and ML, JavaScript/HTML/CSS for frontend.
- **Platform**: Cross-platform development, with deployment on cloud platforms (e.g., Heroku or AWS).
- **Database**: SQLite, if needed, for data storage.

**Estimated Hours**: ~75 hours
- Planning and Design: 15 hours
- Development: 45 hours
- Documentation: 15 hours

**Projected Completion Date**: November 29, 2024

---

### Implementation and Evaluation

**Execution Steps**:
1. **Data Collection & EDA**: Analyze the Kaggle dataset for insights and prepare it for model training.
2. **Model Training**: Build, tune, and evaluate Logistic Regression and Decision Tree models.
3. **Web Application Development**: Integrate the model into a Flask-based web app with a simple, intuitive UI.
4. **Testing**: Conduct extensive functional and performance testing.
5. **Deployment**: Host the application on a cloud platform and document the deployment process.

**References**:  
Ukani, Vikas. (2019). *Loan Eligible Dataset* [Data set]. Kaggle. [Dataset Link](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)
