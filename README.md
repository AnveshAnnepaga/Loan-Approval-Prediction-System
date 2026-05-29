# FinVantage: Intelligent Loan Approval & Credit Risk Prediction System

FinVantage is a premium, state-of-the-art interactive web dashboard built with **Streamlit** that automates loan appraisals, risk scoring, behavioral customer segmentation, and visual dimensionality reduction.

The platform is designed to run seamlessly in any environment out-of-the-box by leveraging a **smart data loader** that auto-detects the Kaggle dataset or falls back onto a mathematically consistent, high-fidelity synthetic credit simulator.

---

## 🚀 Key Features

The application is structured into four robust interactive spaces:
1. **Executive Portfolio Analytics**: A high-end corporate dashboard displaying key performance indicators (KPIs) of the credit pool, default frequencies grouped by demographics, interest rates, and loan categories.
2. **Interactive Credit Decisioning Engine**: An advanced credit screening interface combining two models:
   - **Logistic Regression Engine**: Predicts regulatory-friendly credit approvals/rejections based on safety thresholds.
   - **Random Forest Classifier**: Computes highly granular credit default probability (risk percentages), classifies risk tiers (Low, Medium, High), and visualizes the local risk drivers and protective factors for the applicant.
3. **K-Means Customer Cohorts**: Dynamically groups the entire borrower database into 4 intuitive customer segments (*Prime Low-Risk*, *High-Debt Leveraged*, *Young Career Builders*, *High-Risk Premium*). It assigns the active applicant to their cohort and recommends personalized financial products and underwriting guidelines.
4. **PCA Dimensional Mapping**: Conducts **Principal Component Analysis** to reduce the 18-dimensional feature space to a 2D plot. It maps the active applicant **live in real-time** as a neon gold marker, letting bank underwriters visually inspect where the borrower sits relative to the overall population.

---

## 📁 System Architecture

The codebase is designed around clean, modular programming principles:
- **`data_loader.py`**: Handles environment safety and database operations. Auto-detects the 255,000+ record Kaggle dataset. If missing, it executes a custom financial multivariate generator matching the Kaggle schema (18 columns, realistic credit distributions, and an ~11.6% baseline default rate).
- **`models.py`**: Houses the scikit-learn machine learning pipelines. Uses a standard `ColumnTransformer` to normalize numeric columns and one-hot encode categoricals. Automatically caches and runs training for Logistic Regression, Random Forest, K-Means Clustering, and PCA.
- **`app.py`**: The Streamlit user interface, customized with a rich glassmorphic dark-theme configuration, customized metric cards, interactive Plotly charts, and responsive session-state form handling.

---

## 📊 Dataset Schema

The system supports the full 18-column schema of the Kaggle **Coursera Loan Default Prediction** dataset:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **LoanID** | Text | Unique identifier for each loan record |
| **Age** | Numerical | Borrower age (18 - 69) |
| **Income** | Numerical | Annual income in USD ($15,000 - $150,000) |
| **LoanAmount** | Numerical | Value of the requested loan ($5,000 - $250,000) |
| **CreditScore** | Numerical | FICO credit score (300 - 850) |
| **MonthsEmployed** | Numerical | Months spent in active employment (0 - 120) |
| **NumCreditLines** | Numerical | Number of active credit lines (1 - 20) |
| **InterestRate** | Numerical | Annual interest rate (2.0% - 25.0%) |
| **LoanTerm** | Numerical | Duration of the loan (12, 24, 36, 48, 60 months) |
| **DTIRatio** | Numerical | Debt-to-Income ratio (0.10 - 0.90) |
| **Education** | Categorical | High School, Bachelor's, Master's, PhD |
| **EmploymentType** | Categorical | Full-time, Part-time, Self-employed, Unemployed |
| **MaritalStatus** | Categorical | Single, Married, Divorced |
| **HasMortgage** | Categorical | Yes, No |
| **HasDependents** | Categorical | Yes, No |
| **LoanPurpose** | Categorical | Home, Auto, Education, Business, Other |
| **HasCoSigner** | Categorical | Yes, No |
| **Default** | Binary Target | 1 = Borrower defaulted; 0 = Paid / Normal |

---

## 🛠️ Installation & Execution Guide

Follow these simple steps to run the application locally on your system:

### 1. Pre-requisites
Ensure you have Python 3.9 or higher installed.

### 2. Clone and Setup
Open your terminal inside the cloned workspace folder:
```bash
# Navigate to the repository root
cd "d:\Tekworks\Loan Approval Prediction System\Loan-Approval-Prediction-System"

# (Optional) Create and activate a python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
Launch the local web server:
```bash
streamlit run app.py
```
This will spin up a local development server and automatically open a web tab at `http://localhost:8501`.

---

## 🔌 Connecting Your Real Kaggle Dataset

To run the platform on the real **Kaggle Loan Default Dataset**:
1. Download the dataset `Loan_default.csv` from [Kaggle Dataset Page](https://www.kaggle.com/datasets/nikhil1e9/loan-default).
2. Place the file `Loan_default.csv` directly into the cloned repository folder (alongside `app.py`).
3. Refresh the web page. The **Database Telemetry** panel in the sidebar will instantly report `Mode: Kaggle CSV Connected`, retrain the pipelines, and update all metrics, charts, and predictions with the 255k+ historical bank records!