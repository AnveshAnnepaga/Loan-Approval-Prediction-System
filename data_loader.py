import os
import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=10000, seed=42):
    """
    Generates a highly realistic synthetic loan default dataset that mimics the 
    Kaggle 'nikhil1e9/loan-default' schema (18 columns) and realistic financial correlations.
    """
    np.random.seed(seed)
    
    # 1. Unique Loan ID
    loan_ids = [f"LN_{i:06d}" for i in range(num_samples)]
    
    # 2. Age (18 to 69)
    age = np.random.randint(18, 70, size=num_samples)
    
    # 3. Income ($15,000 to $150,000)
    income = np.random.randint(15000, 150001, size=num_samples)
    
    # 4. LoanAmount ($5,000 to $250,000)
    loan_amount = np.random.randint(5000, 250001, size=num_samples)
    
    # 5. CreditScore (300 to 850)
    credit_score = np.random.randint(300, 851, size=num_samples)
    
    # 6. MonthsEmployed (0 to 120)
    months_employed = np.random.randint(0, 121, size=num_samples)
    
    # 7. NumCreditLines (1 to 20)
    num_credit_lines = np.random.randint(1, 21, size=num_samples)
    
    # 8. InterestRate (2.0% to 25.0%)
    interest_rate = np.round(np.random.uniform(2.0, 25.0, size=num_samples), 2)
    
    # 9. LoanTerm (12, 24, 36, 48, 60 months)
    loan_term = np.random.choice([12, 24, 36, 48, 60], size=num_samples)
    
    # 10. DTIRatio (Debt-to-Income, 0.10 to 0.90)
    dti_ratio = np.round(np.random.uniform(0.1, 0.9, size=num_samples), 2)
    
    # Categoricals
    # 11. Education
    education_opts = ['High School', "Bachelor's", "Master's", 'PhD']
    education = np.random.choice(education_opts, size=num_samples, p=[0.35, 0.40, 0.18, 0.07])
    
    # 12. EmploymentType
    employment_opts = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    employment_type = np.random.choice(employment_opts, size=num_samples, p=[0.55, 0.20, 0.15, 0.10])
    
    # 13. MaritalStatus
    marital_opts = ['Single', 'Married', 'Divorced']
    marital_status = np.random.choice(marital_opts, size=num_samples, p=[0.40, 0.40, 0.20])
    
    # 14. HasMortgage
    has_mortgage = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.30, 0.70])
    
    # 15. HasDependents
    has_dependents = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.40, 0.60])
    
    # 16. LoanPurpose
    purpose_opts = ['Home', 'Auto', 'Education', 'Business', 'Other']
    loan_purpose = np.random.choice(purpose_opts, size=num_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    
    # 17. HasCoSigner
    has_cosigner = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.45, 0.55])
    
    # 18. Default Flag Generation (Correlated with features)
    # Convert indicators for correlation calculations
    score_norm = 1.0 - (credit_score - 300) / 550.0  # 1 = worst score, 0 = best score
    dti_norm = dti_ratio  # 0.1 to 0.9
    rate_norm = (interest_rate - 2.0) / 23.0  # 0 to 1
    employed_norm = 1.0 - (months_employed / 120.0)  # 1 = unemployed longest, 0 = employed longest
    # Cap loan to income ratio at 4.0 and scale to 0-1 range to prevent logit explosion
    loan_to_income = np.minimum(loan_amount / income, 4.0) / 4.0
    age_norm = 1.0 - (age - 18) / 52.0  # 1 = youngest, 0 = oldest
    
    # Categorical penalty weights
    emp_weight = np.where(employment_type == 'Unemployed', 0.8, 
                 np.where(employment_type == 'Part-time', 0.3, 0.0))
    cosigner_weight = np.where(has_cosigner == 'No', 0.4, 0.0)
    mortgage_weight = np.where(has_mortgage == 'Yes', 0.1, 0.0)
    
    # Compute Logistic risk score calibrated to yield ~11.6% average default rate
    logit = (-4.6 
             + 1.5 * score_norm 
             + 1.2 * rate_norm 
             + 1.0 * dti_norm 
             + 0.8 * loan_to_income 
             + 0.6 * employed_norm 
             + 0.4 * age_norm 
             + emp_weight 
             + cosigner_weight 
             + mortgage_weight)
    
    # Compute probabilities using Sigmoid function
    prob = 1.0 / (1.0 + np.exp(-logit))
    
    # Generate binary target based on probabilities
    default = np.random.binomial(1, prob)
    
    # Create DataFrame matching the Kaggle column names exactly
    df = pd.DataFrame({
        'LoanID': loan_ids,
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Education': education,
        'EmploymentType': employment_type,
        'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage,
        'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner,
        'Default': default
    })
    
    return df

def load_dataset(base_dir=None):
    """
    Checks for a local copy of 'Loan_default.csv' or 'loan_default.csv' in the directory.
    If missing, falls back to generating a realistic synthetic dataset and alerts the application.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        
    possible_names = ['Loan_default.csv', 'loan_default.csv', 'Loan-Default.csv']
    
    # Check current directory and subdirectories (e.g. Loan-Approval-Prediction-System)
    for name in possible_names:
        paths_to_check = [
            os.path.join(base_dir, name),
            os.path.join(base_dir, 'Loan-Approval-Prediction-System', name),
            os.path.join(os.path.dirname(base_dir), name)
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # Verify key columns exist
                    if 'Default' in df.columns and 'CreditScore' in df.columns:
                        return df, False, path  # False = Not synthetic, Path of file
                except Exception as e:
                    print(f"Error loading CSV from {path}: {e}")
                    
    # Fallback to generating synthetic data
    df = generate_synthetic_data()
    # Save synthetic dataset to project root for user review / convenience
    synthetic_path = os.path.join(base_dir, 'Loan_default_synthetic.csv')
    df.to_csv(synthetic_path, index=False)
    
    return df, True, synthetic_path
