import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

class LoanMLPipeline:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.preprocessor = None
        self.log_reg = None
        self.rf_clf = None
        self.kmeans = None
        self.pca = None
        self.feature_names = []
        self.numeric_cols = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 
            'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
            'LoanTerm', 'DTIRatio', 'LoanToIncomeRatio'
        ]
        self.categorical_cols = [
            'Education', 'EmploymentType', 'MaritalStatus', 
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
        ]
        
    def fit(self, df):
        """
        Trains all models: Logistic Regression, Random Forest, K-Means, and PCA
        """
        # Separate features and target with engineered feature
        df_copy = df.copy()
        df_copy['LoanToIncomeRatio'] = df_copy['LoanAmount'] / df_copy['Income']
        X = df_copy[self.numeric_cols + self.categorical_cols].copy()
        y = df_copy['Default'].copy()
        
        # Build Preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
            ]
        )
        
        # Fit preprocessor and get processed X
        X_processed = self.preprocessor.fit_transform(X)
        
        # Extract feature names after OneHotEncoding
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
        self.feature_names = list(self.numeric_cols) + list(encoded_cat_names)
        
        # Split data for supervised learning evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # 1. Train Logistic Regression
        self.log_reg = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.log_reg.fit(X_train, y_train)
        
        log_preds = self.log_reg.predict(X_test)
        log_probs = self.log_reg.predict_proba(X_test)[:, 1]
        
        self.log_metrics = {
            'accuracy': accuracy_score(y_test, log_preds),
            'roc_auc': roc_auc_score(y_test, log_probs),
            'report': classification_report(y_test, log_preds, output_dict=True)
        }
        
        # 2. Train Random Forest
        self.rf_clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.rf_clf.fit(X_train, y_train)
        
        rf_preds = self.rf_clf.predict(X_test)
        rf_probs = self.rf_clf.predict_proba(X_test)[:, 1]
        
        self.rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_preds),
            'roc_auc': roc_auc_score(y_test, rf_probs),
            'report': classification_report(y_test, rf_preds, output_dict=True),
            'importances': dict(zip(self.feature_names, self.rf_clf.feature_importances_))
        }
        
        # 3. Train K-Means on whole processed dataset
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X_processed)
        df_clustered = df.copy()
        df_clustered['Cluster'] = cluster_labels
        
        # Analyze clusters and name them
        self.cluster_profiles = self._analyze_clusters(df_clustered)
        
        # 4. Train PCA on whole processed dataset (for visual rendering in 2D space)
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_processed)
        
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = cluster_labels
        df_pca['Default'] = df['Default']
        df_pca['LoanID'] = df['LoanID']
        df_pca['Age'] = df['Age']
        df_pca['Income'] = df['Income']
        df_pca['LoanAmount'] = df['LoanAmount']
        df_pca['CreditScore'] = df['CreditScore']
        
        return df_clustered, df_pca
        
    def _analyze_clusters(self, df):
        """
        Creates statistical profiles of the clusters to define customer segments.
        """
        profiles = {}
        for c in range(self.n_clusters):
            cluster_df = df[df['Cluster'] == c]
            
            # Compute stats
            avg_income = int(cluster_df['Income'].mean())
            avg_credit = int(cluster_df['CreditScore'].mean())
            avg_loan = int(cluster_df['LoanAmount'].mean())
            avg_dti = float(cluster_df['DTIRatio'].mean())
            default_rate = float(cluster_df['Default'].mean())
            avg_age = int(cluster_df['Age'].mean())
            avg_rate = float(cluster_df['InterestRate'].mean())
            
            # Map semantic name based on quantitative stats
            if avg_credit >= 680 and default_rate < 0.05:
                name = "Prime Low-Risk Borrowers"
                desc = "Excellent credit, low default rate, ideal for premium accounts and rewards."
                color = "#10b981"  # Calm Emerald
            elif avg_dti >= 0.55 or avg_loan / avg_income > 2.0:
                name = "High-Debt Leveraged Profiles"
                desc = "Moderate-to-high loan amount relative to income. Susceptible to market interest rates."
                color = "#f97316"  # Calm Orange
            elif avg_age < 35 and avg_income < 60000:
                name = "Young Career Builders"
                desc = "Younger demographic with lower income but moderate credit. High growth potential."
                color = "#0ea5e9"  # Calm Sky Blue
            else:
                name = "High-Risk Premium Borrowers"
                desc = "Higher interest rates, lower credit scores, elevated default probabilities."
                color = "#ef4444"  # Calm Crimson Red
            
            # Adjust if names overlap due to random distributions (safeguard)
            profiles[c] = {
                'id': c,
                'name': name,
                'description': desc,
                'color': color,
                'size': len(cluster_df),
                'avg_income': avg_income,
                'avg_credit_score': avg_credit,
                'avg_loan_amount': avg_loan,
                'avg_dti': avg_dti,
                'avg_age': avg_age,
                'avg_rate': avg_rate,
                'default_rate': default_rate
            }
            
        # Ensure name uniqueness (fallback formatting if overlapping names occur)
        seen_names = {}
        for c, prof in list(profiles.items()):
            name = prof['name']
            if name in seen_names:
                seen_names[name] += 1
                profiles[c]['name'] = f"{name} (Group {seen_names[name]})"
            else:
                seen_names[name] = 1
                
        return profiles
        
    def predict_applicant(self, applicant_dict):
        """
        Takes a raw applicant dictionary and returns:
        1. Logistic Regression Decision (Approve vs Reject) and Confidence
        2. Random Forest Default Probability and Risk Tier
        3. K-Means Assigned Cluster Details
        4. PCA coordinates for 2D mapping
        """
        # Convert applicant dict to DataFrame with engineered feature
        app_dict = applicant_dict.copy()
        app_dict['LoanToIncomeRatio'] = app_dict['LoanAmount'] / app_dict['Income']
        df_app = pd.DataFrame([app_dict])
        
        # Preprocess features
        X_app = self.preprocessor.transform(df_app)
        
        # 1. Logistic Regression Prediction
        # Predict 0 (Approved / No Default) vs 1 (Rejected / Default)
        log_prob_default = self.log_reg.predict_proba(X_app)[0, 1]
        
        # Decision: Approve if probability of default is less than 15% (standard threshold)
        decision_threshold = 0.15
        
        # Underwriting Override: Low-Exposure Approval Path
        # If the requested loan is less than 25% of annual income, and FICO is >= 580 (decent credit),
        # we adjust the threshold to 55% to account for the extremely low default exposure.
        loan_to_income_val = applicant_dict['LoanAmount'] / applicant_dict['Income']
        if loan_to_income_val < 0.25 and applicant_dict['CreditScore'] >= 580:
            decision_threshold = 0.55
            
        approved = log_prob_default < decision_threshold
        approval_confidence = (1.0 - log_prob_default) if approved else log_prob_default
        
        # 2. Random Forest Prediction
        rf_prob_default = self.rf_clf.predict_proba(X_app)[0, 1]
        
        if rf_prob_default < 0.12:
            risk_tier = "Low Risk"
            risk_color = "#10b981"  # Calm Emerald Green
        elif rf_prob_default < 0.28:
            risk_tier = "Medium Risk"
            risk_color = "#f97316"  # Calm Orange
        else:
            risk_tier = "High Risk"
            risk_color = "#ef4444"  # Calm Red
            
        # Compute local feature contribution (approximate by scaled values relative to coefficients)
        coefs = self.log_reg.coef_[0]
        
        # Robust check: convert to a flat 1D dense array of float to prevent sparse representation bugs
        if hasattr(X_app, "toarray"):
            scaled_features = X_app.toarray()[0]
        else:
            scaled_features = np.asarray(X_app)[0]
            
        contributions = scaled_features * coefs
        contributions = [float(val) for val in contributions]
        
        # Match back to feature names and sort by highest positive contribution (towards Default)
        local_contributions = dict(zip(self.feature_names, contributions))
        sorted_contributions = sorted(local_contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top 3 risk factors and top 3 protective factors
        top_risk_factors = [x for x in sorted_contributions if x[1] > 0][:3]
        top_protective_factors = [x for x in reversed(sorted_contributions) if x[1] < 0][:3]
        
        # 3. K-Means Cluster Assignment
        cluster_id = self.kmeans.predict(X_app)[0]
        cluster_details = self.cluster_profiles[cluster_id]
        
        # 4. PCA Coordinate Transformation
        coords_pca = self.pca.transform(X_app)[0]
        
        return {
            'approved': approved,
            'approval_probability': 1.0 - log_prob_default,
            'decision_threshold': decision_threshold,
            'risk_probability': rf_prob_default,
            'risk_tier': risk_tier,
            'risk_color': risk_color,
            'cluster_id': cluster_id,
            'cluster_details': cluster_details,
            'pca_coords': (coords_pca[0], coords_pca[1]),
            'top_risk_factors': top_risk_factors,
            'top_protective_factors': top_protective_factors
        }
