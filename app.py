import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_loader import load_dataset
from models import LoanMLPipeline

# 1. Page Configuration & Custom Premium Styling
st.set_page_config(
    page_title="FinVantage | Intelligent Loan Decisioning",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Composed Dark UI Styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0f19 !important;
    font-family: 'Inter', sans-serif !important;
    color: #cbd5e1 !important; /* Highly readable Slate-300 */
}

[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* Ensure all main content headers are highly visible crisp off-white */
[data-testid="stAppViewContainer"] h1, 
[data-testid="stAppViewContainer"] h2, 
[data-testid="stAppViewContainer"] h3, 
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
    color: #f8fafc !important; /* Crisp Slate-50 */
    font-family: 'Outfit', sans-serif !important;
}

/* Force excellent contrast on ALL widget labels, headers, and descriptions across the app */
label, 
.stWidgetLabel, 
[data-testid="stWidgetLabel"] p, 
[data-testid="stWidgetLabel"] label, 
[data-testid="stWidgetLabel"] span,
div[data-testid="stWidgetLabel"] {
    color: #f1f5f9 !important; /* Crisp high-contrast Slate-100 */
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important; /* Solid composed Dark Slate */
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Force excellent contrast on ALL sidebar texts and labels */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0 !important; /* Crisp Slate-200 for maximum readability */
    font-family: 'Inter', sans-serif !important;
}

.main-title {
    font-size: 2.8rem;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #38bdf8 !important; /* Solid Calm Ocean Sky-Blue for perfect rendering */
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Premium Solid Slate Cards (High legibility) */
.glass-card {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    margin-bottom: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-1px);
}

/* Custom Color-Accented KPI Cards (Stripe/Plaid fintech grade) */
.kpi-card-blue {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #0ea5e9 !important; /* Sky Blue accent line */
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-indigo {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #6366f1 !important; /* Indigo accent line */
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-orange {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #f97316 !important; /* Calm Orange accent line */
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-red {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #ef4444 !important; /* Calm Red accent line */
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

/* Solid Composed Decision Badges (No cheap pulses or glows) */
.decision-card-approve {
    background: rgba(16, 185, 129, 0.08) !important; /* Stable calm emerald background */
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}

.decision-card-reject {
    background: rgba(239, 68, 68, 0.08) !important; /* Stable calm red background */
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}

.badge-text-approve {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #10b981 !important;
    letter-spacing: 0.1em;
}

.badge-text-reject {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ef4444 !important;
    letter-spacing: 0.1em;
}

/* Custom metrics */
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 2px;
}

.metric-label {
    font-size: 0.85rem;
    color: #cbd5e1; /* Clear Slate-300 labels */
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Interactive elements overrides (Eliminate stark white browser boxes) */
form[data-testid="stForm"] {
    background-color: #111827 !important; /* Deep Slate-900 border matching */
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 28px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
}

/* Number inputs, Text inputs, Selectboxes inner widgets */
.stNumberInput input, 
.stTextInput input, 
.stNumberInput button,
div[data-testid="stNumberInput"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-baseweb="select"],
div[data-baseweb="select"] div,
div[data-baseweb="select"] span {
    background-color: #0f172a !important; /* Deep Slate-950 background */
    color: #f8fafc !important; /* Slate-50 bright text */
    border-color: #334155 !important;
}

/* Ensure the active text in selectboxes is fully legible */
div[data-baseweb="select"] [data-baseweb="select"] div[aria-selected="true"],
div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[role="button"] {
    color: #f8fafc !important;
}

/* Remove default white outline or borders */
div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

.stNumberInput input:focus, 
.stTextInput input:focus,
div[data-baseweb="select"]:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
}

/* Global popover, dropdown list, and option overrides (Force dark theme on body-level portals) */
div[data-baseweb="popover"],
div[data-testid="stVirtualDropdown"],
div[role="listbox"],
ul[role="listbox"],
[data-testid="stSelectboxVirtualList"] {
    background-color: #1e293b !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
}

div[data-baseweb="popover"] ul,
div[data-testid="stVirtualDropdown"] ul {
    background-color: #1e293b !important;
    background: #1e293b !important;
}

div[data-baseweb="popover"] li[role="option"],
div[data-testid="stVirtualDropdown"] li[role="option"],
li[role="option"],
div[role="option"] {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
}

/* Style child text inside dropdown options to prevent ugly background blocks */
div[data-baseweb="popover"] li[role="option"] *,
div[data-testid="stVirtualDropdown"] li[role="option"] *,
li[role="option"] *,
div[role="option"] * {
    background-color: transparent !important;
    background: transparent !important;
    color: #cbd5e1 !important;
}

/* Hover/active option state overrides */
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-testid="stVirtualDropdown"] li[role="option"]:hover,
li[role="option"]:hover,
div[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"],
div[data-testid="stVirtualDropdown"] li[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
    background-color: #0284c7 !important;
    background: #0284c7 !important;
    color: #ffffff !important;
}

/* Ensure child text color inside hovered options is white */
div[data-baseweb="popover"] li[role="option"]:hover *,
div[data-testid="stVirtualDropdown"] li[role="option"]:hover *,
li[role="option"]:hover *,
div[role="option"]:hover * {
    color: #ffffff !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important; /* Composed calm blue gradient */
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.3) !important;
}

/* Tab labels styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: rgba(30, 41, 59, 0.5);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    font-family: 'Outfit', sans-serif;
    border: none !important;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important; /* Active sky blue */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Session State Initialization
if 'active_applicant' not in st.session_state:
    st.session_state.active_applicant = {
        'Age': 38,
        'Income': 72000,
        'LoanAmount': 42000,
        'CreditScore': 685,
        'MonthsEmployed': 48,
        'NumCreditLines': 4,
        'InterestRate': 8.5,
        'LoanTerm': 36,
        'DTIRatio': 0.32,
        'Education': "Bachelor's",
        'EmploymentType': 'Full-time',
        'MaritalStatus': 'Married',
        'HasMortgage': 'No',
        'HasDependents': 'Yes',
        'LoanPurpose': 'Home',
        'HasCoSigner': 'No'
    }

# 3. Data Loading & Model Training Caching
@st.cache_data
def get_dataset():
    """Loads dataset and outputs status parameters"""
    return load_dataset()

# Load primary data
df, is_synthetic, filepath = get_dataset()

@st.cache_resource
def get_trained_pipeline(_df, version=3):
    """Trains scikit-learn models and returns trained pipeline wrapper"""
    pipeline = LoanMLPipeline(n_clusters=4, random_state=42)
    df_clustered, df_pca = pipeline.fit(_df)
    return pipeline, df_clustered, df_pca

# Initialize ML Pipeline
with st.spinner("Initializing FinVantage Predictive Core... Please wait..."):
    pipeline, df_clustered, df_pca = get_trained_pipeline(df, version=3)

# 4. Sidebar System Telemetry
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/bank.png", width=70)
    st.markdown("<h2 style='margin-top:0px;'>FinVantage Portal</h2>", unsafe_allow_html=True)
    
    # Telemetry Status
    st.markdown("### 📡 Database Telemetry")
    if is_synthetic:
        st.markdown(
            f"""
            <div style='background-color:rgba(255, 145, 0, 0.1); border:1px solid #ff9100; border-radius:8px; padding:12px; margin-bottom:15px;'>
                <strong style='color:#ff9100;'>Mode: Synthetic Fallback</strong><br>
                <span style='font-size:0.8rem; color:#94a3b8;'>Real Kaggle CSV not found. Running high-fidelity simulator.</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:rgba(0, 230, 118, 0.1); border:1px solid #00e676; border-radius:8px; padding:12px; margin-bottom:15px;'>
                <strong style='color:#00e676;'>Mode: Kaggle CSV Connected</strong><br>
                <span style='font-size:0.8rem; color:#94a3b8;'>Connected to real dataset at <code>{os.path.basename(filepath)}</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Quick Metrics
    st.markdown("---")
    st.markdown("### 📊 Portfolio Metrics")
    st.metric(label="Total Borrowers Analysed", value=f"{len(df):,}")
    
    default_rate = df['Default'].mean() * 100
    st.metric(label="Average Default Rate", value=f"{default_rate:.2f}%")
    
    avg_credit = df['CreditScore'].mean()
    st.metric(label="Avg Portfolio Credit Score", value=f"{int(avg_credit)}")
    
    # Export Data Option
    st.markdown("---")
    st.markdown("### 📥 Database Export")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Active Dataset (CSV)",
        data=csv_data,
        file_name="active_loan_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<br><p style='text-align:center; font-size:0.75rem; color:#64748b;'>FinVantage ML Platform v1.0.0</p>", unsafe_allow_html=True)

# 5. Header Section
st.markdown("<h1 class='main-title'>FinVantage</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>State-of-the-Art Loan Approval & Credit Risk Intelligence System</p>", unsafe_allow_html=True)

# 6. Tab Definitions
tab_dash, tab_pred, tab_seg, tab_pca = st.tabs([
    "📈 Executive Analytics", 
    "🎯 Credit Decisioning Form", 
    "👥 Behavioral Cohorts", 
    "🌌 Dimensional Mapping (PCA)"
])

# ==============================================================================
# TAB 1: EXECUTIVE ANALYTICS
# ==============================================================================
with tab_dash:
    st.markdown("### Portfolio Analytics & Insights")
    st.write("Quantitative overview of demographics, metrics, and key performance indicators of the credit portfolio.")
    
    # 4 KPI Blocks (Color accented Fintech style)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(
            f"""
            <div class='kpi-card-blue'>
                <div class='metric-label'>Total Active Capital</div>
                <div class='metric-value'>${df['LoanAmount'].sum() / 1e6:.1f}M</div>
                <div style='font-size:0.8rem; color:#10b981; font-weight:600;'>↑ Healthy liquidity pool</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div class='kpi-card-indigo'>
                <div class='metric-label'>Avg Loan size</div>
                <div class='metric-value'>${df['LoanAmount'].mean():,.0f}</div>
                <div style='font-size:0.8rem; color:#cbd5e1;'>Avg Term: {df['LoanTerm'].mean():.1f} months</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div class='kpi-card-orange'>
                <div class='metric-label'>Portfolio Risk Tier</div>
                <div class='metric-value' style='color:#f97316;'>MODERATE</div>
                <div style='font-size:0.8rem; color:#f97316; font-weight:600;'>Avg Interest Rate: {df['InterestRate'].mean():.2f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    with kpi_col4:
        st.markdown(
            f"""
            <div class='kpi-card-red'>
                <div class='metric-label'>Default Risk Ratio</div>
                <div class='metric-value' style='color:#ef4444;'>{default_rate:.2f}%</div>
                <div style='font-size:0.8rem; color:#cbd5e1;'>Trained Accuracy: {pipeline.rf_metrics['accuracy']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    # Interactive Plots Row 1
    plot_row1_c1, plot_row1_c2 = st.columns(2)
    
    with plot_row1_c1:
        st.markdown("#### Credit Score & Interest Rate Default Distribution")
        # Sample data to keep Plotly responsive
        sample_df = df.sample(min(3000, len(df)))
        sample_df['Default Status'] = sample_df['Default'].map({1: 'Defaulted', 0: 'Paid / Normal'})
        
        fig1 = px.scatter(
            sample_df,
            x='CreditScore',
            y='InterestRate',
            color='Default Status',
            color_discrete_map={'Defaulted': '#f44336', 'Paid / Normal': '#00e676'},
            opacity=0.6,
            labels={'CreditScore': 'Credit Score (300-850)', 'InterestRate': 'Interest Rate (%)'},
            hover_data=['Age', 'Income', 'LoanAmount']
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig1.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig1.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig1, use_container_width=True)
        
    with plot_row1_c2:
        st.markdown("#### Default Probability by Loan Purpose")
        purpose_default = df.groupby('LoanPurpose')['Default'].mean().reset_index()
        purpose_default['Default %'] = purpose_default['Default'] * 100
        purpose_default = purpose_default.sort_values(by='Default %', ascending=False)
        
        fig2 = px.bar(
            purpose_default,
            x='LoanPurpose',
            y='Default %',
            color='Default %',
            color_continuous_scale=['#00e676', '#ff9100', '#f44336'],
            labels={'LoanPurpose': 'Loan Purpose', 'Default %': 'Historical Default Rate (%)'}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            coloraxis_showscale=False
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig2, use_container_width=True)

    # Interactive Plots Row 2
    plot_row2_c1, plot_row2_c2 = st.columns(2)
    
    with plot_row2_c1:
        st.markdown("#### Debt-to-Income (DTI) vs. Annual Income density")
        fig3 = px.density_contour(
            sample_df,
            x='Income',
            y='DTIRatio',
            color='Default Status',
            color_discrete_map={'Defaulted': '#f44336', 'Paid / Normal': '#00e676'},
            labels={'Income': 'Annual Income ($)', 'DTIRatio': 'Debt-to-Income Ratio'},
            marginal_x="histogram",
            marginal_y="histogram"
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    with plot_row2_c2:
        st.markdown("#### Employment Status & Credit Quality Breakdown")
        fig4 = px.box(
            sample_df,
            x='EmploymentType',
            y='CreditScore',
            color='Default Status',
            color_discrete_map={'Defaulted': '#f44336', 'Paid / Normal': '#00e676'},
            labels={'EmploymentType': 'Employment Status', 'CreditScore': 'Credit Score'}
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1'
        )
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig4, use_container_width=True)

# ==============================================================================
# TAB 2: CREDIT DECISIONING FORM
# ==============================================================================
with tab_pred:
    st.markdown("### Interactive Applicant Appraisal System")
    st.write("Input applicant metrics below to run real-time dual-model evaluations (Logistic Regression Decisioning & Random Forest Risk scoring).")
    
    # Grid of inputs
    input_form = st.form(key='decision_form')
    
    with input_form:
        st.markdown("#### 1. Demographic & Core Financials")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            age = st.slider("Borrower Age", min_value=18, max_value=70, value=st.session_state.active_applicant['Age'], step=1)
        with col_f2:
            income = st.number_input("Annual Income ($)", min_value=10000, max_value=250000, value=st.session_state.active_applicant['Income'], step=5000)
        with col_f3:
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=2000, max_value=300000, value=st.session_state.active_applicant['LoanAmount'], step=5000)
        with col_f4:
            credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=st.session_state.active_applicant['CreditScore'], step=5)
            
        st.markdown("#### 2. Employment & Credit History")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            months_employed = st.slider("Months Employed", min_value=0, max_value=120, value=st.session_state.active_applicant['MonthsEmployed'], step=1)
        with col_c2:
            num_credit_lines = st.slider("Active Credit Lines", min_value=1, max_value=20, value=st.session_state.active_applicant['NumCreditLines'], step=1)
        with col_c3:
            interest_rate = st.slider("Proposed Interest Rate (%)", min_value=1.0, max_value=30.0, value=st.session_state.active_applicant['InterestRate'], step=0.1)
        with col_c4:
            loan_term = st.selectbox("Loan Term", options=[12, 24, 36, 48, 60], index=[12, 24, 36, 48, 60].index(st.session_state.active_applicant['LoanTerm']))
            
        st.markdown("#### 3. Risk Ratios & Profiles")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.05, max_value=0.95, value=st.session_state.active_applicant['DTIRatio'], step=0.01)
        with col_r2:
            education = st.selectbox("Education Level", options=['High School', "Bachelor's", "Master's", 'PhD'], index=['High School', "Bachelor's", "Master's", 'PhD'].index(st.session_state.active_applicant['Education']))
        with col_r3:
            employment_type = st.selectbox("Employment Type", options=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], index=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'].index(st.session_state.active_applicant['EmploymentType']))
        with col_r4:
            marital_status = st.selectbox("Marital Status", options=['Single', 'Married', 'Divorced'], index=['Single', 'Married', 'Divorced'].index(st.session_state.active_applicant['MaritalStatus']))
            
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            has_mortgage = st.selectbox("Has Mortgage?", options=['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.active_applicant['HasMortgage']))
        with col_k2:
            has_dependents = st.selectbox("Has Dependents?", options=['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.active_applicant['HasDependents']))
        with col_k3:
            loan_purpose = st.selectbox("Loan Purpose", options=['Home', 'Auto', 'Education', 'Business', 'Other'], index=['Home', 'Auto', 'Education', 'Business', 'Other'].index(st.session_state.active_applicant['LoanPurpose']))
        with col_k4:
            has_cosigner = st.selectbox("Has Co-Signer?", options=['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.active_applicant['HasCoSigner']))
            
        submit_btn = st.form_submit_button("Run Credit Evaluation Pipeline", type="primary")

    # If form submitted, update session state and compute
    if submit_btn:
        st.session_state.active_applicant = {
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
            'HasCoSigner': has_cosigner
        }
        st.success("Applicant metrics updated and processed through credit cores!")

    # Perform prediction based on session state
    pred_res = pipeline.predict_applicant(st.session_state.active_applicant)

    # Display Results Side-by-Side
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("#### ⚙️ Credit Decision Engine (Logistic Regression)")
        if pred_res['approved']:
            st.markdown(
                f"""
                <div class='decision-card-approve'>
                    <div class='badge-text-approve'>APPROVED</div>
                    <p style='color:#e2e8f0; margin-top:15px; font-size:1.05rem;'>
                        Applicant satisfies regulatory compliance and internal probability thresholds.<br>
                        <strong>Approval Confidence: {pred_res['approval_probability']*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='decision-card-reject'>
                    <div class='badge-text-reject'>REJECTED</div>
                    <p style='color:#e2e8f0; margin-top:15px; font-size:1.05rem;'>
                        Applicant displays elevated risk profile exceeding tolerance threshold of {pred_res['decision_threshold']*100:.0f}%.<br>
                        <strong>Risk Probability: {pred_res['risk_probability']*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Random Forest Risk Level")
        
        # Plotly gauge indicator (Mathematically centered)
        risk_pct = pred_res['risk_probability'] * 100
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Score: {pred_res['risk_tier']}", 'font': {'size': 16, 'color': '#cbd5e1', 'family': 'Outfit'}},
            number = {'font': {'size': 44, 'color': '#ffffff', 'family': 'Outfit'}, 'suffix': '%'},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': pred_res['risk_color']},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, 12], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [12, 28], 'color': 'rgba(249, 115, 22, 0.15)'},
                    {'range': [28, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            height=200,
            margin=dict(l=30, r=30, t=55, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("#### 🔍 Risk Contribution Driver Analysis")
        st.write("Identifies which of the applicant's variables push them towards default (risk drivers) or safety (protective factors).")
        
        # Prepare Plotly chart of feature contributions
        risk_drivers = pred_res['top_risk_factors']
        protective_factors = pred_res['top_protective_factors']
        
        drivers_df = pd.DataFrame(risk_drivers + protective_factors, columns=['Feature', 'Coefficient Impact'])
        # Sort so highest risk is at the top
        drivers_df = drivers_df.sort_values(by='Coefficient Impact', ascending=True)
        
        # Make feature names readable
        drivers_df['Feature'] = drivers_df['Feature'].str.replace('num__', '').str.replace('cat__', '').str.replace('_', ' ')
        
        # Color code: red for risk drivers, green for protective
        drivers_df['Impact Category'] = np.where(drivers_df['Coefficient Impact'] > 0, 'Risk Driver (+)', 'Protective Factor (-)')
        
        fig_drivers = px.bar(
            drivers_df,
            y='Feature',
            x='Coefficient Impact',
            color='Impact Category',
            color_discrete_map={'Risk Driver (+)': '#ef4444', 'Protective Factor (-)': '#10b981'},
            orientation='h',
            labels={'Coefficient Impact': 'Strength of Influence', 'Feature': 'Applicant Feature'}
        )
        fig_drivers.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_drivers.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_drivers.update_yaxes(showgrid=False)
        st.plotly_chart(fig_drivers, use_container_width=True)
        
        # Text summaries
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#38bdf8; font-family:Outfit; margin-top:0px; margin-bottom:15px;'>📝 Underwriter's Decision Support Notes</h5>", unsafe_allow_html=True)
        
        primary_driver = risk_drivers[0][0].replace('num__', '').replace('cat__', '').replace('_', ' ') if len(risk_drivers) > 0 else 'None'
        primary_protector = protective_factors[0][0].replace('num__', '').replace('cat__', '').replace('_', ' ') if len(protective_factors) > 0 else 'None'
        
        is_low_exposure = (st.session_state.active_applicant['LoanAmount'] / st.session_state.active_applicant['Income'] < 0.25) and (st.session_state.active_applicant['CreditScore'] >= 580)
        
        if pred_res['approved']:
            stipulations = "Standard pricing is approved. Consider matching with optimal cohort benefits."
            if is_low_exposure:
                stipulations = "Bypassed standard rejection via the **Low-Exposure Underwriting Override** (loan-to-income is exceptionally healthy at {:.1f}%).".format((st.session_state.active_applicant['LoanAmount'] / st.session_state.active_applicant['Income']) * 100)
            st.markdown(
                f"""
                <ul>
                    <li><strong>Decision Outcome</strong>: <span style='color:#10b981; font-weight:600;'>APPROVED</span></li>
                    <li><strong>Key Strength</strong>: High credit health driven primarily by <strong>{primary_protector}</strong>.</li>
                    <li><strong>Key Risk Exposure</strong>: Marginal risk presented by <strong>{primary_driver}</strong>, well within acceptable thresholds.</li>
                    <li><strong>Stipulations</strong>: {stipulations}</li>
                </ul>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <ul>
                    <li><strong>Decision Outcome</strong>: <span style='color:#ef4444; font-weight:600;'>REJECTED</span></li>
                    <li><strong>Primary Reject Code</strong>: Elevated risk driven by <strong>{primary_driver}</strong>.</li>
                    <li><strong>Mitigating Factor</strong>: Moderate credit protection offered by <strong>{primary_protector}</strong> was insufficient.</li>
                    <li><strong>Action Plan</strong>: Deny loan or recommend a <strong>Co-Signer</strong>, increase collateral, or adjust requested amount below <strong>${loan_amount * 0.6:,.0f}</strong>.</li>
                </ul>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("🔍 Deep Underwriting Diagnostics (Logistic Regression)"):
            st.markdown("This section displays the mathematical breakdown of the Logistic Regression decision.")
            
            # Compute diagnostics
            app_dict = st.session_state.active_applicant.copy()
            app_dict['LoanToIncomeRatio'] = app_dict['LoanAmount'] / app_dict['Income']
            df_diag = pd.DataFrame([app_dict])
            X_diag = pipeline.preprocessor.transform(df_diag)
            if hasattr(X_diag, "toarray"):
                scaled_vals = X_diag.toarray()[0]
            else:
                scaled_vals = np.asarray(X_diag)[0]
            coefs = pipeline.log_reg.coef_[0]
            contributions = scaled_vals * coefs
            intercept = pipeline.log_reg.intercept_[0]
            
            diag_rows = []
            for name, scaled_val, coef, contrib in zip(pipeline.feature_names, scaled_vals, coefs, contributions):
                clean_name = name.replace("num__", "").replace("cat__", "").replace("_", " ")
                diag_rows.append({
                    "Feature": clean_name,
                    "Scaled Value": f"{scaled_val:.4f}",
                    "Model Coefficient": f"{coef:.4f}",
                    "Logit Contribution": f"{contrib:.4f}"
                })
            
            diag_df = pd.DataFrame(diag_rows)
            st.dataframe(diag_df, use_container_width=True)
            
            total_logit = sum(contributions) + intercept
            final_prob = 1.0 / (1.0 + np.exp(-total_logit))
            
            st.markdown(f"**Model Intercept**: `{intercept:.4f}`")
            st.markdown(f"**Sum of Contributions**: `{sum(contributions):.4f}`")
            st.markdown(f"**Total Logit Score**: `{total_logit:.4f}`")
            st.markdown(f"**Default Probability**: `{final_prob*100:.2f}%` (Threshold: `15.00%`)")

# ==============================================================================
# TAB 3: BEHAVIORAL COHORTS (K-MEANS)
# ==============================================================================
with tab_seg:
    st.markdown("### K-Means Customer Cohorts & Segmentation")
    st.write("Our algorithm groups the entire loan portfolio into 4 distinct borrower personas. Find applicant's classification and banking offers below.")
    
    # Active Applicant Segment Allocation
    app_cluster_id = pred_res['cluster_id']
    app_cluster_details = pred_res['cluster_details']
    
    st.markdown(
        f"""
        <div style='background-color:#1e293b; border-left:6px solid {app_cluster_details['color']}; border-radius:12px; padding:20px; margin-bottom:25px; border:1px solid rgba(255,255,255,0.05);'>
            <span style='font-size:0.9rem; color:#cbd5e1; text-transform:uppercase;'>Current Applicant Behavioral Cohort</span>
            <h3 style='margin:5px 0px; color:{app_cluster_details['color']}; font-family:Outfit; font-weight:700;'>{app_cluster_details['name']}</h3>
            <p style='margin:0px; font-size:1.05rem; color:#f8fafc;'>{app_cluster_details['description']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Cards layout for all segments
    st.markdown("#### Portfolio Segments Summary")
    seg_cols = st.columns(4)
    
    for c in range(4):
        prof = pipeline.cluster_profiles[c]
        is_active = (c == app_cluster_id)
        border_style = f"border: 2px solid {prof['color']};" if is_active else "border: 1px solid rgba(255,255,255,0.08);"
        
        # Remove all leading spacing inside f-string lines to prevent Streamlit from interpreting it as an indented code block
        card_html = f"""<div style='background:#1e293b; border-radius:12px; padding:20px; {border_style} height:360px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);'>
<div style='height:25px;'>
{"<span style='background:#0f172a; color:#ffffff; font-size:0.75rem; padding:4px 8px; border-radius:12px; font-weight:600; float:right;'>ACTIVE APPLICANT</span>" if is_active else ""}
</div>
<h4 style='color:{prof['color']}; margin-top:5px; font-family:Outfit; font-weight:700; margin-bottom:5px;'>{prof['name']}</h4>
<p style='font-size:0.8rem; color:#cbd5e1; height:60px; overflow:hidden; margin-bottom:10px;'>{prof['description']}</p>
<hr style='border: 0; height: 1px; background: rgba(255,255,255,0.1); margin:10px 0;'>
<table style='width:100%; font-size:0.85rem; color:#f1f5f9;'>
<tr>
<td style='padding:4px 0; color:#cbd5e1;'>Avg Credit Score</td>
<td style='text-align:right; font-weight:600;'>{prof['avg_credit_score']}</td>
</tr>
<tr>
<td style='padding:4px 0; color:#cbd5e1;'>Avg Income</td>
<td style='text-align:right; font-weight:600;'>${prof['avg_income']:,}</td>
</tr>
<tr>
<td style='padding:4px 0; color:#cbd5e1;'>Avg Loan Amount</td>
<td style='text-align:right; font-weight:600;'>${prof['avg_loan_amount']:,}</td>
</tr>
<tr>
<td style='padding:4px 0; color:#cbd5e1;'>Avg DTI Ratio</td>
<td style='text-align:right; font-weight:600;'>{prof['avg_dti']:.2f}</td>
</tr>
<tr>
<td style='padding:4px 0; color:#ef4444;'>Default Probability</td>
<td style='text-align:right; font-weight:700; color:#ef4444;'>{prof['default_rate']*100:.1f}%</td>
</tr>
</table>
</div>"""
        with seg_cols[c]:
            st.markdown(card_html, unsafe_allow_html=True)
            
    # Business actions row
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Personalized Action Plan & Marketing Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    # 1. Tailored Banking Products Card
    if app_cluster_id == 0 or "Prime" in app_cluster_details['name']:
        prod_list = """<ul>
<li><strong>Primary Product</strong>: <em>Vantage Platinum Cash-Back Credit Card</em> (0% introductory APR for 18 months).</li>
<li><strong>Secondary Product</strong>: Pre-approved for a home equity line of credit (HELOC) up to <strong>$50,000</strong>.</li>
<li><strong>Wealth Management</strong>: Connect with a premier private banking portfolio advisor.</li>
</ul>"""
    elif app_cluster_id == 1 or "Debt" in app_cluster_details['name']:
        prod_list = """<ul>
<li><strong>Primary Product</strong>: <em>Credit Consolidation Mortgage Option</em> (merge high-interest credit lines into structured 3.5% APR loans).</li>
<li><strong>Educational Material</strong>: Automate debt payoffs using the debt avalanche utility in the FinVantage app.</li>
<li><strong>Risk Protection</strong>: Premium overdraft protection with direct deposit alert system.</li>
</ul>"""
    elif app_cluster_id == 2 or "Young" in app_cluster_details['name']:
        prod_list = """<ul>
<li><strong>Primary Product</strong>: <em>FinVantage Builder Savings Cohort</em> (Earn 4.8% APY on regular savings with no minimum balance).</li>
<li><strong>Reward Structure</strong>: Students and young graduates enjoy cash-back matching on student loan repayments.</li>
<li><strong>Credit Line</strong>: Offer micro-limit overdraft with automated repayment templates.</li>
</ul>"""
    else:
        prod_list = """<ul>
<li><strong>Primary Product</strong>: <em>Secured Credit Rebuilder Package</em> (requires deposits as collateral to rebuild payment histories safely).</li>
<li><strong>Rate Adjuster</strong>: Offer refinancing options that lower interest rates automatically by 0.5% for every 6 consecutive on-time monthly payments.</li>
<li><strong>Alternative Decisions</strong>: Direct to co-signer matching templates.</li>
</ul>"""

    # 2. Underwriting Actions Card
    if app_cluster_id == 0 or "Prime" in app_cluster_details['name']:
        underwrite_list = """<ul>
<li><strong>Pricing Action</strong>: Offer <strong>-0.25% discount</strong> off standard interest rates.</li>
<li><strong>Collateral Terms</strong>: Unsecured credit is highly recommended; skip mortgage/collateral checks.</li>
<li><strong>Approval Path</strong>: Auto-approval enabled; standard audits can be bypassed.</li>
</ul>"""
    elif app_cluster_id == 1 or "Debt" in app_cluster_details['name']:
        underwrite_list = """<ul>
<li><strong>Pricing Action</strong>: Standard market pricing.</li>
<li><strong>Collateral Terms</strong>: Require verification of current home mortgage and existing auto loan titles.</li>
<li><strong>Approval Path</strong>: Enhanced verification of debt-to-income limits via certified bank statements.</li>
</ul>"""
    elif app_cluster_id == 2 or "Young" in app_cluster_details['name']:
        underwrite_list = """<ul>
<li><strong>Pricing Action</strong>: Standard credit pricing.</li>
<li><strong>Collateral Terms</strong>: No mortgage requirement, but encourage a Co-signer if the debt-to-income exceeds 40%.</li>
<li><strong>Approval Path</strong>: Automatic approval with digital verification of employment contract.</li>
</ul>"""
    else:
        underwrite_list = """<ul>
<li><strong>Pricing Action</strong>: <strong>+1.5% premium loading</strong> to protect bank margins against elevated default ratios.</li>
<li><strong>Collateral Terms</strong>: Require a hard co-signer signature or lien against active assets (e.g. savings, vehicle).</li>
<li><strong>Approval Path</strong>: Mandatory secondary manual underwriting review.</li>
</ul>"""

    with rec_col1:
        st.markdown(
            f"""
            <div class='glass-card' style='height:280px;'>
                <h5 style='color:#38bdf8; font-family:Outfit; margin-top:0px; margin-bottom:15px;'>💳 Tailored Banking Products</h5>
                <div style='color:#cbd5e1; font-size:0.95rem; line-height:1.6;'>
                    {prod_list}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with rec_col2:
        st.markdown(
            f"""
            <div class='glass-card' style='height:280px;'>
                <h5 style='color:#38bdf8; font-family:Outfit; margin-top:0px; margin-bottom:15px;'>🛡️ Underwriting Actions & Pricing Adjustments</h5>
                <div style='color:#cbd5e1; font-size:0.95rem; line-height:1.6;'>
                    {underwrite_list}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ==============================================================================
# TAB 4: DIMENSIONAL MAPPING (PCA)
# ==============================================================================
with tab_pca:
    st.markdown("### Principal Component Analysis (PCA) Dimensional Reduction")
    st.write("This tab visualizes the complex, high-dimensional space of the entire loan database into 2 principal components. Your applicant is mapped LIVE as the neon gold marker!")
    
    pca_c1, pca_c2 = st.columns([3, 1])
    
    with pca_c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("##### 🌌 Map Settings")
        color_by = st.radio(
            "Color Scatter Plot By:",
            options=["Customer Segments", "Default Status"],
            index=0
        )
        
        # Summary details of active applicant in PCA space
        coord_x, coord_y = pred_res['pca_coords']
        st.markdown("---")
        st.markdown("##### 📍 Active Applicant Coordinates")
        st.markdown(f"**PC1 (Income/Credit Factor)**: `{coord_x:.3f}`")
        st.markdown(f"**PC2 (Loan/Debt Factor)**: `{coord_y:.3f}`")
        st.markdown(
            f"""
            <div style='margin-top:15px; font-size:0.8rem; color:#94a3b8;'>
                PC1 captures variance in borrower assets and credit ratings. PC2 correlates with loan sizes and debt ratios.
            </div>
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with pca_c1:
        # Sample data to plot
        sample_size = min(2000, len(df_pca))
        plot_pca_df = df_pca.sample(sample_size, random_state=42).copy()
        
        # Map labels
        plot_pca_df['Customer Segments'] = plot_pca_df['Cluster'].map(
            lambda x: pipeline.cluster_profiles[x]['name']
        )
        plot_pca_df['Default Status'] = plot_pca_df['Default'].map({1: 'Defaulted', 0: 'Paid / Normal'})
        
        # Define color maps
        seg_color_map = {pipeline.cluster_profiles[i]['name']: pipeline.cluster_profiles[i]['color'] for i in range(4)}
        default_color_map = {'Defaulted': '#f44336', 'Paid / Normal': '#00e676'}
        
        active_color_map = seg_color_map if color_by == "Customer Segments" else default_color_map
        
        # Plotly Scatter
        fig_pca = px.scatter(
            plot_pca_df,
            x='PC1',
            y='PC2',
            color=color_by,
            color_discrete_map=active_color_map,
            opacity=0.4,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            hover_data=['Age', 'Income', 'LoanAmount', 'CreditScore']
        )
        
        # Add Active Applicant point as a large golden star
        fig_pca.add_trace(go.Scatter(
            x=[coord_x],
            y=[coord_y],
            mode='markers',
            marker=dict(
                color='#f59e0b', # Composed Gold/Amber
                size=18,
                symbol='star',
                line=dict(color='#ffffff', width=2)
            ),
            name='Active Applicant',
            hoverinfo='text',
            text=f"<b>ACTIVE APPLICANT</b><br>Score: {st.session_state.active_applicant['CreditScore']}<br>Income: ${st.session_state.active_applicant['Income']:,}<br>Loan: ${st.session_state.active_applicant['LoanAmount']:,}"
        ))
        
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig_pca.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_pca.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        
        st.plotly_chart(fig_pca, use_container_width=True)
