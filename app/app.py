import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Loan Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📊 Loan Eligibility Prediction Dashboard</h1>", unsafe_allow_html=True)

# ======================== Load Data and Model ========================
@st.cache_resource
def load_model():
    try:
        with open(r'C:\Users\hp\Desktop\My_apps\Loan_predictor\note_books\Loan_Random_model.pkl', 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\hp\Desktop\My_apps\Loan_predictor\data\training_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load resources
model = load_model()
data = load_data()

# Check if resources loaded correctly
if model is None or data is None:
    st.error("Failed to load model or data. Please ensure both files are in the correct location.")
    st.stop()

# ======================== Sidebar Navigation ========================
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select a page:", ["📈 Dashboard", "🔮 Single Prediction", "📊 Data Explorer", "📉 Model Analytics", "⚙️ Model Settings"])
st.info("Model Accuracy is : 76% Model : Random forest classifier")
    # ======================== PAGE 1: DASHBOARD ========================
if page == "📈 Dashboard":
    st.header("Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(data)
    ELigible_loaners = (data['Loan_Status'] == "Y").sum()
    Not_Eligible_loaners = (data['Loan_Status'] == "N").sum()
    Eligibility_Rate = (ELigible_loaners / total_customers) * 100
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Eligible Loaners", f"{ELigible_loaners:,}", delta=f"{Eligibility_Rate:.1f}%")
    col3.metric("Not Eligible Loaners", f"{Not_Eligible_loaners:,}")
    
    st.divider()
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    # Churn Distribution
    with col1:
        st.subheader("Eligibility Distribution")
        Eligiblity_counts = data['Loan_Status'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Eligible', 'Not Eligible'],
            values=[Eligiblity_counts[0], Eligiblity_counts[1]],
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textposition='inside',
            textinfo='label+percent'
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Married vs Eligibility
    with col2:
        st.subheader("Married vs Eligibilty Status")
        fig = px.box(data, x='Loan_Status', y='Married', 
                     labels={'Loan_Status': 'Eligibility Status'},
                     color='Loan_Status',
                     color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'})
        fig.update_xaxes(ticktext=['Eligible', 'Not Eligible'], tickvals=['Y', 'N'])
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    # Education vs Eligibility
    with col1:
        st.subheader("Education vs Eligibility Status")
        fig = px.box(data, x='Loan_Status', y='Education', 
                     labels={'Loan_Status': 'Eligibility Status'},
                     color='Loan_Status',
                     color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'})
        fig.update_xaxes(ticktext=['Eligible', 'Not Eligible'], tickvals=['Y', 'N'])
        st.plotly_chart(fig, use_container_width=True)
        
    
    # Credit History vs Eligibility
    with col2:
        st.subheader("Credit History vs Eligibility Status")
        fig = px.box(data, x='Loan_Status', y='Credit_History', 
                     labels={'Loan_Status': 'Eligibility Status'},
                     color='Loan_Status',
                     color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'})
        fig.update_xaxes(ticktext=['Eligible', 'Not Eligible'], tickvals=['Y', 'N'])
        st.plotly_chart(fig, use_container_width=True)

# ======================== PAGE 2: SINGLE PREDICTION ========================
elif page == "🔮 Single Prediction":
    st.header("Make a Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        married = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Married" if x == 0 else "Single")
        dependents = st.number_input("Dependents", min_value=0, max_value=3)
        CoapplicantIncome = st.number_input("Enter Co Applicant Income ($)")
        LoanAmount = st.number_input("Enter Loan Amount ($)")
       
        
    with col2:
        education = st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 0 else "Not Graduate")
        self_employed = st.selectbox("Self Employed ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ApplicantIncome = st.number_input("Enter Applicant income ($)")
        Credit_History = st.selectbox("Credit_History ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        property_area = st.selectbox("Property Area ", [0, 1, 2], format_func=lambda x: "Rural" if x == 0 else "Semi Urban" if x == 1 else "Urban")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [360],
        'Credit_History': [Credit_History],
        'Property_Area': [property_area]
            
    })
    
    st.divider()
    
    if st.button("🔮 Predict Loan Eligibility", use_container_width=True, type="primary"):
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0]
        
        st.success("✅ Prediction Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.metric("", "Not Eligible ⚠️", delta="0%", delta_color="inverse")
                st.info(f"**Eligibility Probability: {prediction_prob[1]*100:.2f}%**")
            else:
                st.metric("Churn Status", "Eligible ✅", delta="High Risk", delta_color="off")
                st.warning(f"**Eligibility Probability: {prediction_prob[1]*100:.2f}%**")
        
        with col2:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_prob[1]*100,
                title="Eligibility Risk %",
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': '#e74c3c'},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 70], 'color': "#f39c12"},
                        {'range': [70, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction details
        st.subheader("Input Summary")
        input_display = input_data.copy()
        st.dataframe(input_display, use_container_width=True)



# ======================== PAGE 4: MODEL ANALYTICS ========================
elif page == "📉 Model Analytics":
    st.header("Model Performance Analytics")
    
    # Summary statistics
    col1, col2, col3,col4 = st.columns(4)
    
    predictions = data['Loan_Status'].values
    pred_Eligibles = (data['Loan_Status'] == 'Y').sum()
    
    col1.metric("Predicted Loan Approvals", pred_Eligibles)
    col2.metric("Total Applicants", len(data))
    col3.metric("Approval Rate", f"{(pred_Eligibles/len(data)*100):.2f}%")
    col4.metric("Model Accuracy", "76%")
    
    st.divider()
    
    # Feature Importance-like analysis
    st.header("Feature Distribution Analysis")
    data['IncomeGroup'] = pd.cut(data['ApplicantIncome'], bins=[0, 2000, 4000, 6000, 8000, 12000], 
                                labels=['<2k', '2k-4k', '4k-6k', '6k-8k', '8k+'])
    income_churn = data.groupby('IncomeGroup')['Loan_Status'].apply(lambda x: (x == 'Y').sum()).to_frame(name='Approved')
    income_churn['Total'] = data.groupby('IncomeGroup')['Loan_Status'].count()
    income_churn['rate'] = (income_churn['Approved'] / income_churn['Total'] * 100)
    
    fig = px.bar(income_churn.reset_index(), x='IncomeGroup', y='rate',
                title='Loan Approval Rate by Applicant Income',
                labels={'rate': 'Approval Rate (%)', 'IncomeGroup': 'Income Group'},
                color='rate',
                color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)

# ======================== PAGE 5: MODEL SETTINGS ========================
elif page == "⚙️ Model Settings":
    st.header("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        
        **Features:** 10
        - Gender
        - Marital Status
        - Dependents
        - Education
        - Self Employed
        - Applicant Income
        - Coapplicant Income
        - Loan Amount
        - Loan Amount Term
        - Credit History
        
        **Output:** Binary Classification
        - 0 = Not Eligible
        - 1 = Eligible
        """)
    
    with col2:
        st.subheader("Data Statistics")
        stats_data = {
            'Metric': ['Total Records', 'Approved', 'Not Approved', 'Approval Rate', 'Features'],
            'Value': [
                f"{len(data):,}",
                f"{(data['Loan_Status'] == 'Y').sum():,}",
                f"{(data['Loan_Status'] == 'N').sum():,}",
                f"{((data['Loan_Status'] == 'Y').sum() / len(data) * 100):.2f}%",
                f"{len(data.columns) - 1}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("Data Sample")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.divider()
    
    # Model feature encoding info
    st.subheader("Feature Encoding Reference")
    encoding_info = {
        'Feature': ['Gender', 'Marital Status', 'Dependents', 'Education', 'Self Employed', 'Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Amount Term', 'Credit History'],
        'Encoding': [
            '0=Female, 1=Male',
            '0=Not Married, 1=Married',
            '0=0, 1=1, 2=2, 3=3+',
            '0=Not Graduate, 1=Graduate',
            '0=No, 1=Yes',
            'Numeric Value',
            'Numeric Value',
            'Numeric Value',
            'Numeric Value (in days)',
            '0=Poor Credit History, 1=Good Credit History'
        ]
    }
    encoding_df = pd.DataFrame(encoding_info)
    st.dataframe(encoding_df, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
📊 Loan Eligibility Dashboard | Powered by Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
