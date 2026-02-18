"""
LSI Predictive AI Model - Interactive Web Interface
A Streamlit-based dashboard for LSI prediction and monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="LSI Predictive Model",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .header-style {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Set random seed
np.random.seed(42)

@st.cache_data
def load_data():
df = pd.read_csv("AI_LSI_Demo_Historical_Data.csv")
    return df  

# Train models
@st.cache_resource
def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gb
    
    return models

# Main Title
st.markdown('<p class="header-style">💧 LSI Predictive AI Model</p>', unsafe_allow_html=True)
st.markdown('**Langelier Saturation Index Prediction Dashboard**')
st.markdown('---')

# Load data
df = load_data()
df = pd.read_csv("AI_LSI_Demo_Historical_Data.csv")
    return df  
# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", 
    ["Dashboard", "Predictions", "Model Analysis", "Historical Data"])

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df), "samples")
    with col2:
        st.metric("Avg LSI", f"{df['LSI'].mean():.4f}", "units")
    with col3:
        st.metric("LSI Min", f"{df['LSI'].min():.4f}", "units")
    with col4:
        st.metric("LSI Max", f"{df['LSI'].max():.4f}", "units")
    
    st.markdown("---")
    
    # LSI Trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**LSI Trend Over Time**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['LSI'], linewidth=1.5, color='#1f77b4')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral (LSI=0)')
        ax.fill_between(df.index, df['LSI'], alpha=0.3, color='#1f77b4')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('LSI Value')
        ax.set_title('LSI Historical Trend')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**LSI Distribution**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df['LSI'], bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.axvline(df['LSI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["LSI"].mean():.4f}')
        ax.set_xlabel('LSI Value')
        ax.set_ylabel('Frequency')
        ax.set_title('LSI Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Statistics
    st.write("**Detailed Statistics**")
    stats_df = df[['Temperature_C', 'Flow_m3_h', 'pH', 'Calcium_mg_L', 'Alkalinity_mg_L', 'TDS_mg_L', 'LSI']].describe()
    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# PAGE 2: PREDICTIONS
# ============================================================================
elif page == "Predictions":
    st.subheader("🔮 Real-time LSI Prediction")
    
    # Prepare data
    X = df.drop(['Date', 'LSI'], axis=1)
    y = df['LSI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # User Input Form
    st.write("**Enter Water Quality Parameters:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input(
            "Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=38.6,
            step=0.1
        )
    with col2:
        flow = st.number_input(
            "Flow Rate (m³/h)",
            min_value=1100.0,
            max_value=1250.0,
            value=1182.8,
            step=1.0
        )
    with col3:
        ph = st.number_input(
            "pH",
            min_value=6.5,
            max_value=7.5,
            value=7.09,
            step=0.01
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        calcium = st.number_input(
            "Calcium (mg/L)",
            min_value=80.0,
            max_value=115.0,
            value=97.3,
            step=0.1
        )
    with col2:
        alkalinity = st.number_input(
            "Alkalinity (mg/L)",
            min_value=170.0,
            max_value=200.0,
            value=185.5,
            step=0.1
        )
    with col3:
        tds = st.number_input(
            "TDS (mg/L)",
            min_value=27500.0,
            max_value=30000.0,
            value=28726.2,
            step=10.0
        )
    
    # Make predictions
    if st.button("Predict LSI", use_container_width=True):
        input_data = np.array([[temp, flow, ph, calcium, alkalinity, tds]])
        input_scaled = scaler.transform(input_data)
        
        st.markdown("---")
        st.write("**Prediction Results:**")
        
        col1, col2, col3 = st.columns(3)
        
        for idx, (model_name, model) in enumerate(models.items()):
            prediction = model.predict(input_scaled)[0]
            
            # Determine risk level
            if prediction > 0.1:
                risk = "SCALING RISK"
                color = "🔴"
            elif prediction < -0.15:
                risk = "CORROSION RISK"
                color = "🔴"
            else:
                risk = "BALANCED"
                color = "🟢"
            
            with col1 if idx == 0 else (col2 if idx == 1 else col3):
                st.metric(
                    model_name,
                    f"{prediction:.6f}",
                    f"{risk} {color}"
                )
        
        # Explanation
        st.markdown("---")
        st.info("""
        **LSI Interpretation:**
        - **LSI > 0.1**: Water tends to precipitate CaCO₃ (Scaling Risk)
        - **-0.15 < LSI < 0.1**: Water is balanced and stable
        - **LSI < -0.15**: Water tends to dissolve CaCO₃ (Corrosion Risk)
        """)

# ============================================================================
# PAGE 3: MODEL ANALYSIS
# ============================================================================
elif page == "Model Analysis":
    st.subheader("📈 Model Performance Analysis")
    
    # Prepare data
    X = df.drop(['Date', 'LSI'], axis=1)
    y = df['LSI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Model Performance Comparison
    st.write("**Model Performance Comparison:**")
    
    performance_data = []
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        performance_data.append({
            'Model': model_name,
            'R² Score': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MSE': mse
        })
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)
    
    st.markdown("---")
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**R² Score Comparison**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(perf_df['Model'], perf_df['R² Score'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('R² Score')
        ax.set_xlim(0.99, 1.0)
        for i, v in enumerate(perf_df['R² Score']):
            ax.text(v - 0.0001, i, f'{v:.6f}', ha='right', va='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**MAE Comparison**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(perf_df['Model'], perf_df['MAE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Mean Absolute Error')
        for i, v in enumerate(perf_df['MAE']):
            ax.text(v + 0.00001, i, f'{v:.6f}', ha='left', va='center')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Feature Importance
    st.write("**Feature Importance (Random Forest)**")
    
    rf_model = models['Random Forest']
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='#9467bd')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance for LSI Prediction')
    for i, v in enumerate(importance_df['Importance']):
        ax.text(v + 0.01, i, f'{v:.4f}', ha='left', va='center')
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 4: HISTORICAL DATA
# ============================================================================
elif page == "Historical Data":
    st.subheader("📋 Historical Data View")
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.slider("Start Index", 0, len(df) - 1, 0)
    with col2:
        end_idx = st.slider("End Index", start_idx + 1, len(df), len(df))
    
    # Display filtered data
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    
    st.markdown("---")
    
    # Download data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="lsi_historical_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Correlation Analysis
    st.write("**Correlation Analysis**")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[['Temperature_C', 'Flow_m3_h', 'pH', 'Calcium_mg_L', 'Alkalinity_mg_L', 'TDS_mg_L', 'LSI']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.3f')
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>LSI Predictive AI Model Dashboard | Created with Streamlit</p>
    <p>Last Updated: February 5, 2026</p>
</div>
""", unsafe_allow_html=True)
