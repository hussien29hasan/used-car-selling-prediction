import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Load the original data
    data = pd.read_csv('reg_cars_selling.csv')
    return data

# Load raw data for EDA
raw_data = load_data()

# Main title
st.title("🚗 Car Price Prediction Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Price Prediction"])

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("Shape of dataset:", raw_data.shape)
    st.write("Sample of the data:")
    st.dataframe(raw_data.head())
    
    st.header("Statistical Summary")
    st.dataframe(raw_data.describe())
    
    st.header("Missing Values")
    missing_data = pd.DataFrame({
        'Missing Values': raw_data.isnull().sum(),
        'Percentage': (raw_data.isnull().sum() / len(raw_data)) * 100
    })
    st.dataframe(missing_data)

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=raw_data, x='selling_price', bins=30)
        plt.title("Distribution of Car Prices")
        st.pyplot(fig)
        
        st.subheader("Price by Fuel Type")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=raw_data, x='fuel', y='selling_price')
        plt.xticks(rotation=45)
        plt.title("Car Prices by Fuel Type")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Price vs Year")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=raw_data, x='year', y='selling_price')
        plt.title("Car Prices vs Manufacturing Year")
        st.pyplot(fig)
        
        st.subheader("Price by Transmission")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=raw_data, x='transmission', y='selling_price')
        plt.title("Car Prices by Transmission Type")
        st.pyplot(fig)

elif page == "Price Prediction":
    st.header("Car Price Prediction")
    
    # Preprocessing function
    @st.cache_data
    def preprocess_data(data):
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Add missing columns if they don't exist
        required_columns = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 
                           'owner', 'mileage', 'engine', 'max_power', 'seats']
        
        for col in required_columns:
            if col not in data.columns:
                if col == 'seats':
                    data[col] = 5  # default value
                else:
                    data[col] = 0  # default value
        
        # Extract numeric values
        def extract_numeric(value):
            if isinstance(value, str):
                numbers = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
                return float(numbers) if numbers else np.nan
            return value

        # Process numeric columns
        numeric_cols = ['mileage', 'engine', 'max_power']
        for col in numeric_cols:
            data[col] = data[col].apply(extract_numeric)
        
        # Encode categorical variables
        categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
        encoded_df = pd.get_dummies(data[categorical_columns], drop_first=True)
        
        # Feature engineering
        data['vehicle_age'] = 2024 - data['year']
        
        # Drop unnecessary columns and concatenate encoded ones
        data = data.drop(categorical_columns + ['name', 'year'] if 'name' in data.columns else categorical_columns + ['year'], axis=1)
        data = pd.concat([data, encoded_df], axis=1)
        
        # Handle missing values and scale
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_columns]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        data_processed = pd.DataFrame(
            scaler.fit_transform(
                imputer.fit_transform(data)
            ),
            columns=data.columns,
            index=data.index
        )
        
        return data_processed, imputer, scaler
    
    # Process data and train models
    data_processed, imputer, scaler = preprocess_data(raw_data)
    X = data_processed.drop('selling_price', axis=1)
    y = data_processed['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    @st.cache_resource
    def train_models():
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        
        return lr, rf
    
    lr, rf = train_models()
    
    # Input form
    st.subheader("Enter Car Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=2024, value=2020)
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
        fuel = st.selectbox("Fuel Type", options=raw_data['fuel'].unique())
    
    with col2:
        transmission = st.selectbox("Transmission", options=raw_data['transmission'].unique())
        owner = st.selectbox("Owner", options=raw_data['owner'].unique())
        seller_type = st.selectbox("Seller Type", options=raw_data['seller_type'].unique())
    
    with col3:
        mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=20.0)
        engine = st.number_input("Engine (CC)", min_value=0, value=1200)
        max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=100.0)
    
    # Make prediction
    if st.button("Predict Price"):
        # Prepare input data
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power]
        })
        
        # Process input data
        input_processed, _, _ = preprocess_data(input_data)
        
        # Make predictions (standardized)
        lr_pred_std = lr.predict(input_processed)[0]
        rf_pred_std = rf.predict(input_processed)[0]
        
        # Inverse transform predictions to get actual prices
        selling_price_std = raw_data['selling_price'].std()
        selling_price_mean = raw_data['selling_price'].mean()
        
        lr_pred_actual = (lr_pred_std * selling_price_std) + selling_price_mean
        rf_pred_actual = (rf_pred_std * selling_price_std) + selling_price_mean
        
        # Display predictions
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Linear Regression Prediction: ₹{lr_pred_actual:,.2f}")
        with col2:
            st.info(f"Random Forest Prediction: ₹{rf_pred_actual:,.2f}")
        
        # Feature importance
        if st.checkbox("Show Feature Importance"):
            st.subheader("Feature Importance (Random Forest)")
            importances = pd.DataFrame({
                "Feature": X.columns,
                "Importance": rf.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            
            st.bar_chart(importances.set_index("Feature")) 