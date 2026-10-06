import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up page configuration
st.set_page_config(page_title="Netflix Churn Predictor", layout="wide", page_icon="🎬")

st.title("🎬 Netflix Customer Churn Prediction Dashboard")
st.write("Predict whether a customer is likely to churn based on their activity.")

@st.cache_data
def load_and_train():
    # Load dataset
    df = pd.read_csv('netflix_customer_churn.csv')
    
    # Features & Target
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    # One-hot encoding for categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_encoded.columns

try:
    model, scaler, feature_columns = load_and_train()

    # Sidebar inputs for Customer Data
    st.sidebar.header("📋 Enter Customer Details")

    age = st.sidebar.slider("Age", 18, 80, 35)
    watch_hours = st.sidebar.number_input("Total Watch Hours", min_value=0.0, max_value=500.0, value=15.0)
    last_login_days = st.sidebar.number_input("Days Since Last Login", min_value=0, max_value=60, value=5)
    monthly_fee = st.sidebar.number_input("Monthly Fee ($)", min_value=5.0, max_value=25.0, value=13.99)
    number_of_profiles = st.sidebar.slider("Number of Profiles", 1, 5, 2)
    avg_watch_time_per_day = st.sidebar.number_input("Avg Watch Time Per Day (Hours)", min_value=0.0, max_value=24.0, value=1.5)

    gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
    subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    region = st.sidebar.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    device = st.sidebar.selectbox("Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Bank Transfer", "Credit Card", "Crypto", "Gift Card", "PayPal"])
    favorite_genre = st.sidebar.selectbox("Favorite Genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])

    # Prediction Action
    if st.sidebar.button("Predict Churn Risk", type="primary"):
        # Create input dataframe matching model feature schema
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Numeric inputs
        input_data['age'] = age
        input_data['watch_hours'] = watch_hours
        input_data['last_login_days'] = last_login_days
        input_data['monthly_fee'] = monthly_fee
        input_data['number_of_profiles'] = number_of_profiles
        input_data['avg_watch_time_per_day'] = avg_watch_time_per_day

        # One-hot encoded categorical inputs
        cat_mappings = {
            f'gender_{gender}': 1,
            f'subscription_type_{subscription_type}': 1,
            f'region_{region}': 1,
            f'device_{device}': 1,
            f'payment_method_{payment_method}': 1,
            f'favorite_genre_{favorite_genre}': 1
        }
        
        for col, val in cat_mappings.items():
            if col in input_data.columns:
                input_data[col] = val

        # Scale and Predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # Display Outcome Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")

        with col2:
            if prediction == 1:
                st.error("🚨 **High Churn Risk** — Customer is likely to cancel.")
            else:
                st.success("✅ **Low Churn Risk** — Customer is likely to stay.")

except FileNotFoundError:
    st.error("⚠️ `netflix_customer_churn.csv` file not found! Make sure `app.py` is saved in the exact same folder as your CSV file.")