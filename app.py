import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/vegetable_prices.csv")

# Clean data
data = data.dropna(subset=['Price'])
data = data[data['Price'] > 0]

# Convert date
data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)

# Feature engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Filter Beet Root
beet_data = data[data['Item_Name'] == 'Beet Root']

X = beet_data[['Year','Month']]
y = beet_data['Price']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Vegetable Price Prediction")

year = st.number_input("Enter Year", min_value=2010, max_value=2035)
month = st.number_input("Enter Month", min_value=1, max_value=12)

if st.button("Predict Price"):
    
    input_data = pd.DataFrame({
        'Year':[year],
        'Month':[month]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Beet Root Price: ₹{round(prediction[0],2)}")