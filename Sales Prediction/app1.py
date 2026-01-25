import streamlit as st
import joblib
import pandas as pd

st.title("Sales Prediction App")

model = joblib.load("sales_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.write("Enter the input values: ")

order_date = st.date_input("Order Date")
product_name = st.text_input("Product Name")
category = st.text_input("Category")
region = st.text_input("Region")
quantity = st.number_input("Quantity", min_value=1)
profit = st.number_input("Profit", min_value=0.0)

if st.button("Predict Sales"):
    input_data = {
        "Order Date": order_date,
        "Product Name": product_name,
        "Category": category,
        "Region": region,
        "Quantity": quantity,
        "Profit": profit
    }

    df = pd.DataFrame([input_data])

    # Date processing
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df = df.drop("Order Date", axis=1)

    # Encode categorical data
    categorical_cols = ["Product Name", "Category", "Region"]
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    st.success(f"Predicted Sales: {prediction:.2f}")

