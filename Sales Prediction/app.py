from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

'''
We are creating an API server
This server will listen for requests
Everything else attaches to app
'''

# Load trained components
model = joblib.load("sales_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

#When someone visits /, this function runs
@app.get("/")
def home():
    return {"message": "Sales Prediction API"}
'''
User sends data
FastAPI converts it into a Python dictionary
Function runs
'''
@app.post("/predict")
def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])

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

    return {"predicted_sales": float(prediction)}
#uvicorn app:app --reload