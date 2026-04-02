import streamlit as st
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Real-Time Credit Card Fraud Detection")
st.markdown("This dashboard interfaces with a machine learning backend API to predict whether a credit card transaction is fraudulent.")

# Sidebar
st.sidebar.header("Options")
mode = st.sidebar.radio("Select Mode", ["Manual Entry", "Batch Upload (CSV)"])

if mode == "Manual Entry":
    st.subheader("Manual Transaction Input")
    st.write("Enter the transaction details below:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Input fields
    time_val = col1.number_input("Time", value=0.0)
    amount_val = col2.number_input("Amount", value=150.0)
    
    v_vals = {}
    for i in range(1, 29):
        col = [col1, col2, col3, col4][i % 4]
        v_vals[f"V{i}"] = col.number_input(f"V{i}", value=0.0)
        
    if st.button("Predict Fraud"):
        payload = {"Time": time_val, "Amount": amount_val, **v_vals}
        
        with st.spinner("Analyzing transaction via API..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    is_fraud = data["is_fraud"]
                    prob = data["fraud_probability"]
                    
                    if is_fraud:
                        st.error(f"🚨 FRAUD DETECTED! (Probability: {prob:.2%})")
                    else:
                        st.success(f"✅ Transaction is Safe. (Fraud Probability: {prob:.2%})")
                    
                    st.write(f"Execution Time: {data['execution_time_sec']} seconds")
                    
                    # Explainability Feature Importance
                    shap_vals = data.get("shap_values")
                    if shap_vals:
                        st.subheader("Feature Importance (SHAP)")
                        
                        # Sort and display top 10 features
                        feature_names = list(payload.keys())
                        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_vals})
                        shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs()
                        shap_df = shap_df.sort_values(by='Abs_SHAP', ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]
                        ax.barh(shap_df['Feature'][::-1], shap_df['SHAP Value'][::-1], color=colors[::-1])
                        ax.set_title("Top 10 Features Influencing This Prediction (Red = Push Towards Fraud)")
                        ax.set_xlabel("SHAP Value")
                        
                        st.pyplot(fig)
                        
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Error: {e}")

elif mode == "Batch Upload (CSV)":
    st.subheader("Batch Predictions via CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} transactions.")
        st.dataframe(df.head())
        
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for index, row in df.iterrows():
                try:
                    # Ignore 'Class' if it exists in the test CSV
                    payload = row.drop('Class', errors='ignore').to_dict()
                    resp = requests.post(API_URL, json=payload)
                    if resp.status_code == 200:
                        pred_data = resp.json()
                        results.append(pred_data["is_fraud"])
                    else:
                        results.append(None)
                except Exception:
                    results.append(None)
                progress_bar.progress((index + 1) / len(df))
                
            df["Predicted_Fraud"] = results
            st.success("Batch processing complete!")
            st.dataframe(df)
            
            # Simple bar chart
            fraud_counts = df["Predicted_Fraud"].value_counts()
            fig, ax = plt.subplots()
            
            # Make sure both categories are present if possible
            categories = ['Normal', 'Fraud']
            counts = [
                fraud_counts.get(False, 0),
                fraud_counts.get(True, 0)
            ]
            
            ax.bar(categories, counts, color=['green', 'red'])
            ax.set_title("Fraud vs Normal Transactions (Predicted)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
