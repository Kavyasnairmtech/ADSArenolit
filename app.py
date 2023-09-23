import streamlit as st
import pandas as pd
import joblib

import requests

# Function to load the model from a GitHub URL
def load_model():
    github_model_url = 'https://github.com/Kavyasnairmtech/ADSArenolit/blob/main/rf_model.pkl'
    
    # Download the model file from the URL
    response = requests.get(github_model_url)
    
    if response.status_code == 200:
        # Save the downloaded model to a local file
        with open('rf_model.pkl', 'wb') as f:
            f.write(response.content)
        
        # Load the model from the local file
        model = joblib.load('rf_model.pkl')
        return model
    else:
        raise Exception(f"Failed to download the model from {github_model_url}")
# Define the Streamlit app
def main():
    st.title("Roll Classification App")
    
    # Load the model
    model = load_model()
    
    # Upload a CSV file with data
    uploaded_csv = st.file_uploader('resampled_data.csv', type=["csv"])
    
    if uploaded_csv is not None:
        st.write("### Uploaded CSV Data")
        input_data = pd.read_csv(uploaded_csv)
        st.write(input_data)
        
        # Extract the features for prediction (excluding 'target')
        features = input_data.drop(["target"], axis=1)
        
        # Make predictions using the loaded model
        predictions = model.predict(features)
        
        # Define the mapping of predicted classes
        class_mapping = {0: "Normal", 1: "Abnormal"}
        
        # Display the predictions
        st.write("### Prediction Results")
        st.write("Row-wise Predictions:")
        for i in range(len(predictions)):
            st.write(f"Row {i+1}: Actual Output = {input_data['target'].values[i]}, Predicted Output = {class_mapping.get(predictions[i], 'Unknown')}")

if __name__ == "__main__":
    main()
