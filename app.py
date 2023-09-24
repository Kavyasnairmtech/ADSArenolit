import streamlit as st
import pandas as pd
import joblib
import sklearn
import joblib

# Load the Random Forest model
def load_model():
    model = joblib.load("lr_retrained_model_final.pkl")  # Make sure the file name matches the model file in the same directory
    return model

# Define the Streamlit app
def main():
    st.title("Roll Classification App")
    
    # Load the model
    model = load_model()
    
    # Upload a CSV file with data
    uploaded_csv = st.file_uploader('augumented_dataset.csv', type=["csv"])
    
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
