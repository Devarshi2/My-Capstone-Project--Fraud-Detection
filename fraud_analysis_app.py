import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Decision_Tree_Model.joblib')

# Set up the Streamlit app
st.title("Fraud Detection App")

# Input fields for the user
step = st.number_input("Step", min_value=1, max_value=1000, value=1)
transaction_type = st.selectbox("Transaction Type", options=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0, value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=0.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

# Encode the transaction type (assuming the same encoding used during training)
transaction_type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}
transaction_type_encoded = transaction_type_mapping[transaction_type]

# Button to make prediction
if st.button("Predict"):
    # Create a feature array for prediction
    features = np.array([[step, transaction_type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])

    # Make a prediction
    prediction = model.predict(features)

    # Display the result
    if prediction[0] == 1:
        st.error("This transaction is predicted to be fraudulent.")
    else:
        st.success("This transaction is predicted to be non-fraudulent.")




# 9,CASH_OUT,21922,21922,0,1929861.61,2046918.8,1
# 1,PAYMENT,4910.14,41551,36640.86,,0,0,0