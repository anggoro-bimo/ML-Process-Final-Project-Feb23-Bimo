import streamlit as st
import requests
from PIL import Image


# Title
st.title("Credit Card Application Approval Prediction")
st.subheader("Please fill the columns below to predict your credit card applicat ion approval.")

# Create the form to collect the predictors values
with st.form(key = "air_data_form"):
    income = st.number_input(
        label = "Please state your annual income in US$:",
        min_value = 0,
        max_value = 150000,
        help = "Value range from 0 to 800"
    )
    expenditure = st.number_input(
        label = "Please state your monthly credit card expenditure in US$:",
        min_value = 0,
        max_value = 5000,
        help = "Value range from 0 to 4000"
    )
  
    age = st.slider(
        label = "How old are you?",
        min_value = 18, 
        max_value = 99
    )
    
    reports = st.slider(
        label = "How many times do you get derogatory mark?",
        min_value = 0, 
        max_value = 15
    )
    
    dependents = st.slider(
        label = "How many dependents do you take care of?",
        min_value = 0, 
        max_value = 6
    )
    
    months = st.slider(
        label = "How many months do you live in the current address?",
        min_value = 0, 
        max_value = 540 
    )
    
    active = st.slider(
        label = "How many active credit accounts do you have currently?",
        min_value = 0, 
        max_value = 46
    )
    
    owner = st.radio(
        label = "Do you currently live in the home with your ownership status?",
        options = ("yes", "no")
    )
    
    selfemp = st.radio(
        label = "Are you a self-employed worker?",
        options = ("yes", "no")
    )
    
    majorcards = st.radio(
        label = "Do you currently held any major credit cards?",
        options = ("yes", "no")
    )
    if majorcards == "yes":
        majorcards = 1
    else:
        majorcards = 0
        
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "reports": reports,
            "age": age,
            "income": income,
            "expenditure": expenditure,
            "owner": owner,
            "selfemp": selfemp,
            "dependents": dependents,
            "months": months,
            "majorcards": majorcards,
            "active": active
        }

        # Create loading animation while predicting
        with st.spinner("Predicting your approval status, please wait..."):
            res = requests.post("http://localhost:8080/predict/", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "none":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["result"] == "rejected":
                st.warning("Your credit card application is predicted to be REJECTED!")
            else:
                st.success("Your credit card application is predicted to be APPROVED!")