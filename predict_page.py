import streamlit as st;
import pickle 
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()


model_logistic_loaded = data["model"]

def show_predict_page():
    st.title("Heart Disease Prediction Using Machine Learning")
    st.write("""### We need some information to predict the hear disease""")


    


    
    age = st.number_input("age", 59)
    sex = st.number_input("sex", 1)
    cp = st.number_input("cp", 1)
    trestbps = st.number_input("trestbps", 140)
    chol = st.number_input("chol", 221)
    fbs = st.number_input("fbs",  0)
    restecge = st.number_input("restecg",  1)
    thalach = st.number_input("thalach", 164)
    exang = st.number_input("exang",1)
    oldpeak =  st.number_input(label="systolic blood pressure",step=1.,format="%.1f")
    slope = st.number_input("slope", 2)
    ca = st.number_input("ca", 0)
    thal = st.number_input("thal",2)
   
    

    ok = st.button("Check Now")
    
    if ok:
        X_test = np.array([[age, sex, cp,trestbps,chol,fbs,restecge,thalach,exang,oldpeak,slope,ca,thal ]])

        X_test = X_test.astype(int)
        
        st.subheader(f"Heart disease prediction is {X_test}")

        

        y_pred_logistic = model_logistic_loaded.predict(X_test)
        st.subheader(f"Heart disease prediction is {y_pred_logistic[0]:.1f}")
        st.subheader(f"Heart disease prediction is {y_pred_logistic}")
