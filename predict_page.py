import streamlit as st;
import pickle 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv


df = pd.read_csv('heartt.csv')
df = df[["age", "sex", "ca", "trestbps", "chol","target"]]

X = df.drop("target", axis=1)
Y = df["target"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)

y_pred_logistic = model_logistic.predict(X_test)


from sklearn.metrics import accuracy_score
logistic_Acc = accuracy_score(Y_test,y_pred_logistic)


def load_model():
    with open('model_logistic.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()


model_logistic_loaded = data["model"]

def show_predict_page():
    st.title("Heart Disease Prediction Using Machine Learning")
    st.write("""### We need some information to predict the heart disease""")


    


    
    age = st.number_input("age", 59)
    sex = st.slider("Male/Female", 0,1,)
    ca = st.number_input("ca", 0)
    trestbps = st.number_input("trestbps", 140)
    chol = st.number_input("chol", 221)
    # fbs = st.number_input("fbs",  0)
    # restecge = st.number_input("restecg",  1)
    # thalach = st.number_input("thalach", 164)
    # exang = st.number_input("exang",1)
    # oldpeak =  st.number_input(label="systolic blood pressure",step=1.,format="%.1f")
    # slope = st.number_input("slope", 2)
    # ca = st.number_input("ca", 0)
    # thal = st.number_input("thal",2)
   
    

    ok = st.button("Check Now")
    
    if ok:
        X_test = np.array([[age, sex,ca,trestbps,chol]])

        # X_test = X_test.astype(int)
        
        # st.subheader(f"Heart disease prediction is {X_test}")

        

        y_pred_logistic = model_logistic_loaded.predict(X_test)

        if(y_pred_logistic[0]==1):
            st.subheader("Heart disease prediction result is Positive!")
        else:
            st.subheader("Heart disease prediction result is Negative!")

        st.subheader(f"Accuracy of LogisticRegression is {logistic_Acc}")

        # st.subheader(f"Heart disease prediction is {y_pred_logistic[0]}")
        # st.subheader(f"Heart disease prediction is {y_pred_logistic}")
