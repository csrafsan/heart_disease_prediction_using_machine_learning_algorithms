import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache
def load_data():
    df = pd.read_csv("heartt.csv")
    df = df[["age", "sex", "ca", "trestbps", "chol","target"]]
    return df

df = load_data()

def show_explore_page():
    st.title("Explore heart disease related data.")
    st.write(
        """
        ### Real life medical data
    """)

    data = df["sex"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=False,startangle=90)
    ax1.axis("equal")#equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""## Number of Data from different features""")
    st.pyplot(fig1)



