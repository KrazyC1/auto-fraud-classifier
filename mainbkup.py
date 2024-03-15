import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("This is my first data science application!")
    st.text("We first describe the project")

with dataset:
    st.header("Dataset description")
    st.text("This is the auto fraud dataset")
    cancer_df = pd.read_csv("dataset/fraud_oracle.csv")
    st.write(cancer_df.head())


with features:
    st.header("Here we do feature engineering")
    st.text("We do feature selection and feature engineering here")


with model_training:
    st.header("Here we train a Machine Learning model")
    st.text("We train the model here")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider(label="What should be the max depth", min_value=3, max_value=10, value=5)

    n_estimators = sel_col.selectbox(label="Select your n_estimator here", options=['100', '200', '300'], index=0)

    input_feature = sel_col.text_input("Which feature to use?", "radius_mean")
