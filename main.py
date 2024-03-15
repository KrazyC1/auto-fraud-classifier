import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# give it a title
st.title("Machine Learning Web App for CIS335")

st.write("""
### Explore different classifiers on different datasets
""")

dataset_name = st.sidebar.selectbox(label="Select Dataset", 
                                    options=["Auto Fraud Dataset"])
st.write(f'You have selected {dataset_name} for your experiments!')

classifier_name = st.sidebar.selectbox(label="Select Classifier", options=["SVM", "Random Forest"])
st.write(f'You have selected {classifier_name} for your experiments\'s classifier!')


def get_dataset(dataset_name):
    if dataset_name == "Auto Fraud Dataset":
        # Load the custom dataset
        data = pd.read_csv("dataset/fraud_oracle.csv")
        
        # Label encoding non-numeric columns
        non_numeric_cols = data.select_dtypes(exclude=np.number).columns
        label_encoders = {}
        for col in non_numeric_cols:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
        
        x = data.drop(columns=['FraudFound_P'])
        y = data['FraudFound_P']
        return x, y

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif classifier_name == "Random Forest":
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 2, 10)
        n_estimators = st.sidebar.slider("Number of Estimators", 2, 50)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
    return params


def get_classifier(classifier_name, parameters):
    if classifier_name == "SVM":
        clf = SVC(C=parameters["C"])
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(criterion=parameters["criterion"],
                                     max_depth=parameters["max_depth"],
                                     n_estimators=parameters["n_estimators"],
                                     random_state=42)
    return clf


x, y = get_dataset(dataset_name)

st.write(f'Shape of dataset is : {x.shape}')
st.write(f'Number of classes in the dataset is : {len(np.unique(y))}')
params = add_parameter_ui(classifier_name)
classifier = get_classifier(classifier_name, params)

#### CLASSIFICATION ####
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the classifier
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy}')
