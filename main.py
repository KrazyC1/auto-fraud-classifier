import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# give it a title
st.title("Machine Learning Web App")

st.write("""
### Explore different classifiers on differnt datasets
""")

dataset_name = st.sidebar.selectbox(label="Select Dataset", options=["Iris Dataset", "Cancer Dataset", "Wine Dataset"])
st.write(f'You have selected {dataset_name} for your experiments!')

classifier_name = st.sidebar.selectbox(label="Select Dataset", options=["SVM", "Random Forest","Decision Tree"])
st.write(f'You have selected {classifier_name} for your experiments\'s classifier!')



def get_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()
    elif dataset_name == "Cancer Dataset":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif classifier_name == "Random Forest":
        criterion = st.sidebar.selectbox(label="Select Crtiterion", options=["gini", "entropy", "log_loss"])
        max_depth = st.sidebar.slider("max depth", 2, 10)
        n_estimators = st.sidebar.slider("n_estimators", 2, 50)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
    elif classifier_name == "Decision Tree":
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 2, 10)
        max_features = st.sidebar.slider("Max Features", 2, 10)
        params["max_depth"] = max_depth
        params["max_features"] = max_features
        params["criterion"] = criterion     

    return params


def get_classifier(classifier_name, parameters):
    if classifier_name == "SVM":
        clf = SVC(C=params["C"])
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(criterion=parameters["criterion"],
                                     max_depth=parameters["max_depth"],
                                     n_estimators=parameters["n_estimators"])
    elif classifier_name == "Decision Tree":
        clf = DecisionTreeClassifier(criterion=parameters["criterion"],
                                     max_depth=parameters["max_depth"],
                                     max_features=parameters["max_features"])
    return clf


x, y = get_dataset(dataset_name)

st.write(f'Shape of dataset is : {x.shape}')
st.write(f'number of classes in the dataset is : {len(np.unique(y))}')
params = add_parameter_ui(classifier_name)
classifier = get_classifier(classifier_name, params)

#### CLASSIFICATION ####
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

