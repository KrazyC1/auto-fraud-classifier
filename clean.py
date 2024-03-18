# this is where we inspect and clean our data, 
# make it ready to put into streamlit
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("dataset/fraud_oracle.csv")

# inspect dataset
df.info()

# split data into train and test
x = df.drop(columns=['FraudFound_P'])
y = df['FraudFound_P']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train_zscore = (x_train - x_train.mean()) / (x_train.std())

from sklearn.decomposition import PCA

var_list = []
for i in range(5):
    pca = PCA(n_components=i)
    principalComponents = pca.fit_transform(x_train_zscore)
    var = principalComponents.explained_variance_ratio_
    var_list.append(var)

print(var_list)


