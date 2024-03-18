import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/fraud_oracle.csv")
df.head()
 
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']
    # format all fields as string
    X.iloc[:,[1,7,30]] = X.iloc[:,[1,7,30]].astype(str)
    X = X.drop(columns=['PolicyNumber','RepNumber'])
    return X, y
 
# load the dataset
X, y = load_dataset('dataset/fraud_oracle.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

#print(X.info())