import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

def main_model(dataset):
    dataset = pd.read_csv(dataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # Saving model using pickle
    pickle.dump(regressor, open('model.pkl','wb'))
    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    return model

if __name__ == "__main__":
    model = main_model('salary_data.csv')
    print(model.predict([[2.8]]))