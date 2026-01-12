import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data collection and Data Processing"""

#loading the dataset to pandas Dataframe

sonar_data = pd.read_csv("/Users/vanshsaxena/Documents/Machine Learning Models/SONAR Rock vs Mine Prediction (Logistic Regression/sonar data.csv", header=None)

#Separating Datas and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

"""Training and Testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1 )

"""Model Training -> **LOGISTIC REGRESSION MODEL**"""

model = LogisticRegression(max_iter=1000)

"""Training the Logistic Regression model with training data"""

model.fit(X_train, Y_train)

"""**MODEL EVALUATION**"""

 #Accuracy on training Data
X_train_Prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_Prediction)

#print('Accuracy on training data : ',training_data_accuracy)

 #Accuracy on test data
X_test_Prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_Prediction)

#print('Accuracy on test data: ',test_data_accuracy)

"""MAKING A PREDICTIVE SYSTEM"""

print("Enter 60 SONAR values separated by commas:")
user_input = input().strip()
input_data = tuple(float(x) for x in user_input.split(","))

if len(input_data) != 60:
    raise ValueError(f"Expected 60 values, but got {len(input_data)}")

#changing the input data as Numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

Prediction = model.predict(input_data_reshaped)
print(Prediction)

if(Prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')