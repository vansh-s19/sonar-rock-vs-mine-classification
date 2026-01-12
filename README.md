SONAR Rock vs Mine Prediction using Logistic Regression

Overview

This project implements a binary classification machine learning model that predicts whether an object detected by SONAR is a Rock or a Mine based on 60 signal features. The model is trained using Logistic Regression, a supervised learning algorithm suitable for linearly separable classification problems.

The dataset used is the well-known UCI SONAR Dataset, which contains SONAR signal returns bounced off metal cylinders (mines) and rocks.

⸻

Problem Statement

Naval SONAR systems receive reflected signals from underwater objects. Identifying whether the object is a rock or a mine is critical for maritime safety.
This model learns patterns in SONAR signals and classifies unknown objects based on those patterns.

⸻

Tech Stack
	•	Python 3.11
	•	NumPy
	•	Pandas
	•	Scikit-learn
	•	Flask (optional, for future API deployment)

⸻

Dataset

The dataset contains:
	•	60 numerical features representing SONAR signal strengths	•	1 target column
	•	R → Rock
	•	M → Mine

Source: UCI Machine Learning Repository

______

Project Structure
SONAR-Rock-vs-Mine/
│
├── data/
│   └── sonar.csv
│
├── model/
│   └── sonar_rock_vs_mine_prediction_model.py
│
├── requirements.txt
├── README.md
└── .gitignore

______

How to Run

1. Install dependencies
pip install -r requirements.txt

2. Run the model
python model/sonar_rock_vs_mine_prediction_model.py

3. Enter input

The program will ask for 60 SONAR values separated by commas.
Paste one full row from the dataset or your own SONAR reading.

Example:
0.0200,0.0371,0.0428,0.0207,...,0.0103,0.0025

The model will output:
The object is a Rock
or
The object is a Mine

______


Machine Learning Pipeline
	1.	Load and preprocess data
	2.	Split into training and test sets
	3.	Train Logistic Regression model
	4.	Evaluate accuracy
	5.	Accept live user input for prediction

______


Model Performance

The Logistic Regression model achieves approximately 80–90% accuracy on unseen test data, depending on the train-test split.

⸻

Future Improvements
	•	Feature scaling
	•	Try other models (SVM, Random Forest)
	•	Deploy as a Flask web app
	•	Add visualization of predictions
