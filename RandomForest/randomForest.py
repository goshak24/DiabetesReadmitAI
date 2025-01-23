import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read in csv
data = pd.read_csv('data/diabetic_data.csv')

# Remove the columns that have ~0.5 or more '?'
data.drop(['weight', 'medical_specialty', 'payer_code'], axis=1, inplace=True)

# Remove unique identifiers
data.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)



# Replace <30 or >30 days readmission to YES
data['readmitted'] = data['readmitted'].replace('>30', 'YES')
data['readmitted'] = data['readmitted'].replace('<30', 'YES')

# Select target column to predict
X = data.drop(columns=['readmitted'])
y = data['readmitted']

# Encode strings to unique integers
le = LabelEncoder()

X_encoded = X.apply(le.fit_transform)
y_encoded = le.fit_transform(y)

#print(X[:5])

#Instantiate the model with 10 trees and entropy as splitting criteria
Random_Forest_model = RandomForestClassifier(n_estimators=120,criterion="entropy")

#Training/testing split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)

#Train the model
Random_Forest_model.fit(X_train, y_train)

#make predictions
y_pred = Random_Forest_model.predict(X_test)

#print(y_pred[:5])
#print(y_test[:5])

#Calculate accuracy metric
accuracy = accuracy_score(y_pred, y_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()

plt.savefig('confusion_matrix.png')

print('The accuracy is: ',accuracy*100,'%')