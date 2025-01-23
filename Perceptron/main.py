import random

import pandas as pd
import matplotlib.pyplot as plt
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

average = 0

for i in range(0, 10):
    print(i)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=random.randint(0, 100))

    # Create a Perceptron, with its training parameters
    ppn = Perceptron(max_iter=40, tol=0.005, eta0=1)

    # Train the model
    ppn.fit(X_train, y_train)

    # Make predictions
    y_pred = ppn.predict(X_test)

    # Evaluate accuracy
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    average += accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()

    plt.savefig('confusion_matrix.png')

print("average" + str(average/10))
