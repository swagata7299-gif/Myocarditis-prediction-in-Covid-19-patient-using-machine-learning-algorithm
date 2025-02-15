import mysql.connector

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="user"
)

# Create a cursor object to interact with the database
tableName = 'predict'

# Execute SQL queries
query = f"SELECT * FROM predict"

# You can perform various database operations here

# Don't forget to commit your changes (for insert, update, delete operations)
from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

data=pd.read_sql_query(query,conn)


print(data.shape)

info = ["age","0:male, 1:female", "chest pain type, 1:typical angina, 2:atypical angina, 3:non-anginal pain, 4:asymptomatic", "resting blood pressure","serum cholestrol in mg/dl", "fasting blood sugar >120 mg/dl", "resting electrocardiographic results (values 0,1,2)", "maximum heart rate achieved", "exercise induced angina", "oldpeak=ST depression induced by exercise relative to rest","slope of the peak exercise ST segment", "number of major vessels(0-3) colored by flourosopy", "thal: 3= normal;6=fixed defect; 7=reverse defect"]

for i in range(len(info)):
    print(data.columns[i]+":\t\t\t"+info[i])
    
print(data.columns)
print(data.head())
print(data.describe())
print(data.corr())

data = data.drop_duplicates()
print(data.shape)

print(data.isnull().sum())
data = data.dropna()
print(data.isnull().sum())

data["Target"].value_counts().plot(kind="bar", color=["salmon","lightblue"])
plt.xlabel("0 = No Disease, 1 = Disease")
plt.title("Myocarditis")
plt.show()

pd.crosstab(data.Target,data.Sex).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])
plt.title("Myocarditis Disease Frequency for Sex")
plt.legend(["Female","Male"])
plt.show()

Y = data.Target

X = data.drop('Target', axis=1)

#split X and Y into train and tests data sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model1 = LogisticRegression()
model2 = RandomForestClassifier(random_state=285) #285,1673
model3 = SVC(kernel='linear' , C=10, gamma=0.0009)

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)

Y_pred1 = model1.predict(X_test)
Y_pred2 = model2.predict(X_test)
Y_pred3 = model3.predict(X_test)

acc1 = accuracy_score(Y_test, Y_pred1)  ##get the accuracy on testing data LR
print("Accuracy of Logistic Regression is {:.2f}%".format(acc1*100))

acc2 = accuracy_score(Y_test, Y_pred2)  ##get the accuracy on testing data RF
print("Accuracy of Random Forest Classifier is {:.2f}%".format(acc2*100))

acc3 = accuracy_score(Y_test, Y_pred3)  ##get the accuracy on testing data SVC
print("Accuracy of Support vector classification is {:.2f}%".format(acc3*100))





# Define the Flask app and its routes
app = Flask(__name__, template_folder='templates')

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fetch user input from the form
        user_input = {
            'Age': request.form['Age'],
            'Sex': request.form['Sex'],
            'Covid-19': request.form['Covid-19'],
    'TP': request.form['Total protein count'],
    'ALB': request.form['Total albumin count'],
    'CRP': request.form['CRP'],
    'LHD': request.form['LHD'],
    'CK_MB': request.form['CK_MB'],
    'HR': request.form['HeartRate'],
    'PLT': request.form['Platelets count'],
    'LYMPH': request.form['Lyphnocites'],
    'NU': request.form['Neurophils'],
    'BAS': request.form['Basophils']
            # Fetch other inputs in a similar manner
            # ...
        }
 # Validate that all fields are non-empty
        for key, value in user_input.items():
            if not value:
                return render_template('form1.html', error="Please fill out all fields.")

        # Convert to integers after validation
        try:
            user_input = {key: int(value) for key, value in user_input.items()}
        except ValueError:
            return render_template('form1.html', error="Invalid input. Please enter valid numbers.")
        # Predict the probability of getting PCOS for the new data
        new_data = pd.DataFrame([user_input], index=[0])

        probability = model2.predict_proba(new_data)[:, 1] * 100

        # Render the result page with the prediction
        return render_template('result1.html', probability=probability[0])

    # Render the input form initially
    return render_template('form1.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user input from the form
    if request.method == 'POST':
        user_input = {
           'Age': request.form['Age'],
            'Sex': request.form['Sex'],
            'Covid-19': request.form['Covid-19'],
    'TP': request.form['Total protein count'],
    'ALB': request.form['Total albumin count'],
    'CRP': request.form['CRP'],
    'LHD': request.form['LHD'],
    'CK_MB': request.form['CK_MB'],
    'HR': request.form['HeartRate'],
    'PLT': request.form['Platelets count'],
    'LYMPH': request.form['Lyphnocites'],
    'NU': request.form['Neurophils'],
    'BAS': request.form['Basophils']
        # Fetch other inputs in a similar manner
        # ...
    }

    # Predict the probability of getting PCOS for the new data
    new_data = pd.DataFrame([user_input])
    probability = model2.predict_proba(new_data)[:, 1] * 100

     # Round the probability to two decimal places6

    # Render the result page with the formatted prediction
    if (probability>=50):
        return render_template('result1.html', prediction_text=f"{probability}% OPPS!!! Consult doctor as soon as possible")
    else:
        return render_template('result1.html', prediction_text=f"{probability}% Hurray!!! You are safe")
    
# Your other routes...

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
if __name__ == '__main__':
    app.run(debug=False)
