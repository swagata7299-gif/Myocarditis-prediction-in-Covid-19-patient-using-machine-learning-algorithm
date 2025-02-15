from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from pymongo import MongoClient
app = Flask(__name__,template_folder='templates')
client = MongoClient('localhost', 27017)
db = client.heart
collection = db.disease
cursor = collection.find()
df = pd.DataFrame(list(cursor))

# Ensure numeric columns contain only numeric values
numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
df_numeric = df[numeric_columns].copy()

# Replace non-numeric values with NaN
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Impute missing values using SimpleImputer (fill NaN values with the mean)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)


# Load the trained model
# Your model training code remains unchanged
# ... (MongoDB connection and data processing)

# Assuming df_imputed has the columns 'age' and 'regular mentrual cycle or not'

# Training features and target
features = [
    'Age', 'Sex', 'Covid-19', 'TP', 'ALB',
    'CRP', 'LHD', 'CK_MB', 'HR', 'PLT',
    'LYMPH', 'NU',  'BAS'
]
df_imputed.columns = [
    'Age', 'Sex', 'Covid-19', 'TP', 'ALB',
    'CRP', 'LHD', 'CK_MB', 'HR', 'PLT',
    'LYMPH', 'NU',  'BAS','Target'
]

# Training features and target
X = df_imputed[features]
y = df_imputed['Target']

# Instantiate and fit the model before the Flask app definition
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

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

        probability = clf.predict_proba(new_data)[:, 1] * 100

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
    probability = clf.predict_proba(new_data)[:, 1] * 100

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
