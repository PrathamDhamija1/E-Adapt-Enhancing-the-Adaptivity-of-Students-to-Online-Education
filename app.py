import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the saved models and scaler with error handling
try:
    gbm_model = joblib.load('gbm_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    tcn_model = tf.keras.models.load_model('tcn_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading models or scaler: {str(e)}")
    gbm_model, svm_model, tcn_model, scaler = None, None, None, None

# Define the label encoder mappings for categorical variables
label_encoders = {}
categorical_columns = [
    'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student', 
    'Location', 'Load-shedding', 'Financial Condition', 'Internet Type', 
    'Network Type', 'Class Duration', 'Self Lms', 'Device'
]
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    if col == 'Gender':
        label_encoders[col].fit(['boy', 'girl'])
    elif col == 'Age':
        label_encoders[col].fit(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'])
    elif col == 'Education Level':
        label_encoders[col].fit(['school', 'college', 'university'])
    elif col == 'Institution Type':
        label_encoders[col].fit(['government', 'non government'])
    elif col == 'IT Student':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Location':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Load-shedding':
        label_encoders[col].fit(['low', 'high'])
    elif col == 'Financial Condition':
        label_encoders[col].fit(['poor', 'mid', 'rich'])
    elif col == 'Internet Type':
        label_encoders[col].fit(['wifi', 'mobile data'])
    elif col == 'Network Type':
        label_encoders[col].fit(['2g', '3g', '4g'])
    elif col == 'Class Duration':
        label_encoders[col].fit(['0-1', '1-3', '3-6', '6-10'])
    elif col == 'Self Lms':
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'Device':
        label_encoders[col].fit(['tab', 'mobile', 'computer'])

# LMS data columns (from e_adapt_ml.py)
lms_columns = ['login_frequency', 'task_completion_rate', 'interaction_duration']

# Label encoder for the target variable
adaptivity_encoder = LabelEncoder()
adaptivity_encoder.fit(['Low', 'Moderate', 'High'])

# RL intervention function (from e_adapt_ml.py)
def rl_intervention(predictions, gamma=0.9):
    interventions = []
    for pred in predictions:
        if pred == 0:  # Low adaptivity
            interventions.append("Recommend additional resources and peer support")
        elif pred == 1:  # Moderate adaptivity
            interventions.append("Suggest time management workshops")
        else:  # High adaptivity
            interventions.append("Maintain current support")
    return interventions

# Gamification function (from e_adapt_ml.py)
def gamification_nn(predictions):
    rewards = []
    for pred in predictions:
        if pred == 0:
            rewards.append("Bronze badge: Keep going!")
        elif pred == 1:
            rewards.append("Silver badge: Great progress!")
        else:
            rewards.append("Gold badge: Outstanding effort!")
    return rewards

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/adapt')
def adapt():
    return render_template('adaptivity.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if any(model is None for model in [gbm_model, svm_model, tcn_model, scaler]):
            return render_template('result.html', 
                                 adaptivity_level="Error: One or more models/scaler not loaded. Check server logs.")

        # Get form data for categorical variables
        form_data = {
            'Gender': request.form['gender'],
            'Age': request.form['age'],
            'Education Level': request.form['education_level'],
            'Institution Type': request.form['institute_type'],
            'IT Student': request.form['it_student'],
            'Location': request.form['location'],
            'Load-shedding': request.form['load_shedding'],
            'Financial Condition': request.form['financial_condition'],
            'Internet Type': request.form['internet_type'],
            'Network Type': request.form['network_type'],
            'Class Duration': request.form['class_duration'],
            'Self Lms': request.form['self_lms'],
            'Device': request.form['device']
        }

        # Validate Class Duration format
        class_duration = form_data['Class Duration']
        if '-' not in class_duration or len(class_duration.split('-')) != 2:
            return render_template('result.html', 
                                 adaptivity_level=f"Error: Invalid Class Duration format: {class_duration}")

        # Simulate LMS data (as in e_adapt_ml.py)
        np.random.seed(116)
        lms_data = {
            'login_frequency': np.random.exponential(scale=2, size=1)[0],
            'task_completion_rate': np.random.uniform(0, 1, size=1)[0],
            'interaction_duration': np.random.normal(60, 15, size=1)[0]
        }

        # Encode categorical inputs
        encoded_data = []
        for col in categorical_columns:
            try:
                encoded_value = label_encoders[col].transform([form_data[col]])[0]
                encoded_data.append(encoded_value)
            except Exception as e:
                return render_template('result.html', 
                                     adaptivity_level=f"Error encoding {col}: {str(e)}")

        # Combine categorical and LMS data
        input_data = pd.DataFrame([encoded_data + list(lms_data.values())], 
                                 columns=categorical_columns + lms_columns)

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Prepare TCN input
        tcn_data = scaled_data.reshape((1, scaled_data.shape[1], 1))

        # Ensemble predictions
        gbm_pred = gbm_model.predict_proba(scaled_data)
        svm_pred = svm_model.predict_proba(scaled_data)
        tcn_pred = tcn_model.predict(tcn_data, verbose=0)

        # Weighted combination (from e_adapt_ml.py)
        alpha, beta, gamma = 0.4, 0.3, 0.3
        ensemble_pred = alpha * gbm_pred + beta * svm_pred + gamma * tcn_pred
        final_pred = np.argmax(ensemble_pred, axis=1)[0]

        # Decode the prediction
        adaptivity_level = adaptivity_encoder.inverse_transform([final_pred])[0]

        # Apply RL interventions and gamification
        intervention = rl_intervention([final_pred])[0]
        reward = gamification_nn([final_pred])[0]

        return render_template('result.html', 
                             adaptivity_level=adaptivity_level,
                             intervention=intervention,
                             reward=reward)

    except Exception as e:
        return render_template('result.html', 
                             adaptivity_level=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
