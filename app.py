import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load models and scaler with error handling
try:
    gbm_model = joblib.load('gbm_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    tcn_model = tf.keras.models.load_model('tcn_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading models or scaler: {str(e)}")
    gbm_model, svm_model, tcn_model, scaler = None, None, None, None

# Define mappings for categorical variables
categorical_columns = [
    'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
    'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
    'Network Type', 'Class Duration', 'Self Lms', 'Device'
]
label_encoders = {col: LabelEncoder() for col in categorical_columns}
label_encoders['Gender'].fit(['boy', 'girl'])
label_encoders['Age'].fit(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'])
label_encoders['Education Level'].fit(['school', 'college', 'university'])
label_encoders['Institution Type'].fit(['government', 'non government'])
label_encoders['IT Student'].fit(['yes', 'no'])
label_encoders['Location'].fit(['yes', 'no'])
label_encoders['Load-shedding'].fit(['low', 'high'])
label_encoders['Financial Condition'].fit(['poor', 'mid', 'rich'])
label_encoders['Internet Type'].fit(['wifi', 'mobile data'])
label_encoders['Network Type'].fit(['2g', '3g', '4g'])
label_encoders['Class Duration'].fit(['0-1', '1-3', '3-6', '6-10'])
label_encoders['Self Lms'].fit(['yes', 'no'])
label_encoders['Device'].fit(['tab', 'mobile', 'computer'])

lms_columns = ['login_frequency', 'task_completion_rate', 'interaction_duration']
adaptivity_encoder = LabelEncoder().fit(['Low', 'Moderate', 'High'])

def rl_intervention(pred_score):
    if pred_score == 0: return "Recommend additional resources and peer support"
    if pred_score == 1: return "Suggest time management workshops"
    return "Maintain current support"

def gamification_nn(pred_score):
    if pred_score == 0: return "Bronze badge: Keep going!"
    if pred_score == 1: return "Silver badge: Great progress!"
    return "Gold badge: Outstanding effort!"

@app.route('/')
def home(): return render_template('home.html')
@app.route('/predict', methods=['GET'])
def index(): return render_template('index.html')
@app.route('/adapt')
def adapt(): return render_template('adaptivity.html')
@app.route('/resources')
def resources(): return render_template('resources.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_to_df_map = {
            'gender': 'Gender', 'age': 'Age', 'education_level': 'Education Level',
            'institute_type': 'Institution Type', 'it_student': 'IT Student', 'location': 'Location',
            'load_shedding': 'Load-shedding', 'financial_condition': 'Financial Condition',
            'internet_type': 'Internet Type', 'network_type': 'Network Type',
            'class_duration': 'Class Duration', 'self_lms': 'Self Lms', 'device': 'Device'
        }
        form_data = {df_col: request.form[form_key] for form_key, df_col in form_to_df_map.items()}

        np.random.seed(116)
        lms_data = {
            'login_frequency': np.random.exponential(scale=2, size=1)[0],
            'task_completion_rate': np.random.uniform(0, 1, size=1)[0],
            'interaction_duration': np.random.normal(60, 15, size=1)[0]
        }
        
        encoded_data = [label_encoders[col].transform([form_data[col]])[0] for col in categorical_columns]
        input_data = pd.DataFrame([encoded_data + list(lms_data.values())], columns=categorical_columns + lms_columns)
        
        scaled_data = scaler.transform(input_data)
        tcn_data = scaled_data.reshape((1, scaled_data.shape[1], 1))
        
        alpha, beta, gamma = 0.4, 0.3, 0.3
        gbm_pred = gbm_model.predict_proba(scaled_data)
        svm_pred = svm_model.predict_proba(scaled_data)
        tcn_pred = tcn_model.predict(tcn_data, verbose=0)
        ensemble_pred = alpha * gbm_pred + beta * svm_pred + gamma * tcn_pred
        final_pred_score = np.argmax(ensemble_pred, axis=1)[0]

        return render_template('result.html',
                               adaptivity_level=adaptivity_encoder.inverse_transform([final_pred_score])[0],
                               intervention=rl_intervention(final_pred_score),
                               reward=gamification_nn(final_pred_score),
                               form_data=form_data)

    except Exception as e:
        return f"<h1>Error during prediction</h1><p>{str(e)}</p><p>Please check that all form fields are filled out correctly.</p>"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        form_data = {col: request.form[col] for col in categorical_columns}

        np.random.seed(116)
        lms_data = {
            'login_frequency': np.random.exponential(scale=2, size=1)[0],
            'task_completion_rate': np.random.uniform(0, 1, size=1)[0],
            'interaction_duration': np.random.normal(60, 15, size=1)[0]
        }
        
        encoded_data = [label_encoders[col].transform([form_data[col]])[0] for col in categorical_columns]
        input_data = pd.DataFrame([encoded_data + list(lms_data.values())], columns=categorical_columns + lms_columns)
        
        recommendations = []
        changeable_features = {
            'Internet Type': ['wifi', 'mobile data'], 'Network Type': ['2g', '3g', '4g'],
            'Device': ['tab', 'mobile', 'computer'], 'Class Duration': ['0-1', '1-3', '3-6', '6-10'],
        }

        for feature, all_options in changeable_features.items():
            original_user_choice = form_data[feature]
            best_option_for_feature = None
            highest_score_for_feature = -1

            for option in all_options:
                temp_input_data = input_data.copy()
                temp_input_data[feature] = label_encoders[feature].transform([option])[0]
                
                scaled_temp = scaler.transform(temp_input_data)
                tcn_temp = scaled_temp.reshape((1, scaled_temp.shape[1], 1))
                
                alpha, beta, gamma = 0.4, 0.3, 0.3
                ensemble_p = alpha * gbm_model.predict_proba(scaled_temp) + beta * svm_model.predict_proba(scaled_temp) + gamma * tcn_model.predict(tcn_temp, verbose=0)
                prediction_score = np.argmax(ensemble_p, axis=1)[0]
                
                if prediction_score > highest_score_for_feature:
                    highest_score_for_feature = prediction_score
                    best_option_for_feature = option
            
            # **FIXED LOGIC:** Show the optimal setting if it's different from the user's choice.
            if best_option_for_feature != original_user_choice:
                recommendations.append(
                    f"For '<b>{feature}</b>', the model suggests '<b>{best_option_for_feature}</b>' is the optimal setting."
                )
        
        return render_template('recommendations.html', recommendations=recommendations)

    except Exception as e:
        return render_template('recommendations.html', recommendations=[f"Error generating recommendations: {str(e)}"])

if __name__ == '__main__':
    app.run(debug=True)
