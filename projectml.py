import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: 'imblearn' not installed. Install it using 'pip install imblearn'.")
    exit()
import joblib
import tensorflow as tf

# Set TensorFlow environment variable to disable oneDNN optimizations
os.environ['_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Data Collection and Preprocessing
# Load questionnaire data
try:
    df = pd.read_csv('saloe.csv')
except FileNotFoundError:
    print("Error: 'questionnaire.csv' not found in the current directory.")
    exit()

# Simulate real-time LMS data (login frequency, task completion, interaction duration)
np.random.seed(116)
lms_data = pd.DataFrame({
    'login_frequency': np.random.exponential(scale=2, size=len(df)),
    'task_completion_rate': np.random.uniform(0, 1, size=len(df)),
    'interaction_duration': np.random.normal(60, 15, size=len(df))
})
df = pd.concat([df, lms_data], axis=1)

# Handle missing values with median imputation
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].median(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    try:
        df[column] = label_encoder.fit_transform(df[column])
    except Exception as e:
        print(f"Error encoding column {column}: {str(e)}")
        exit()

# Apply SMOTE to address class imbalance
X = df.drop('Adaptivity Level', axis=1)
y = df['Adaptivity Level']
smote = SMOTE(random_state=116)
try:
    X_resampled, y_resampled = smote.fit_resample(X, y)
except ValueError as e:
    print(f"Error applying SMOTE: {str(e)}")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=116)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Adaptive Ensemble Model
# Gradient Boosting Machine (GBM)
try:
    gbm = GradientBoostingClassifier(learning_rate=0.05, n_estimators=100, random_state=116)
    gbm.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"Error training GBM: {str(e)}")
    exit()

# Support Vector Machine (SVM)
try:
    svm = SVC(kernel='rbf', probability=True, random_state=116)
    svm.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"Error training SVM: {str(e)}")
    exit()

# Temporal Convolutional Network (TCN)
# Reshape data for TCN (7-day sliding window simulation)
X_train_tcn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_tcn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

try:
    tcn_model = Sequential([
        Input(shape=(X_train_tcn.shape[1], 1)),
        Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
        Conv1D(filters=32, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
        GlobalAveragePooling1D(),  # Reduce temporal dimension to match target shape
        Dense(3, activation='softmax')  # 3 adaptivity levels (low, moderate, high)
    ])
    tcn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tcn_model.fit(X_train_tcn, y_train, epochs=10, batch_size=32, verbose=1)
except Exception as e:
    print(f"Error training TCN: {str(e)}")
    exit()

# Ensemble predictions (weighted combination)
try:
    gbm_pred = gbm.predict_proba(X_test_scaled)
    svm_pred = svm.predict_proba(X_test_scaled)
    tcn_pred = tcn_model.predict(X_test_tcn, verbose=0)
except Exception as e:
    print(f"Error generating predictions: {str(e)}")
    exit()

# Weights optimized via grid search (example values)
alpha, beta, gamma = 0.4, 0.3, 0.3
ensemble_pred = alpha * gbm_pred + beta * svm_pred + gamma * tcn_pred
final_pred = np.argmax(ensemble_pred, axis=1)

# 3. Reinforcement Learning (RL) for Interventions (Placeholder)
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

# Apply RL interventions
interventions = rl_intervention(final_pred)

# 4. Neural Network for Gamification (Placeholder)
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

# Apply gamification
rewards = gamification_nn(final_pred)

# 5. Save Models and Scaler
try:
    joblib.dump(gbm, 'gbm_model.pkl')
    joblib.dump(svm, 'svm_model.pkl')
    tcn_model.save('tcn_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models and scaler saved as 'gbm_model.pkl', 'svm_model.pkl', 'tcn_model.h5', and 'scaler.pkl'")
except Exception as e:
    print(f"Error saving models or scaler: {str(e)}")
    exit()

# 6. Output Sample Results
print("Sample Predictions:", final_pred[:5])
print("Sample Interventions:", interventions[:5])
print("Sample Rewards:", rewards[:5])