import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("--- Starting Model Training ---")

# --- Part 1: Load Data ---
try:
    data = pd.read_csv('StressLevelDataset.csv')
    print("Dataset 'StressLevelDataset.csv' loaded successfully.")
except FileNotFoundError:
    print("\n[ERROR] 'StressLevelDataset.csv' not found.")
    print("Please make sure the dataset file is in the same folder as this script.")
    exit()

# Drop rows with missing values, if any
data = data.dropna()

# --- Part 2: 3-Class Model (Low, Medium, High) - Our App's Main Model ---
print("\n--- Training 3-Class Model (for app.py) ---")
X = data.drop('stress_level', axis=1)
y = data['stress_level']

# Ensure feature order matches app.py (though it's good practice, RF is robust to this)
X = X[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
       'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
       'noise_level', 'living_conditions', 'safety', 'basic_needs',
       'academic_performance', 'study_load', 'teacher_student_relationship',
       'future_career_concerns', 'social_support', 'peer_pressure',
       'extracurricular_activities', 'bullying']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_3class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_3class.fit(X_train, y_train)

# Save the 3-class model for the web app
joblib.dump(model_3class, 'stress_model.pkl')
print("Model 'stress_model.pkl' (3-class) saved successfully.")

# --- Part 3: Generate 3x3 Confusion Matrix (Original) ---
y_pred_3class = model_3class.predict(X_test)
accuracy_3class = accuracy_score(y_test, y_pred_3class)
print(f"3-Class Model Accuracy: {accuracy_3class * 100:.2f}%")
print(classification_report(y_test, y_pred_3class, target_names=['Low (0)', 'Medium (1)', 'High (2)']))

cm_3class = confusion_matrix(y_test, y_pred_3class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_3class, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - 3-Class Model (Low, Medium, High)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_3class.png')
print("Plot 'confusion_matrix_3class.png' saved successfully.")
plt.close()


# --- Part 4: 2-Class Model (Calm, Stressed) - For Figure 5.2 ---
print("\n--- Training 2-Class Model (for Figure 5.2) ---")
# Create a new binary target variable: 0 = 'Calm' (original Low), 1 = 'Stressed' (original Medium or High)
y_binary = y.apply(lambda x: 0 if x == 0 else 1)

# Split again with the new binary target
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

model_2class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_2class.fit(X_train_bin, y_train_bin)

y_pred_2class = model_2class.predict(X_test_bin)
accuracy_2class = accuracy_score(y_test_bin, y_pred_2class)
print(f"2-Class Model Accuracy: {accuracy_2class * 100:.2f}%")
print(classification_report(y_test_bin, y_pred_2class, target_names=['Calm (0)', 'Stressed (1)']))

# --- Part 5: Generate 2x2 Confusion Matrix (Figure 5.2) ---
cm_2class = confusion_matrix(y_test_bin, y_pred_2class)
plt.figure(figsize=(8, 6))
# We use the exact labels from your description
sns.heatmap(cm_2class, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Calm', 'Stressed'],
            yticklabels=['Calm', 'Stressed'])
plt.title('Confusion Matrix for Stress Prediction (Figure 5.2)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_2class.png') # This is your Figure 5.2
print("Plot 'confusion_matrix_2class.png' (Figure 5.2) saved successfully.")
plt.close()

print("\n--- Model Training Complete ---")

