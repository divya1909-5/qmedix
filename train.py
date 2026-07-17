import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🧬 QMedix - Training Models (Correct Risk Thresholds)")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

# Generate realistic medical data
print("\n📊 Generating synthetic medical data...")
np.random.seed(42)
n_samples = 3000

def generate_medical_data(n):
    age = np.random.randint(25, 85, n)
    
    # Blood pressure (correlated with age)
    bp_mean = 110 + (age - 25) * 0.3
    blood_pressure = np.random.normal(bp_mean, 12, n)
    blood_pressure = np.clip(blood_pressure, 90, 200)
    
    # Cholesterol (correlated with age)
    chol_mean = 160 + (age - 25) * 0.7
    cholesterol = np.random.normal(chol_mean, 25, n)
    cholesterol = np.clip(cholesterol, 120, 350)
    
    # Heart rate
    heart_rate = np.random.normal(72, 10, n)
    heart_rate = np.clip(heart_rate, 50, 120)
    
    # BMI
    bmi = np.random.normal(26, 4, n)
    bmi = np.clip(bmi, 18, 45)
    
    # Glucose (correlated with BMI)
    glucose_mean = 90 + (bmi - 25) * 1.5
    glucose = np.random.normal(glucose_mean, 15, n)
    glucose = np.clip(glucose, 70, 250)
    
    # Smoking
    smoke_prob = np.where(age < 30, 0.15, np.where(age < 50, 0.25, 0.20))
    smoking = np.random.binomial(1, smoke_prob, n)
    
    return pd.DataFrame({
        'age': age,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'heart_rate': heart_rate,
        'bmi': bmi,
        'glucose': glucose,
        'smoking': smoking
    })

df = generate_medical_data(n_samples)

# ============================================
# CORRECT RISK CALCULATION
# ============================================
def calculate_risk_score(row):
    risk = 0
    
    # 1. Age Risk
    if row['age'] >= 65:
        risk += 3
    elif row['age'] >= 55:
        risk += 2
    elif row['age'] >= 45:
        risk += 1
    
    # 2. Blood Pressure Risk
    if row['blood_pressure'] >= 160:
        risk += 3
    elif row['blood_pressure'] >= 140:
        risk += 2
    elif row['blood_pressure'] >= 130:
        risk += 1
    
    # 3. Cholesterol Risk
    if row['cholesterol'] >= 280:
        risk += 3
    elif row['cholesterol'] >= 240:
        risk += 2
    elif row['cholesterol'] >= 200:
        risk += 1
    
    # 4. BMI Risk
    if row['bmi'] >= 35:
        risk += 3
    elif row['bmi'] >= 30:
        risk += 2
    elif row['bmi'] >= 27:
        risk += 1
    
    # 5. Glucose Risk
    if row['glucose'] >= 180:
        risk += 3
    elif row['glucose'] >= 140:
        risk += 2
    elif row['glucose'] >= 100:
        risk += 1
    
    # 6. Smoking Risk
    if row['smoking'] == 1:
        risk += 2
    
    # 7. Interaction Effects
    if row['age'] > 60 and row['smoking'] == 1:
        risk += 1
    if row['bmi'] > 30 and row['glucose'] > 140:
        risk += 1
    if row['blood_pressure'] > 140 and row['cholesterol'] > 240:
        risk += 1
    
    return risk

df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# ============================================
# CORRECT TARGET THRESHOLDS
# ============================================
# Low Risk: 0-3
# Medium Risk: 4-6
# High Risk: 7+

df['risk_category'] = pd.cut(df['risk_score'], 
                             bins=[-1, 3, 6, 100], 
                             labels=['Low', 'Medium', 'High'])

# Binary target: Medium/High Risk vs Low Risk (threshold at 4)
df['target'] = (df['risk_score'] >= 4).astype(int)

print(f"\n📊 Risk Score Distribution:")
print(df['risk_score'].describe())
print(f"\n📊 Risk Categories:")
print(df['risk_category'].value_counts())
print(f"\n📊 Target Distribution (Binary):")
print(f"   Low Risk (0): {sum(df['target']==0)} ({sum(df['target']==0)/len(df)*100:.1f}%)")
print(f"   Medium/High Risk (1): {sum(df['target']==1)} ({sum(df['target']==1)/len(df)*100:.1f}%)")

# Prepare features
feature_columns = ['age', 'blood_pressure', 'cholesterol', 'heart_rate', 'bmi', 'glucose', 'smoking']
X = df[feature_columns].values
y = df['target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training data: {len(X_train)} samples")
print(f"   Low Risk (0): {sum(y_train==0)} ({sum(y_train==0)/len(X_train)*100:.1f}%)")
print(f"   Medium/High Risk (1): {sum(y_train==1)} ({sum(y_train==1)/len(X_train)*100:.1f}%)")

# ============================================
# TRAIN MODELS
# ============================================

# 1. Gradient Boosting (Best for medical data)
print("\n🧠 Training Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"✅ Gradient Boosting Accuracy: {gb_acc:.4f}")

# 2. SVM
print("\n🧠 Training Support Vector Machine...")
svm = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"✅ SVM Accuracy: {svm_acc:.4f}")

# Select best model
if gb_acc >= svm_acc:
    best_model = gb
    best_name = "Gradient Boosting"
    best_acc = gb_acc
else:
    best_model = svm
    best_name = "SVM"
    best_acc = svm_acc

print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")

# ============================================
# SAVE MODELS
# ============================================
print("\n💾 Saving models...")
joblib.dump(best_model, 'models/classical_svm.pkl')
joblib.dump(svm, 'models/quantum_qsvc.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Models saved successfully!")

# ============================================
# TEST PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("🔬 Testing Predictions on Sample Patients")
print("=" * 60)

test_cases = [
    {
        'name': 'Young Healthy',
        'features': [30, 110, 170, 70, 22, 85, 0],
        'expected': '🟢 LOW RISK'
    },
    {
        'name': 'Medium Risk (Your Case)',
        'features': [50, 130, 210, 75, 26, 105, 0],
        'expected': '🟡 MEDIUM RISK'
    },
    {
        'name': 'High Risk Patient',
        'features': [70, 170, 290, 90, 34, 170, 1],
        'expected': '🔴 HIGH RISK'
    },
    {
        'name': 'Elderly Healthy',
        'features': [68, 125, 190, 72, 24, 95, 0],
        'expected': '🟢 LOW RISK'
    },
    {
        'name': 'Smoker with High BP',
        'features': [45, 145, 230, 82, 28, 110, 1],
        'expected': '🔴 HIGH RISK'
    },
    {
        'name': 'Overweight Pre-diabetic',
        'features': [55, 135, 220, 78, 29, 120, 0],
        'expected': '🟡 MEDIUM RISK'
    }
]

for case in test_cases:
    features = np.array([case['features']])
    features_scaled = scaler.transform(features)
    
    # Get prediction
    pred = best_model.predict(features_scaled)[0]
    probs = best_model.predict_proba(features_scaled)[0]
    confidence = float(max(probs))
    
    # Calculate risk percentage
    risk_pct = probs[1] * 100
    
    # Determine risk level
    if risk_pct >= 65:
        predicted = '🔴 HIGH RISK'
    elif risk_pct >= 40:
        predicted = '🟡 MEDIUM RISK'
    else:
        predicted = '🟢 LOW RISK'
    
    print(f"\n{'='*50}")
    print(f"👤 {case['name']}")
    print(f"   Expected: {case['expected']}")
    print(f"   Features: Age={case['features'][0]}, BP={case['features'][1]}, Chol={case['features'][2]}")
    print(f"   BMI={case['features'][4]}, Glucose={case['features'][5]}, Smoking={'Yes' if case['features'][6]==1 else 'No'}")
    print(f"   Prediction: {predicted}")
    print(f"   Risk Score: {risk_pct:.1f}%")
    print(f"   Confidence: {confidence:.1%}")

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("=" * 60)