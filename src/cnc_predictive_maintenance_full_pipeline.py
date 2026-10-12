"""
CNC Milling Machine - Predictive Maintenance
Full Pipeline: Preprocessing → EDA → Model Training → Evaluation → Alertin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_excel('cncmilingdata2023-2026.xlsx',
                   sheet_name='Sensor_Data')
numeric_cols = [
    'Spindle_Speed_RPM',
    'Feed_Rate_mm_per_min',
    'Cutting_Depth_mm',
    'Vibration_X_mm_s',
    'Vibration_Y_mm_s',
    'Vibration_Z_mm_s',
    'Spindle_Temp_C',
    'Coolant_Temp_C',
    'Coolant_Flow_L_per_min',
    'Power_Consumption_kW',
    'Tool_Wear_Percent',
    'Acoustic_Emission_dB',
    'Servo_Load_X_Percent',
    'Servo_Load_Y_Percent',
    'Servo_Load_Z_Percent',
    'Surface_Roughness_Ra_um'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df.dtypes)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn types:\n{df.dtypes}")


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# Check for nulls
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop non-feature columns
df_clean = df.drop(columns=['Timestamp', 'Failure_Mode'])

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Combined vibration magnitude
df_clean['Vibration_Magnitude'] = np.sqrt(
    df_clean['Vibration_X_mm_s']**2 +
    df_clean['Vibration_Y_mm_s']**2 +
    df_clean['Vibration_Z_mm_s']**2
)

# Temperature difference
df_clean['Temp_Difference'] = (
    df_clean['Spindle_Temp_C']
    - df_clean['Coolant_Temp_C']
)

# Efficiency ratio
df_clean['Efficiency_Ratio'] = (
    df_clean['Power_Consumption_kW']
    / df_clean['Spindle_Speed_RPM']
)

# Rolling averages
df_clean['Spindle_Temp_Rolling'] = (
    df_clean['Spindle_Temp_C']
    .rolling(window=10)
    .mean()
)

df_clean['Vibration_Rolling'] = (
    df_clean['Vibration_Magnitude']
    .rolling(window=10)
    .mean()
)

# Temperature rate of change
df_clean['Temp_Rate_Change'] = (
    df_clean['Spindle_Temp_C']
    .diff()
)

# Remove NaN rows created by rolling() and diff()
df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

# Separate features and target
X = df_clean.drop(columns=['Defect_Label'])
y = df_clean['Defect_Label']

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nFeatures:")
print(list(X.columns))

print(f"\nFeatures: {list(X.columns)}")
print("\nTarget: Defect_Label (0=Normal, 1=Defective)")

# Check class distribution
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Defect rate: {y.mean()*100:.1f}%")

# Normalise features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CNC Milling Machine - EDA', fontsize=16, fontweight='bold')

# 3a. Class imbalance
ax = axes[0, 0]
counts = y.value_counts()
bars = ax.bar(['Normal (0)', 'Defective (1)'], counts.values,
               color=['#2196F3', '#F44336'], edgecolor='white', linewidth=1.5)
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(count), ha='center', fontweight='bold')

# 3b. Correlation heatmap
ax = axes[0, 1]
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='coolwarm', center=0,
            fmt='.1f', annot=False, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', labelsize=7)

# 3c. Vibration distribution — normal vs defective
ax = axes[1, 0]
df_plot = df.copy()
ax.hist(df_plot[df_plot['Defect_Label']==0]['Vibration_X_mm_s'],
        bins=50, alpha=0.6, label='Normal', color='#2196F3')
ax.hist(df_plot[df_plot['Defect_Label']==1]['Vibration_X_mm_s'],
        bins=50, alpha=0.6, label='Defective', color='#F44336')
ax.set_title('Vibration X — Normal vs Defective', fontweight='bold')
ax.set_xlabel('Vibration X (mm/s)')
ax.set_ylabel('Frequency')
ax.legend()

# 3d. Spindle temp distribution
ax = axes[1, 1]
ax.hist(df_plot[df_plot['Defect_Label']==0]['Spindle_Temp_C'],
        bins=50, alpha=0.6, label='Normal', color='#2196F3')
ax.hist(df_plot[df_plot['Defect_Label']==1]['Spindle_Temp_C'],
        bins=50, alpha=0.6, label='Defective', color='#F44336')
ax.set_title('Spindle Temp — Normal vs Defective', fontweight='bold')
ax.set_xlabel('Spindle Temperature (°C)')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plots saved → eda_plots.png")


# ─────────────────────────────────────────────
# 4. TRAIN BASELINE MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training Baseline Models")
print("=" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"  Accuracy: {results[name]['report']['accuracy']*100:.2f}%")
    print(f"  Precision (defect): {results[name]['report']['1']['precision']*100:.2f}%")
    print(f"  Recall (defect):    {results[name]['report']['1']['recall']*100:.2f}%")
    print(f"  F1 Score (defect):  {results[name]['report']['1']['f1-score']*100:.2f}%")

# ============================================================
# LOGISTIC REGRESSION ANALYSIS
# ============================================================

log_model = results['Logistic Regression']['model']

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
})

coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])

coefficients = coefficients.sort_values(
    by='Abs_Coefficient',
    ascending=False
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

print(coefficients[['Feature', 'Coefficient']])

log_predictions = log_model.predict(X_test)

cm = confusion_matrix(y_test, log_predictions)

print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Normal','Predicted Defective'],
    yticklabels=['Actual Normal','Actual Defective']
)

plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()

plt.show()


# ─────────────────────────────────────────────
# 5. MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Evaluation — Confusion Matrices & ROC Curves',
             fontsize=14, fontweight='bold')

colors = ['#1976D2', '#388E3C', '#F57C00']
for idx, (name, result) in enumerate(results.items()):
    # Confusion matrix
    ax = axes[0, idx]
    cm = confusion_matrix(y_test, result['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Defective'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')

    # ROC curve
    ax = axes[1, idx]
    model = result['model']
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"{name} AUC = {roc_auc:.4f}")
    ax.plot(fpr, tpr, color=colors[idx], lw=2,
            label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name}\nROC Curve', fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Evaluation plots saved → model_evaluation.png")

# Feature importance (Random Forest)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='#1976D2', edgecolor='white')
plt.title('Random Forest — Feature Importances', fontweight='bold', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance plot saved → feature_importance.png")


# ─────────────────────────────────────────────
# 6. THRESHOLD-BASED ALERTING LOGIC
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Threshold-Based Alerting System")
print("=" * 60)

THRESHOLDS = {
    'Vibration_X_mm_s':       3.0,
    'Vibration_Y_mm_s':       3.0,
    'Vibration_Z_mm_s':       2.5,
    'Spindle_Temp_C':         85.0,
    'Coolant_Temp_C':         55.0,
    'Coolant_Flow_L_per_min': 2.0,   # below this = alert
    'Tool_Wear_Percent':      80.0,
    'Acoustic_Emission_dB':   90.0,
    'Power_Consumption_kW':   7.0,
}

def check_alerts(row):
    alerts = []
    for sensor, threshold in THRESHOLDS.items():
        val = row[sensor]
        if sensor == 'Coolant_Flow_L_per_min':
            if val < threshold:
                alerts.append(f"⚠️  LOW {sensor}: {val:.2f} (threshold: >{threshold})")
        else:
            if val > threshold:
                alerts.append(f" HIGH {sensor}: {val:.2f} (threshold: <{threshold})")
    return alerts

# Also use the best ML model (Random Forest) for prediction
best_model = results['Random Forest']['model']

# -----------------------------
# Feature Engineering
# -----------------------------

def predict_and_alert(sensor_reading: dict):

    print("\n--- Sensor Reading ---")
    for k, v in sensor_reading.items():
        print(f"  {k}: {v}")

    # Threshold alerts
    row = pd.Series(sensor_reading)
    alerts = check_alerts(row)

    # ML prediction
    input_df = pd.DataFrame([sensor_reading])

    input_df['Vibration_Magnitude'] = np.sqrt(
        input_df['Vibration_X_mm_s']**2 +
        input_df['Vibration_Y_mm_s']**2 +
        input_df['Vibration_Z_mm_s']**2
    )

    input_df['Temp_Difference'] = (
        input_df['Spindle_Temp_C']
        - input_df['Coolant_Temp_C']
    )

    input_df['Efficiency_Ratio'] = (
        input_df['Power_Consumption_kW']
        / input_df['Spindle_Speed_RPM']
    )

    input_df['Spindle_Temp_Rolling'] = input_df['Spindle_Temp_C']
    input_df['Vibration_Rolling'] = input_df['Vibration_Magnitude']
    input_df['Temp_Rate_Change'] = 0

    input_df = input_df[X.columns]

    input_scaled = scaler.transform(input_df)

    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1]

    print(f"\n🤖 ML Prediction: {'⛔ DEFECT DETECTED' if prediction == 1 else '✅ NORMAL'}")
    print(f"   Defect Probability: {probability*100:.1f}%")

    if alerts:
        print("\n📋 Threshold Alerts:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("\n📋 No threshold alerts triggered.")


# Test with a normal reading
normal_reading = {
    'Spindle_Speed_RPM': 3010.0,
    'Feed_Rate_mm_per_min': 498.0,
    'Cutting_Depth_mm': 2.5,
    'Vibration_X_mm_s': 0.52,
    'Vibration_Y_mm_s': 0.48,
    'Vibration_Z_mm_s': 0.31,
    'Spindle_Temp_C': 44.0,
    'Coolant_Temp_C': 25.5,
    'Coolant_Flow_L_per_min': 8.1,
    'Power_Consumption_kW': 3.6,
    'Tool_Wear_Percent': 22.0,
    'Acoustic_Emission_dB': 61.0,
    'Servo_Load_X_Percent': 40.0,
    'Servo_Load_Y_Percent': 38.5,
    'Servo_Load_Z_Percent': 35.0,
    'Surface_Roughness_Ra_um': 1.62,
}

# Test with a defective reading (thermal overload scenario)
defective_reading = {
    'Spindle_Speed_RPM': 3000.0,
    'Feed_Rate_mm_per_min': 500.0,
    'Cutting_Depth_mm': 2.5,
    'Vibration_X_mm_s': 0.5,
    'Vibration_Y_mm_s': 0.5,
    'Vibration_Z_mm_s': 0.3,
    'Spindle_Temp_C': 102.0,
    'Coolant_Temp_C': 62.0,
    'Coolant_Flow_L_per_min': 1.2,
    'Power_Consumption_kW': 8.5,
    'Tool_Wear_Percent': 88.0,
    'Acoustic_Emission_dB': 95.0,
    'Servo_Load_X_Percent': 72.0,
    'Servo_Load_Y_Percent': 68.0,
    'Servo_Load_Z_Percent': 35.0,
    'Surface_Roughness_Ra_um': 1.6,
}

print("\n TEST 1 — Normal Machine Reading:")
predict_and_alert(normal_reading)

print("\n TEST 2 — Defective Machine Reading (Thermal Overload):")
predict_and_alert(defective_reading)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("Output files: eda_plots.png | model_evaluation.png | feature_importance.png")
print("=" * 60)
# ─────────────────────────────────────────────
# 7. LSTM MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: LSTM MODEL")
print("=" * 60)

sequence_length = 20

X_seq = []
y_seq = []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(
        X_scaled.iloc[i:i+sequence_length].values
    )

    y_seq.append(
        y.iloc[i+sequence_length]
    )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence Shape:", X_seq.shape)
split = int(len(X_seq) * 0.8)

X_train_lstm = X_seq[:split]
X_test_lstm = X_seq[split:]

y_train_lstm = y_seq[:split]
y_test_lstm = y_seq[split:]

print("LSTM Training Samples:", len(X_train_lstm))
print("LSTM Testing Samples:", len(X_test_lstm))    
lstm_model = Sequential([

    LSTM(
        64,
        return_sequences=False,
        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    ),

    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')

])
lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_lstm)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_lstm
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print(class_weights)

history = lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=15,
    batch_size=32,
    validation_data=(X_test_lstm, y_test_lstm),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

loss, accuracy = lstm_model.evaluate(
    X_test_lstm,
    y_test_lstm,
    verbose=0
)

print("\nLSTM RESULTS")
print("-" * 40)
print(f"Accuracy: {accuracy*100:.2f}%")

from sklearn.metrics import roc_auc_score

y_prob_lstm = lstm_model.predict(
    X_test_lstm,
    verbose=0
)

auc_lstm = roc_auc_score(
    y_test_lstm,
    y_prob_lstm
)

print(f"LSTM AUC: {auc_lstm:.4f}")
