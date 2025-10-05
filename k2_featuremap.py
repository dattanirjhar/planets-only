# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn - Model Selection
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Scikit-learn - Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn - Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)

# XGBoost (install if needed: pip install xgboost)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("✓ All libraries imported successfully")
print(f"✓ XGBoost available: {XGBOOST_AVAILABLE}")

# File paths - UPDATE THESE TO YOUR PATHS
# NOTE: Replace the file paths below with your actual paths
K2_FILE = "datasets/K2_Dataset.csv"
PS_FILE = "datasets/Kep_Dataset.csv"

# Base feature set (physics-based parameters)
BASE_FEATURES = [
    'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',      # Orbital period
    'pl_rade', 'pl_radeerr1', 'pl_radeerr2',            # Planet radius
    'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2',   # Transit depth
    'pl_trandur', 'pl_trandurerr1', 'pl_trandurerr2',   # Transit duration
    'st_teff', 'st_tefferr1', 'st_tefferr2',            # Stellar temperature
    'st_rad', 'st_raderr1', 'st_raderr2',               # Stellar radius
    'st_mass', 'st_masserr1', 'st_masserr2',            # Stellar mass
]

# Target columns
K2_TARGET = 'disposition'
PS_TARGET = 'default_flag'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print(f"✓ Configuration loaded")
print(f"  - Base features: {len(BASE_FEATURES)}")
print(f"  - Test size: {TEST_SIZE}")
print(f"  - CV folds: {CV_FOLDS}")

def load_data_with_fallback(filepath, skiprows=None):
    """Load CSV with automatic fallback"""
    try:
        if skiprows:
            df = pd.read_csv(filepath, skiprows=skiprows)
            print(f"✓ Loaded {filepath.split('/')[-1]} (skipped {skiprows} rows)")
        else:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {filepath.split('/')[-1]}")
        return df
    except Exception as e:
        print(f"⚠ Error with skiprows={skiprows}, trying without skipping...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {filepath.split('/')[-1]} (no rows skipped)")
        return df

# Load datasets
df_k2_raw = load_data_with_fallback(K2_FILE, skiprows=298)
df_ps_raw = load_data_with_fallback(PS_FILE, skiprows=96)

print(f"\nDataset shapes:")
print(f"  K2: {df_k2_raw.shape}")
print(f"  PS: {df_ps_raw.shape}")

def preprocess_dataset(df, features, target_col, target_mapping, dataset_name):
    """Enhanced preprocessing with validation"""
    print(f"\nProcessing {dataset_name}...")
    
    # Check target column
    if target_col not in df.columns:
        print(f"  ⚠ Warning: Target column '{target_col}' not found")
        return None
    
    # Show original target distribution
    print(f"  Original target values: {df[target_col].unique()}")
    
    # Select available features
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    # Create dataframe
    df_clean = df[available_features + [target_col]].copy()
    
    # Add missing features
    for col in missing_features:
        df_clean[col] = np.nan
    
    # Map target - handle unmapped values
    df_clean['target'] = df_clean[target_col].map(target_mapping)
    
    # Check for NaN in target after mapping
    unmapped_values = df_clean[df_clean['target'].isna()][target_col].unique()
    if len(unmapped_values) > 0:
        print(f"  ⚠ Unmapped target values found: {unmapped_values}")
        print(f"  → Dropping {df_clean['target'].isna().sum()} rows with unmapped targets")
        df_clean = df_clean.dropna(subset=['target'])
    
    df_clean = df_clean.drop(target_col, axis=1)
    df_clean = df_clean[features + ['target']]
    
    print(f"  ✓ Final shape: {df_clean.shape}")
    print(f"  ✓ Target distribution:\n{df_clean['target'].value_counts()}")
    
    return df_clean

# Process K2 - handle multiple disposition types
k2_mapping = {
    'CONFIRMED': 0,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 1,
    'NOT DISPOSITIONED': 1
}

df_k2_clean = preprocess_dataset(
    df_k2_raw, BASE_FEATURES, K2_TARGET, k2_mapping, "K2"
)

# Process PS
ps_mapping = {1: 0, 0: 1}
df_ps_clean = preprocess_dataset(
    df_ps_raw, BASE_FEATURES, PS_TARGET, ps_mapping, "PS"
)

# Combine datasets
df_unified = pd.concat([df_k2_clean, df_ps_clean], ignore_index=True)

print("=" * 60)
print("UNIFIED DATASET")
print("=" * 60)
print(f"Total samples: {len(df_unified)}")
print(f"Shape: {df_unified.shape}")

# Remove any remaining NaN in target
if df_unified['target'].isna().sum() > 0:
    print(f"\n⚠ Removing {df_unified['target'].isna().sum()} rows with NaN targets")
    df_unified = df_unified.dropna(subset=['target'])

print(f"\nFinal shape: {df_unified.shape}")
print(f"\nClass distribution:")
class_counts = df_unified['target'].value_counts()
print(class_counts)
print(f"\nClass proportions:")
print(df_unified['target'].value_counts(normalize=True))

# Calculate imbalance ratio
imbalance_ratio = class_counts[1] / class_counts[0]
print(f"\nImbalance ratio: 1:{imbalance_ratio:.2f}")

print("=" * 60)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# Separate features and target
X = df_unified.drop('target', axis=1).copy()
y = df_unified['target'].copy()

print(f"Original features: {X.shape[1]}")

# Replace zeros with NaN for proper SNR calculation
X_processed = X.replace(0, np.nan)

# 1. Signal-to-Noise Ratios (Critical for exoplanet detection)
print("\nCreating SNR features...")
X_processed['transit_depth_snr'] = X_processed['pl_trandep'] / X_processed['pl_trandeperr1'].abs()
X_processed['transit_duration_snr'] = X_processed['pl_trandur'] / X_processed['pl_trandurerr1'].abs()
X_processed['period_snr'] = X_processed['pl_orbper'] / X_processed['pl_orbpererr1'].abs()
X_processed['radius_snr'] = X_processed['pl_rade'] / X_processed['pl_radeerr1'].abs()

# 2. Planet-to-Star Ratios (Key physical relationships)
print("Creating planet-star ratio features...")
X_processed['planet_star_radius_ratio'] = X_processed['pl_rade'] / X_processed['st_rad']
X_processed['planet_star_mass_proxy'] = X_processed['pl_rade']**3 / (X_processed['st_mass'] * X_processed['st_rad']**2)

# 3. Transit Geometry Features
print("Creating transit geometry features...")
X_processed['transit_depth_to_duration_ratio'] = X_processed['pl_trandep'] / X_processed['pl_trandur']
X_processed['impact_parameter_proxy'] = (X_processed['pl_trandur'] * X_processed['st_rad']) / X_processed['pl_orbper']

# 4. Stellar Properties
print("Creating stellar features...")
X_processed['stellar_density_proxy'] = X_processed['st_mass'] / (X_processed['st_rad']**3)

# 5. Uncertainty Metrics (High uncertainty = less reliable)
print("Creating uncertainty metrics...")
X_processed['total_uncertainty'] = (
    X_processed['pl_orbpererr1'].abs() + 
    X_processed['pl_radeerr1'].abs() + 
    X_processed['pl_trandeperr1'].abs() + 
    X_processed['pl_trandurerr1'].abs()
)

# 6. Physical Plausibility Checks
print("Creating plausibility features...")
# Very large planets around small stars are suspicious
X_processed['size_plausibility'] = X_processed['planet_star_radius_ratio'].apply(
    lambda x: 1 if x < 0.2 else 0 if pd.isna(x) else 0.5
)

print(f"\n✓ Feature engineering complete!")
print(f"  Total features: {X_processed.shape[1]}")
print(f"  New features added: {X_processed.shape[1] - len(BASE_FEATURES)}")

# Handle infinite values from division
X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

print("=" * 60)
print("IMPUTATION & SCALING")
print("=" * 60)

# Strategy: Use median for robustness against outliers
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_processed),
    columns=X_processed.columns
)

print(f"✓ Imputation complete")
print(f"  Remaining NaN values: {X_imputed.isnull().sum().sum()}")

# Standardize features (important for Logistic Regression and some other models)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns
)

print(f"✓ Feature scaling complete")
print(f"  Mean of scaled features: {X_scaled.mean().mean():.6f}")
print(f"  Std of scaled features: {X_scaled.std().mean():.6f}")

# Final dataset
X_final = X_scaled
y_final = y

print(f"\n✓ Final dataset ready:")
print(f"  Features: {X_final.shape}")
print(f"  Target: {y_final.shape}")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, 
    y_final, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y_final
)

print("=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Training set: {X_train.shape[0]} samples ({(1-TEST_SIZE)*100:.0f}%)")
print(f"Test set: {X_test.shape[0]} samples ({TEST_SIZE*100:.0f}%)")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

print("=" * 60)
print("BUILDING ENSEMBLE MODEL")
print("=" * 60)

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = (y_train == 1).sum() / (y_train == 0).sum()

# Model 1: Random Forest (Your current model, but optimized)
print("\n1. Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Model 2: Gradient Boosting
print("2. Gradient Boosting Classifier")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

# Model 3: Logistic Regression (Linear baseline)
print("3. Logistic Regression")
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=RANDOM_STATE
)

# Model 4: XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("4. XGBoost Classifier")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    
    # Create ensemble with all 4 models
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model),
            ('xgb', xgb_model)
        ],
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    print("\n✓ Ensemble created with 4 models (RF + GB + LR + XGB)")
else:
    # Create ensemble with 3 models
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    print("\n✓ Ensemble created with 3 models (RF + GB + LR)")

print("\n🚀 Training ensemble model...")
ensemble.fit(X_train, y_train)
print("✓ Training complete!")

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predictions
y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0)

print(f"\n🎯 OVERALL METRICS:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

print(f"\n🌍 CONFIRMED PLANETS (Class 0) - What We Care About Most:")
print(f"  Precision: {precision_0:.4f}")
print(f"  Recall: {recall_0:.4f}")
print(f"  F1-Score: {f1_0:.4f}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test, 
    y_pred, 
    target_names=['Confirmed (0)', 'Not Confirmed (1)'],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                 Confirmed  Not Confirmed")
print(f"Actual Confirmed     {cm[0,0]:5d}      {cm[0,1]:5d}")
print(f"Actual Not Conf      {cm[1,0]:5d}      {cm[1,1]:5d}")

# Error analysis
print(f"\n⚠️ ERROR BREAKDOWN:")
print(f"  False Negatives (Missed planets): {cm[0,1]}")
print(f"  False Positives (False alarms): {cm[1,0]}")
print(f"  Total errors: {cm[0,1] + cm[1,0]}")