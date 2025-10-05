import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# --- Configuration ---
KOI_FILE = 'datasets/KOI_dataset_1.csv'
TOI_FILE = 'datasets/TOI_dataset.csv'
K2_FILE = 'datasets/K2_Dataset.csv'
OUTPUT_DIR = 'output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'final_exoplanet_model.joblib')
DATA_FILE = os.path.join(OUTPUT_DIR, 'translated_data', 'mega_unified_dataset_final.csv')

UNIFIED_FEATURES_MAP = {
    'koi_period': 'pl_orbper', 'koi_period_err1': 'pl_orbpererr1', 'koi_period_err2': 'pl_orbpererr2',
    'koi_prad': 'pl_rade', 'koi_prad_err1': 'pl_radeerr1', 'koi_prad_err2': 'pl_radeerr2',
    'koi_depth': 'pl_trandep', 'koi_depth_err1': 'pl_trandeperr1', 'koi_depth_err2': 'pl_trandeperr2',
    'koi_duration': 'pl_trandur', 'koi_duration_err1': 'pl_trandurerr1', 'koi_duration_err2': 'pl_trandurerr2',
    'koi_steff': 'st_teff', 'koi_steff_err1': 'st_tefferr1', 'koi_steff_err2': 'st_tefferr2',
    'koi_srad': 'st_rad', 'koi_srad_err1': 'st_raderr1', 'koi_srad_err2': 'st_raderr2',
    'koi_smass': 'st_mass', 'koi_smass_err1': 'st_masserr1', 'koi_smass_err2': 'st_masserr2',
    'koi_insol': 'pl_insol', 'koi_insol_err1': 'pl_insolerr1', 'koi_insol_err2': 'pl_insolerr2',
    'koi_teq': 'pl_eqt',
    'koi_model_snr': 'pl_snr',
}

def process_kepler_data(filepath):
    print(f"--- Processing Kepler data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    cols_to_drop = [col for col in df.columns if 'koi_fpflag' in col or 'koi_score' in col or 'kepler_name' in col or 'rowid' in col]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.rename(columns=UNIFIED_FEATURES_MAP, inplace=True)
    df.rename(columns={'koi_disposition': 'disposition'}, inplace=True)
    df['source'] = 'Kepler'
    return df

def process_tess_data(filepath):
    print(f"--- Processing TESS data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    disposition_map = {"FP": "FALSE POSITIVE", "CP": "CONFIRMED", "KP": "CONFIRMED", "PC": "CANDIDATE"}
    df['disposition'] = df['tfopwg_disp'].map(disposition_map)
    df['source'] = 'TESS'
    return df

def process_k2_data(filepath):
    print(f"--- Processing K2 data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    df['source'] = 'K2'
    return df

def feature_engineering(df):
    print("\n--- Performing Advanced Feature Engineering ---")
    if 'pl_rade' in df.columns and 'st_rad' in df.columns:
        df['pl_rad_to_star_rad_ratio'] = np.divide(df['pl_rade'], df['st_rad'])
        print("✅ Created feature: 'pl_rad_to_star_rad_ratio'")
    else:
        print("⚠️ Skipped feature 'pl_rad_to_star_rad_ratio'.")

    if 'st_mass' in df.columns and 'st_rad' in df.columns:
        df['st_density'] = np.divide(df['st_mass'], (df['st_rad']**3))
        print("✅ Created feature: 'st_density'")
    else:
        print("⚠️ Skipped feature 'st_density'.")

    if 'pl_trandep' in df.columns and 'pl_trandeperr1' in df.columns:
        df['pl_signal_significance'] = np.divide(df['pl_trandep'], df['pl_trandeperr1'])
        print("✅ Created feature: 'pl_signal_significance'")
    else:
        print("⚠️ Skipped feature 'pl_signal_significance'.")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def main():
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'translated_data')):
        os.makedirs(os.path.join(OUTPUT_DIR, 'translated_data'))

    try:
        df_koi = process_kepler_data(KOI_FILE)
        df_toi = process_tess_data(TOI_FILE)
        df_k2 = process_k2_data(K2_FILE)

        koi_features = [col for col in df_koi.columns if col not in ['disposition', 'source']]
        toi_features = [col for col in df_toi.columns if col not in ['disposition', 'source']]
        k2_features = [col for col in df_k2.columns if col not in ['disposition', 'source']]
        
        common_features = list(set(koi_features) & set(toi_features) & set(k2_features))
        final_cols = common_features + ['disposition', 'source']
        print(f"\nFound {len(common_features)} common features. Total columns with metadata: {len(final_cols)}")

        mega_df = pd.concat([
            df_koi.reindex(columns=final_cols),
            df_toi.reindex(columns=final_cols),
            df_k2.reindex(columns=final_cols)
        ], ignore_index=True)

        mega_df.dropna(subset=['disposition'], inplace=True)
        feature_cols = [col for col in mega_df.columns if col not in ['disposition', 'source']]
        for col in feature_cols:
            mega_df[col] = pd.to_numeric(mega_df[col], errors='coerce')
        
        mega_df = feature_engineering(mega_df)
        mega_df.fillna(0, inplace=True)

        mega_df.to_csv(DATA_FILE, index=False)
        print(f"\n✅ Mega unified dataset with new features created. Shape: {mega_df.shape}")

        print("\n--- Preparing data for the final model ---")
        X = mega_df.drop(columns=['disposition', 'source'])
        y_raw = mega_df['disposition']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        best_params = {
            'colsample_bytree': 0.7979622306417505,
            'learning_rate': 0.12408879488107988,
            'max_depth': 6,
            'n_estimators': 289,
            'subsample': 0.7975990992289792
        }
        
        print("\n--- Training final optimized model on ALL feature-engineered data... ---")
        # We train on the entire dataset to create the final production model
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42, **best_params)
        model.fit(X, y)
        print("✅ Final champion model trained on all available data.")
        
        # --- NEW STEP: Save the trained model and the label encoder ---
        print(f"\n--- Saving final model to '{MODEL_FILE}' ---")
        joblib.dump(model, MODEL_FILE)
        joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
        print("✅ Model and label encoder saved successfully.")
        print("\nThis model can now be loaded and used for future predictions without retraining.")

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
