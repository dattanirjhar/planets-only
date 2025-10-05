import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

# --- Configuration ---
KOI_FILE = 'datasets/KOI_dataset_1.csv'
TOI_FILE = 'datasets/TOI_dataset.csv'
K2_FILE = 'datasets/K2_Dataset.csv'
OUTPUT_DIR = 'datasets'
OUTPUT_FILE = 'mega_unified_dataset.csv'

UNIFIED_FEATURES_MAP = {
    # This map now only contains feature columns, not disposition or source
    # Kepler -> Universal
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
    # TESS & K2 already use the 'pl_' standard for most features
}

def process_kepler_data(filepath):
    """Loads and processes the Kepler (KOI) data, stripping flags."""
    print(f"--- Processing Kepler data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    cols_to_drop = [col for col in df.columns if 'koi_fpflag' in col or 'koi_score' in col or 'kepler_name' in col or 'rowid' in col]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Rename features and the disposition column
    df.rename(columns=UNIFIED_FEATURES_MAP, inplace=True)
    df.rename(columns={'koi_disposition': 'disposition'}, inplace=True)
    df['source'] = 'Kepler'
    return df

def process_tess_data(filepath):
    """Loads, translates, and processes the TESS (TOI) data."""
    print(f"--- Processing TESS data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    disposition_map = {"FP": "FALSE POSITIVE", "CP": "CONFIRMED", "KP": "CONFIRMED", "PC": "CANDIDATE"}
    df['disposition'] = df['tfopwg_disp'].map(disposition_map)
    df['source'] = 'TESS'
    # No feature rename needed as TESS uses the target 'pl_' schema
    return df

def process_k2_data(filepath):
    """Loads and processes the K2 data."""
    print(f"--- Processing K2 data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')
    df['source'] = 'K2'
    # K2 already has 'disposition' and 'pl_' feature names
    return df

def main():
    """Main function to run the full unification and training pipeline."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        # --- Part 1: Unify Datasets ---
        df_koi = process_kepler_data(KOI_FILE)
        df_toi = process_tess_data(TOI_FILE)
        df_k2 = process_k2_data(K2_FILE)

        # --- FIX START: Robust column unification ---
        # Identify feature columns (everything not 'disposition' or 'source')
        koi_features = [col for col in df_koi.columns if col not in ['disposition', 'source']]
        toi_features = [col for col in df_toi.columns if col not in ['disposition', 'source']]
        k2_features = [col for col in df_k2.columns if col not in ['disposition', 'source']]
        
        # Find the intersection of ONLY the feature columns
        common_features = list(set(koi_features) & set(toi_features) & set(k2_features))
        
        # Define the final columns we want to keep: common features + metadata
        final_cols = common_features + ['disposition', 'source']
        print(f"\nFound {len(common_features)} common features. Total columns with metadata: {len(final_cols)}")
        # --- FIX END ---

        # Filter and combine, ensuring all DFs have the final columns
        mega_df = pd.concat([
            df_koi.reindex(columns=final_cols),
            df_toi.reindex(columns=final_cols),
            df_k2.reindex(columns=final_cols)
        ], ignore_index=True)

        # Final Cleaning
        mega_df.dropna(subset=['disposition'], inplace=True)
        feature_cols = [col for col in mega_df.columns if col not in ['disposition', 'source']]
        for col in feature_cols:
            mega_df[col] = pd.to_numeric(mega_df[col], errors='coerce')
        mega_df.fillna(0, inplace=True)

        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        mega_df.to_csv(output_path, index=False)
        print(f"\n✅ Mega unified dataset created at '{output_path}'. Shape: {mega_df.shape}")
        print("\nDistribution of sources:")
        print(mega_df['source'].value_counts())

        # --- Part 2: Train the Champion Model ---
        print("\n--- Training the Final XGBoost Champion Model ---")
        X = mega_df.drop(columns=['disposition', 'source'])
        y_raw = mega_df['disposition']
        source_labels = mega_df['source']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = list(label_encoder.classes_) # Convert to list for safety

        X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
            X, y, source_labels, test_size=0.3, random_state=42, stratify=y
        )
        
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Champion model trained.")

        # --- Part 3: The Ultimate Validation ---
        print("\n--- Performing the Ultimate 3-Way Segmented Validation ---")
        y_pred = model.predict(X_test)
        results_df = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred, 'source': source_test})
        
        # --- FIX START: Robust validation reporting ---
        # Get the full list of numeric labels that the model knows about
        all_labels = np.arange(len(class_names))
        # --- FIX END ---

        sources = ['Kepler', 'TESS', 'K2']
        for source in sources:
            print(f"\n--- Validation on {source.upper()} data ---")
            source_results = results_df[results_df['source'] == source]
            if not source_results.empty:
                accuracy = accuracy_score(source_results['true_label'], source_results['predicted_label'])
                print(f"Accuracy on {source} test data: {accuracy:.4f}")
                # Use the full list of labels and names to create a consistent report structure
                print(classification_report(
                    source_results['true_label'], 
                    source_results['predicted_label'], 
                    labels=all_labels, # Use all possible labels
                    target_names=class_names, # Use all possible names
                    zero_division=0
                ))
            else:
                print(f"No {source} data in the test set to evaluate.")

        print("\n--- Overall Final Model Performance ---")
        overall_accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall accuracy on combined test data: {overall_accuracy:.4f}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

