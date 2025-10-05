import pandas as pd
import os

# --- Configuration ---
KOI_FILE = 'datasets/KOI_dataset_1.csv'
TOI_FILE = 'datasets/TOI_dataset.csv'
OUTPUT_DIR = 'datasets'
OUTPUT_FILE = 'unified_dataset.csv'

# This map translates TESS column names to the Kepler names our workflow uses.
FEATURE_MAP = {
    'pl_orbper': 'koi_period', 'pl_orbpererr1': 'koi_period_err1', 'pl_orbpererr2': 'koi_period_err2',
    'pl_rade': 'koi_prad', 'pl_radeerr1': 'koi_prad_err1', 'pl_radeerr2': 'koi_prad_err2',
    'pl_trandep': 'koi_depth', 'pl_trandeperr1': 'koi_depth_err1', 'pl_trandeperr2': 'koi_depth_err2',
    'st_teff': 'koi_steff', 'st_tefferr1': 'koi_steff_err1', 'st_tefferr2': 'koi_steff_err2',
    'st_rad': 'koi_srad', 'st_raderr1': 'koi_srad_err1', 'st_raderr2': 'koi_srad_err2',
    'pl_insol': 'koi_insol', 'pl_insolerr1': 'koi_insol_err1', 'pl_insolerr2': 'koi_insol_err2',
    'pl_eqt': 'koi_teq', 'pl_eqterr1': 'koi_teq_err1', 'pl_eqterr2': 'koi_teq_err2', 'pl_snr': 'koi_model_snr'
}


def process_kepler_data(filepath):
    """Loads and processes the Kepler (KOI) data."""
    print(f"--- Processing Kepler data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')

    cols_to_drop = [
        'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'kepler_name', 'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition',
        'koi_disp_prov', 'koi_comment', 'koi_time0_err1', 'koi_time0_err2', 'rowid'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Use the robust, two-step cleaning method
    essential_features = ['koi_period', 'koi_prad', 'koi_depth', 'koi_disposition']
    df.dropna(subset=essential_features, inplace=True)
    df.fillna(0, inplace=True) # Fill remaining less-critical NaNs
    
    df['source'] = 'Kepler'
    print(f"✅ Kepler data processed. Shape: {df.shape}")
    return df


def process_tess_data(filepath):
    """Loads, translates, and processes the TESS (TOI) data."""
    print(f"--- Processing TESS data from '{filepath}' ---")
    df = pd.read_csv(filepath, comment='#')

    disposition_map = {"FP": "FALSE POSITIVE", "CP": "CONFIRMED", "KP": "CONFIRMED", "PC": "CANDIDATE"}
    df['koi_disposition'] = df['tfopwg_disp'].map(disposition_map)
    df.dropna(subset=['koi_disposition'], inplace=True)

    mappable_cols = [col for col in FEATURE_MAP.keys() if col in df.columns]
    df_selected = df[mappable_cols + ['koi_disposition']]
    df_translated = df_selected.rename(columns=FEATURE_MAP)

    for col in df_translated.columns:
        if col != 'koi_disposition':
            df_translated[col] = pd.to_numeric(df_translated[col], errors='coerce')

    # Use the robust, two-step cleaning method
    essential_features = ['koi_period', 'koi_prad', 'koi_depth']
    df_translated.dropna(subset=essential_features, inplace=True)
    df_translated.fillna(0, inplace=True) # Fill remaining less-critical NaNs

    df_translated['source'] = 'TESS'
    print(f"✅ TESS data processed. Shape: {df_translated.shape}")
    return df_translated


def main():
    """Main function to run the data unification process."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        koi_df = process_kepler_data(KOI_FILE)
        toi_df = process_tess_data(TOI_FILE)

        print("\n--- Finding common features between datasets ---")
        koi_cols = set(koi_df.columns)
        toi_cols = set(toi_df.columns)
        common_cols = list(koi_cols.intersection(toi_cols))
        print(f"Found {len(common_cols)} common features.")

        koi_final = koi_df[common_cols]
        toi_final = toi_df[common_cols]

        unified_df = pd.concat([koi_final, toi_final], ignore_index=True)
        print(f"\n✅ Datasets successfully combined. Final shape: {unified_df.shape}")
        
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        unified_df.to_csv(output_path, index=False)
        print(f"✅ Unified dataset saved to '{output_path}'")
        
        print("\nDistribution of sources in the unified dataset:")
        print(unified_df['source'].value_counts())
        
        print("\nUnified data preview:")
        print(unified_df.head())

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
