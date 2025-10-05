import pandas as pd
import os

# --- Configuration ---

# This dictionary maps the TESS column names (keys) to the Kepler names our model expects (values).
# This remains the same as our goal is to match the original model's features.
FEATURE_MAP = {
    "pl_orbper": "koi_period",
    "pl_orbpererr1": "koi_period_err1",
    "pl_orbpererr2": "koi_period_err2",
    "pl_rade": "koi_prad",
    "pl_radeerr1": "koi_prad_err1",
    "pl_radeerr2": "koi_prad_err2",
    "pl_trandur": "koi_duration",
    "pl_trandurerr1": "koi_duration_err1",
    "pl_trandurerr2": "koi_duration_err2",
    "pl_trandep": "koi_depth",
    "pl_trandeperr1": "koi_depth_err1",
    "pl_trandeperr2": "koi_depth_err2",
    "pl_snr": "koi_model_snr",
    "st_teff": "koi_steff",
    "st_tefferr1": "koi_steff_err1",
    "st_tefferr2": "koi_steff_err2",
    "st_rad": "koi_srad",
    "st_raderr1": "koi_srad_err1",
    "st_raderr2": "koi_srad_err2",
    "st_mass": "koi_smass",
    "st_masserr1": "koi_smass_err1",
    "st_masserr2": "koi_smass_err2",
    "pl_insol": "koi_insol",
    "pl_insolerr1": "koi_insol_err1",
    "pl_insolerr2": "koi_insol_err2",
    "pl_eqt": "koi_teq",
    "pl_eqterr1": "koi_teq_err1",
    "pl_eqterr2": "koi_teq_err2",
}

# This maps the TESS disposition labels to the Kepler labels.
DISPOSITION_MAP = {
    "PC": "CANDIDATE",
    "KP": "CONFIRMED",
    "CP": "CONFIRMED",
    "FP": "FALSE POSITIVE",
}

# --- Main Script ---


def translate_tess_data_v2():
    """
    Loads raw TESS TOI data, translates it, cleans it with a more robust method,
    and saves the result to a new CSV file.
    """
    input_file = "datasets/TOI_dataset.csv"
    output_dir = "datasets"
    output_file = "TOI_translated_dataset.csv"  # We will overwrite the old, empty file.
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print(f"--- Loading raw TESS data from '{input_file}' ---")
        toi_df = pd.read_csv(input_file, comment="#")
        print(f"✅ Loaded {len(toi_df)} rows from the raw file.")

        # --- Step 1: Translate Dispositions ---
        print("\n--- Translating disposition labels ---")
        toi_df["koi_disposition"] = toi_df["tfopwg_disp"].map(DISPOSITION_MAP)

        # --- Step 2: Select and Rename Features ---
        print("--- Selecting and renaming features ---")
        mappable_cols = [col for col in FEATURE_MAP.keys() if col in toi_df.columns]
        df_translated = toi_df[mappable_cols + ["koi_disposition"]].copy()
        df_translated = df_translated.rename(columns=FEATURE_MAP)
        print(f"Selected and renamed {len(mappable_cols)} feature columns.")

        # --- Step 3: A SMARTER Cleaning Process ---
        print("\n--- Cleaning the translated data (Robust Method) ---")
        initial_rows = len(df_translated)

        df_translated.dropna(subset=["koi_disposition"], inplace=True)
        print(
            f"Dropped {initial_rows - len(df_translated)} rows with missing dispositions."
        )

        # FIX: Make the essential features list DYNAMIC.
        # This prevents the script from crashing if a column is missing from the source file.
        essential_features_hardcoded = [
            "koi_period",
            "koi_prad",
            "koi_duration",
            "koi_depth",
        ]
        # Find which of our essential features are ACTUALLY in the dataframe
        essential_features_present = [
            feat
            for feat in essential_features_hardcoded
            if feat in df_translated.columns
        ]

        print(
            f"Found {len(essential_features_present)} essential features to clean: {essential_features_present}"
        )

        rows_before_feature_drop = len(df_translated)

        df_translated.dropna(subset=essential_features_present, inplace=True)
        print(
            f"Dropped {rows_before_feature_drop - len(df_translated)} rows missing essential features."
        )

        final_rows = len(df_translated)
        print(f"Final dataset has {final_rows} clean, usable rows.")

        # --- Step 4: Save the Final Dataset ---
        print(f"\n--- Saving corrected translated data to '{output_path}' ---")
        df_translated.to_csv(output_path, index=False)
        print("✅ Translation complete!")
        print("\nCorrected translated data preview:")
        print(df_translated.head())

    except FileNotFoundError:
        print(
            f"❌ ERROR: The file '{input_file}' was not found. Please make sure it's in the correct directory."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    translate_tess_data_v2()
