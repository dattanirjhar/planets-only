import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
file_path = "datasets/TOI_dataset.csv"
output_dir = "charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# This map translates TESS column names to the Kepler names our workflow uses.
# It only includes features with direct, high-confidence equivalents.
FEATURE_MAP = {
    "pl_orbper": "koi_period",
    "pl_orbpererr1": "koi_period_err1",
    "pl_orbpererr2": "koi_period_err2",
    "pl_rade": "koi_prad",
    "pl_radeerr1": "koi_prad_err1",
    "pl_radeerr2": "koi_prad_err2",
    "pl_trandep": "koi_depth",
    "pl_trandeperr1": "koi_depth_err1",
    "pl_trandeperr2": "koi_depth_err2",
    "st_teff": "koi_steff",
    "st_tefferr1": "koi_steff_err1",
    "st_tefferr2": "koi_steff_err2",
    "st_rad": "koi_srad",
    "st_raderr1": "koi_srad_err1",
    "st_raderr2": "koi_srad_err2",
    "pl_insol": "koi_insol",
    "pl_insolerr1": "koi_insol_err1",
    "pl_insolerr2": "koi_insol_err2",
    "pl_eqt": "koi_teq",
    "pl_eqterr1": "koi_teq_err1",
    "pl_eqterr2": "koi_teq_err2",
    "pl_snr": "koi_model_snr",
}


def translate_and_clean_toi(df_raw):
    """Translates and cleans the raw TOI data into a model-ready format."""
    df = df_raw.copy()

    disposition_map = {
        "FP": "FALSE POSITIVE",
        "CP": "CONFIRMED",
        "KP": "CONFIRMED",
        "PC": "CANDIDATE",
    }
    df["koi_disposition"] = df["tfopwg_disp"].apply(lambda x: disposition_map.get(x))
    df.dropna(subset=["koi_disposition"], inplace=True)

    mappable_cols = [col for col in FEATURE_MAP.keys() if col in df.columns]
    df_selected = df[mappable_cols + ["koi_disposition"]]
    df_translated = df_selected.rename(columns=FEATURE_MAP)

    for col in df_translated.columns:
        if col != "koi_disposition":
            df_translated[col] = pd.to_numeric(df_translated[col], errors="coerce")

    essential_features = ["koi_period", "koi_prad", "koi_depth"]
    essential_present = [
        feat for feat in essential_features if feat in df_translated.columns
    ]
    df_translated.dropna(subset=essential_present, inplace=True)
    df_translated.fillna(0, inplace=True)  # Fill any remaining NaNs with 0

    return df_translated


try:
    print("--- Loading and Translating TESS Data ---")
    toi_df_raw = pd.read_csv(file_path, comment="#")
    df = translate_and_clean_toi(toi_df_raw)
    print(f"✅ Data translated and cleaned. Shape: {df.shape}")

    print("\n--- Preparing Data for Modeling ---")
    df["is_false_positive"] = (df["koi_disposition"] == "FALSE POSITIVE").astype(int)

    X = df.drop(columns=["koi_disposition", "is_false_positive"])
    y = df["is_false_positive"]

    print("Features (X) and target (y) are ready.")

    print("\n--- Training Random Forest Classifier on TESS Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy on TESS test data: {accuracy:.4f}")

    print("\n--- Extracting Feature Importances from TESS Model ---")
    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 Most Important Features (TESS Data):\n")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance_df.head(15),
        palette="plasma",
    )
    plt.title("Top 15 Features for Predicting TESS False Positives", fontsize=16)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "tess_false_positive_feature_importance.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\n✅ TESS feature importance plot saved to '{plot_path}'")

except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
