import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
KEPLER_FILE = "datasets/KOI_dataset_1.csv"
TESS_FILE = "datasets/TOI_translated_dataset.csv"
OUTPUT_DIR = "charts"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- Data Cleaning Function for Kepler Data ---
def clean_kepler_data(df):
    """Cleans the raw Kepler KOI dataset."""
    cols_to_drop = [
        "kepler_name",
        "koi_vet_stat",
        "koi_vet_date",
        "koi_pdisposition",
        "koi_disp_prov",
        "koi_comment",
        "koi_time0_err1",
        "koi_time0_err2",
    ]
    missing_frac = df.isnull().mean()
    cols_to_drop.extend(missing_frac[missing_frac > 0.25].index)
    unique_cols_to_drop = list(set(cols_to_drop))
    df_cleaned = df.drop(columns=unique_cols_to_drop)
    df_cleaned.dropna(inplace=True)
    return df_cleaned


# --- Main Script ---
def test_model_on_new_data():
    """
    Trains the original RF model on Kepler data and tests its performance
    on the translated TESS (TOI) data.
    """
    try:
        # --- Part 1: Train the Original Random Forest Model ---
        print("--- Training the original Random Forest model on Kepler data ---")
        koi_df_raw = pd.read_csv(KEPLER_FILE, comment="#")
        koi_df = clean_kepler_data(koi_df_raw.copy())

        # Prepare Kepler data
        X_koi = koi_df.drop(columns=["koi_disposition", "rowid", "koi_score"])
        X_koi = X_koi.select_dtypes(include=["number"])
        y_koi_raw = koi_df["koi_disposition"]

        label_encoder = LabelEncoder()
        y_koi = label_encoder.fit_transform(y_koi_raw)

        # Train the model on the FULL Kepler dataset
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_koi, y_koi)
        print("✅ Original Random Forest model trained successfully.")

        # Store the column order the model was trained on
        training_columns = X_koi.columns.tolist()

        # --- Part 2: Load and Prepare the TESS Data ---
        print(
            f"\n--- Loading and preparing translated TESS data from '{TESS_FILE}' ---"
        )
        toi_df = pd.read_csv(TESS_FILE)

        X_toi = toi_df.drop(columns=["koi_disposition"])
        y_toi_raw = toi_df["koi_disposition"]
        y_toi = label_encoder.transform(y_toi_raw)  # Use the same encoder

        # --- Part 3: Align TESS columns to match Kepler training data ---
        print("--- Aligning TESS data columns to match the trained model ---")

        # Add missing columns (like fpflags) and fill with 0
        for col in training_columns:
            if col not in X_toi.columns:
                X_toi[col] = 0

        # Ensure the column order is exactly the same
        X_toi = X_toi[training_columns]
        print("✅ TESS data columns aligned.")

        # --- Part 4: Predict on TESS data and Evaluate ---
        print("\n--- Making predictions on TESS data ---")
        y_pred = model.predict(X_toi)

        accuracy = accuracy_score(y_toi, y_pred)
        class_names = label_encoder.classes_

        print(f"\n✅ Prediction complete. Final Accuracy on TESS data: {accuracy:.4f}")
        print("\nClassification Report (TESS Data):\n")
        print(
            classification_report(
                y_toi, y_pred, target_names=class_names, zero_division=0
            )
        )

        # Visualize the results
        cm = confusion_matrix(y_toi, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("RF on TESS Data: Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        cm_path = os.path.join(OUTPUT_DIR, "rf_on_tess_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"\n✅ Confusion matrix saved to '{cm_path}'")

    except FileNotFoundError as e:
        print(
            f"❌ ERROR: A required file was not found. Please check paths. Details: {e}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    test_model_on_new_data()
