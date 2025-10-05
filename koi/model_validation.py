import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# --- Setup ---
file_path = "datasets/KOI_dataset_1.csv"


def clean_data(df):
    """Reusing our cleaning logic."""
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


try:
    # --- Load and Clean Data ---
    print("--- Loading and Cleaning Data ---")
    koi_df_raw = pd.read_csv(file_path, comment="#")
    df = clean_data(koi_df_raw.copy())

    # --- Prepare Data ---
    df["is_false_positive"] = (df["koi_disposition"] == "FALSE POSITIVE").astype(int)
    X = df.drop(columns=["koi_disposition", "is_false_positive"]).select_dtypes(
        include=["number"]
    )
    y = df["is_false_positive"]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # --- Test 1: Cross-Validation ---
    print("\n--- Running 5-Fold Cross-Validation (with koi_score) ---")

    # cv=5 means 5 folds
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"Accuracy scores for each fold: {cv_scores}")
    print(
        f"Average CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
    )

    # --- Test 2: The 'koi_score' Experiment ---
    print("\n--- Running Experiment: Removing 'koi_score' ---")

    # Create a new feature set without koi_score
    X_no_score = X.drop(columns=["koi_score"])

    # We'll use a simple train-test split for this experiment
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_score, y, test_size=0.3, random_state=42, stratify=y
    )

    model_no_score = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model_no_score.fit(X_train, y_train)

    y_pred = model_no_score.predict(X_test)
    accuracy_no_score = accuracy_score(y_test, y_pred)

    print(f"Model accuracy WITHOUT 'koi_score': {accuracy_no_score:.4f}")

except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
