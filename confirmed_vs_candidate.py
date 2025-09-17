import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
file_path = "datasets/KOI_dataset_1.csv"
output_dir = "charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    df_cleaned = clean_data(koi_df_raw.copy())
    print(f"✅ Full dataset loaded and cleaned. Shape: {df_cleaned.shape}")

    # --- Step 1: Filter for CONFIRMED and CANDIDATE ---
    print("\n--- Filtering for CONFIRMED and CANDIDATE targets ---")

    # Keep only the rows we are interested in for this analysis
    df_subset = df_cleaned[
        df_cleaned["koi_disposition"].isin(["CONFIRMED", "CANDIDATE"])
    ].copy()

    print(f"✅ Subset created. Shape: {df_subset.shape}")
    print(
        "Class distribution in subset:\n", df_subset["koi_disposition"].value_counts()
    )

    # --- Step 2: Prepare Data for Modeling ---
    print("\n--- Preparing Subset Data for Modeling ---")

    # Define our features (X) and target (y)
    # We drop koi_score as it's a "leaky" feature and rowid as it's an identifier
    X = df_subset.drop(columns=["koi_disposition", "koi_score", "rowid"])
    y = df_subset["koi_disposition"]

    # Ensure all feature columns are numeric
    X = X.select_dtypes(include=["number"])

    print("Features (X) and target (y) are ready.")

    # --- Step 3: Model Training: Random Forest ---
    print("\n--- Training New Random Forest Classifier ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Step 4: Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy on test data: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- Step 5: Feature Importance Analysis ---
    print("\n--- Extracting and Visualizing Feature Importances ---")

    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 Most Important Features:\n")
    print(feature_importance_df.head(10))

    # Plotting the top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance_df.head(15),
        palette="plasma",
    )
    plt.title("Top 15 Features for Distinguishing CONFIRMED vs. CANDIDATE", fontsize=16)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "confirmed_vs_candidate_importance.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\n✅ Feature importance plot saved to '{plot_path}'")

except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
