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
    """A handy function to reuse our cleaning logic."""
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
    print(f"Data loaded and cleaned. Shape: {df.shape}")

    # --- Feature Engineering Step 1: Binary Classification Target ---
    # Our first goal: identify the 'FALSE POSITIVE' class.
    print("\n--- Preparing Data for Modeling ---")

    # Create the binary target variable
    df["is_false_positive"] = (df["koi_disposition"] == "FALSE POSITIVE").astype(int)

    # Define our features (X) and target (y)
    # We drop the original disposition columns as they are the answer!
    X = df.drop(columns=["koi_disposition", "is_false_positive"])
    y = df["is_false_positive"]

    # Ensure all feature columns are numeric
    X = X.select_dtypes(include=["number"])

    print("Features (X) and target (y) are ready.")

    # --- Model Training: Random Forest ---
    print("\n--- Training Random Forest Classifier ---")

    # Split data into training and testing sets to evaluate performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize and train the model
    # random_state=42 ensures we get the same results every time we run this.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy on test data: {accuracy:.4f}")
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- Feature Importance Analysis ---
    print("\n--- Extracting and Visualizing Feature Importances ---")

    importances = model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 Most Important Features:\n")
    print(feature_importance_df.head(20))

    # Plotting the top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance_df.head(15),
        palette="rocket",
    )
    plt.title("Top 15 Features for Predicting False Positives", fontsize=16)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()  # Adjust layout to make sure everything fits

    plot_path = os.path.join(output_dir, "false_positive_feature_importance.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nFeature importance plot saved to '{plot_path}'")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
