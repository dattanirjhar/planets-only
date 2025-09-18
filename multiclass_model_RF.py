import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Setup ---
file_path = "datasets/KOI_dataset_1.csv"
output_dir = "charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def clean_data(df):
    """Reusing our tried-and-true cleaning logic."""
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
    print(f"✅ Data loaded and cleaned. Shape: {df.shape}")

    # --- Step 1: Prepare Data for Multi-Class Modeling ---
    print("\n--- Preparing Data for Multi-Class Model ---")

    # Define features (X) and target (y)
    # We drop rowid and the leaky koi_score feature to make the model work harder
    X = df.drop(columns=["koi_disposition", "rowid", "koi_score"])
    y_raw = df["koi_disposition"]

    # Use LabelEncoder to turn our text labels into numbers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Store the mapping from number to label for later
    class_names = label_encoder.classes_
    print(f"Target labels encoded. Mapping: {list(enumerate(class_names))}")

    X = X.select_dtypes(include=["number"])

    # --- Step 2: Train the Multi-Class Random Forest ---
    print("\n--- Training Multi-Class Random Forest Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Step 3: Evaluate the Model ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy on test data: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Step 4: Visualize with a Confusion Matrix ---
    print("--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix for Multi-Class Model", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    cm_path = os.path.join(output_dir, "multiclass_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved to '{cm_path}'")

    # --- Step 5: Overall Feature Importance ---
    print("\n--- Extracting Overall Feature Importances ---")
    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 Overall Most Important Features:\n")
    print(feature_importance_df.head(10))

except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
