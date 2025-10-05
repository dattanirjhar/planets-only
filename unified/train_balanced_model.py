import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# --- Configuration ---
UNIFIED_DATA_FILE = 'datasets/unified_dataset.csv'

def train_with_smote():
    """
    Trains a Gradient Boosting model on the unified dataset after balancing
    the training data with SMOTE, then validates the performance.
    """
    try:
        # --- Part 1: Load and Prepare Data ---
        print(f"--- Loading unified data from '{UNIFIED_DATA_FILE}' ---")
        df = pd.read_csv(UNIFIED_DATA_FILE)

        X = df.drop(columns=['koi_disposition', 'source'])
        X = X.select_dtypes(include=['number'])
        y_raw = df['koi_disposition']
        source_labels = df['source']
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = label_encoder.classes_

        X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
            X, y, source_labels, test_size=0.3, random_state=42, stratify=y
        )
        print("✅ Data split into training and testing sets.")
        
        # --- Part 2: Apply SMOTE to the Training Data ---
        print("\n--- Applying SMOTE to balance the training data... ---")
        
        # SMOTE should only be applied to the training data, never the test data.
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("✅ SMOTE applied. Training data is now balanced.")

        # --- Part 3: Train the Optimized Model on Balanced Data ---
        print("\n--- Training optimized model on balanced data... ---")
        
        # We use the best parameters we found from our GridSearchCV
        best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
        
        # We need to install the imbalanced-learn library first:
        # pip install -U imbalanced-learn
        
        model = GradientBoostingClassifier(random_state=42, **best_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        print("✅ Final model trained successfully on balanced data.")

        # --- Part 4: Segmented Validation ---
        print("\n--- Performing Segmented Validation on Final Model ---")
        y_pred = model.predict(X_test)
        results_df = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred, 'source': source_test})

        # Evaluate on Kepler Data
        print("\n--- Validation on KEPLER data (SMOTE Trained) ---")
        kepler_results = results_df[results_df['source'] == 'Kepler']
        kepler_accuracy = accuracy_score(kepler_results['true_label'], kepler_results['predicted_label'])
        print(f"Accuracy on Kepler test data: {kepler_accuracy:.4f}")
        print(classification_report(kepler_results['true_label'], kepler_results['predicted_label'], target_names=class_names, zero_division=0))

        # Evaluate on TESS Data
        print("\n--- Validation on TESS data (SMOTE Trained) ---")
        tess_results = results_df[results_df['source'] == 'TESS']
        tess_accuracy = accuracy_score(tess_results['true_label'], tess_results['predicted_label'])
        print(f"Accuracy on TESS test data: {tess_accuracy:.4f}")
        print(classification_report(tess_results['true_label'], tess_results['predicted_label'], target_names=class_names, zero_division=0))
        
        # Overall Performance
        print("\n--- Overall Final Model Performance ---")
        overall_accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall accuracy on combined test data: {overall_accuracy:.4f}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found. Please check paths. Details: {e}")
    except ImportError:
        print("\n❌ ERROR: The 'imblearn' library is not installed.")
        print("Please run: pip install -U imbalanced-learn")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_with_smote()
