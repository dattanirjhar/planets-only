import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
UNIFIED_DATA_FILE = 'datasets/unified_dataset.csv'

def tune_and_validate():
    """
    Performs hyperparameter tuning using GridSearchCV to find the best
    Gradient Boosting model, then trains and validates it.
    """
    try:
        # --- Part 1: Load and Prepare Data ---
        print(f"--- Loading unified data from '{UNIFIED_DATA_FILE}' ---")
        df = pd.read_csv(UNIFIED_DATA_FILE)
        print(f"✅ Unified data loaded. Shape: {df.shape}")

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

        # --- Part 2: Hyperparameter Tuning with GridSearchCV ---
        print("\n--- Starting Hyperparameter Tuning (This may take a few minutes)... ---")
        
        # Define the grid of parameters to search
        # This is a focused grid. A real-world search might be larger.
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4, 5]
        }

        # Initialize the Grid Search with cross-validation
        # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all available CPU cores.
        grid_search = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2, # This will print progress updates
            scoring='accuracy'
        )

        # Run the search
        grid_search.fit(X_train, y_train)

        # --- Part 3: Report Best Settings and Train Final Model ---
        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        print("\n--- Training final optimized model with best parameters... ---")
        best_model = grid_search.best_estimator_
        
        # --- Part 4: Segmented Validation of the Optimized Model ---
        print("\n--- Performing Segmented Validation on Optimized Model ---")
        y_pred = best_model.predict(X_test)
        results_df = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred, 'source': source_test})

        # Evaluate on Kepler Data
        print("\n--- Validation on KEPLER data (Optimized) ---")
        kepler_results = results_df[results_df['source'] == 'Kepler']
        kepler_accuracy = accuracy_score(kepler_results['true_label'], kepler_results['predicted_label'])
        print(f"Accuracy on Kepler test data: {kepler_accuracy:.4f}")
        print(classification_report(kepler_results['true_label'], kepler_results['predicted_label'], target_names=class_names, zero_division=0))

        # Evaluate on TESS Data
        print("\n--- Validation on TESS data (Optimized) ---")
        tess_results = results_df[results_df['source'] == 'TESS']
        tess_accuracy = accuracy_score(tess_results['true_label'], tess_results['predicted_label'])
        print(f"Accuracy on TESS test data: {tess_accuracy:.4f}")
        print(classification_report(tess_results['true_label'], tess_results['predicted_label'], target_names=class_names, zero_division=0))
        
        # Overall Performance
        print("\n--- Overall Optimized Model Performance ---")
        overall_accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall accuracy on combined test data: {overall_accuracy:.4f}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    tune_and_validate()
