import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---
# Paths are updated to match your project structure
file_path = "datasets/TOI_dataset.csv"
output_dir = "charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")


def clean_and_prep_toi(df):
    """Cleans and prepares the raw TOI dataframe for analysis and plotting."""
    # Map the TESS disposition labels to our standard Kepler labels
    disposition_map = {
        "FP": "FALSE POSITIVE",
        "CP": "CONFIRMED",
        "KP": "CONFIRMED",  # Known Planets are also Confirmed
        "PC": "CANDIDATE",
    }
    # Use .get() to avoid errors if an unexpected label appears
    df["koi_disposition"] = df["tfopwg_disp"].apply(lambda x: disposition_map.get(x))

    # Drop rows where disposition is unknown
    df.dropna(subset=["koi_disposition"], inplace=True)

    # Convert key columns to numeric, coercing errors to NaN
    numeric_cols = ["pl_rade", "pl_orbper"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows if essential physical parameters are missing
    df.dropna(subset=["pl_rade", "pl_orbper"], inplace=True)

    return df


try:
    # Load the TESS dataset
    toi_df_raw = pd.read_csv(file_path, comment="#")
    print(f"Successfully loaded '{file_path}'.")

    # --- Step 1: Clean and Prepare TESS Data ---
    print("\n--- Cleaning and Preparing TESS Data ---")
    df_cleaned = clean_and_prep_toi(toi_df_raw.copy())
    print(f"Data cleaned. Usable rows: {len(df_cleaned)}")

    # --- Visualizations ---
    print("\n--- Generating and Saving TESS Disposition Plots ---")

    # 1. Countplot of Disposition
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x="koi_disposition",
        data=df_cleaned,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
        palette="viridis",
    )
    plt.title("Count of TESS Object Dispositions")
    plt.xlabel("Disposition")
    plt.ylabel("Count")

    countplot_path = os.path.join(output_dir, "tess_disposition_countplot.png")
    plt.savefig(countplot_path)
    plt.close()
    print(f"TESS countplot saved to '{countplot_path}'")

    # 2. Boxplot of Planetary Radius
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        x="koi_disposition",
        y="pl_rade", # Use the TESS column name for planetary radius
        data=df_cleaned,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
        palette="viridis",
    )
    plt.title("TESS Planetary Radius by Disposition")
    plt.xlabel("Disposition")
    plt.ylabel("Planetary Radius (Earth Radii)")
    plt.ylim(0, 50)  # Zoom in on the majority of data

    boxplot_path = os.path.join(output_dir, "tess_radius_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"TESS boxplot saved to '{boxplot_path}'")


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

