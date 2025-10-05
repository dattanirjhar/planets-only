import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---
file_path = "datasets/KOI_dataset_1.csv"
# We create a directory to save our beautiful charts
output_dir = "charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")


try:
    # Load the dataset, ignoring commented lines
    koi_df = pd.read_csv(file_path, comment="#")
    print(f"Successfully loaded '{file_path}'.")

    # --- Initial Exploration ---
    print(
        f"\n--- Dataset Dimensions ---\n{koi_df.shape[0]} rows and {koi_df.shape[1]} columns.\n"
    )

    # --- Step 1: Data Cleaning & Preprocessing ---
    print("\n--- Cleaning Data ---")
    null_counts = koi_df.isnull().sum()

    # Drop columns that are entirely or mostly empty, plus some text-based ones.
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
    missing_frac = koi_df.isnull().mean()
    cols_to_drop.extend(missing_frac[missing_frac > 0.25].index)

    unique_cols_to_drop = list(set(cols_to_drop))

    df_cleaned = koi_df.drop(columns=unique_cols_to_drop)
    df_cleaned.dropna(inplace=True)

    print(f"\nDropped {len(unique_cols_to_drop)} columns.")
    print(f"Dataset shape after cleaning: {df_cleaned.shape}")
    print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")

    # --- Step 2: Correlation Analysis ---
    print("\n--- Generating and Saving Correlation Heatmap ---")
    numeric_df = df_cleaned.select_dtypes(include=["number"])
    plt.figure(figsize=(16, 12))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, cmap="viridis")
    plt.title("Correlation Matrix of KOI Numerical Features", fontsize=16)

    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()  # Close the plot to free memory
    print(f"✅ Heatmap saved to '{heatmap_path}'")

    # --- Visualizations ---
    print("\n--- Generating and Saving Disposition Plots ---")

    # 1. Countplot of Disposition
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x="koi_disposition",
        data=df_cleaned,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
    )
    plt.title("Count of Planet Dispositions")
    plt.xlabel("Disposition")
    plt.ylabel("Count")

    countplot_path = os.path.join(output_dir, "disposition_countplot.png")
    plt.savefig(countplot_path)
    plt.close()
    print(f"Countplot saved to '{countplot_path}'")

    # 2. Boxplot of Planetary Radius
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        x="koi_disposition",
        y="koi_prad",
        data=df_cleaned,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
    )
    plt.title("Planetary Radius by Disposition")
    plt.xlabel("Disposition")
    plt.ylabel("Planetary Radius (Earth Radii)")
    plt.ylim(0, 50)  # Zoom in on the majority of data

    boxplot_path = os.path.join(output_dir, "radius_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Boxplot saved to '{boxplot_path}'")


except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
