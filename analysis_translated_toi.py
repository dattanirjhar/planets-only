import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
input_file = "datasets/TOI_translated_dataset.csv"
output_dir = "charts"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def analyze_translated_data():
    """
    Loads the translated TESS data and performs an exploratory data analysis (EDA)
    to understand its properties and distributions.
    """
    try:
        # Load the translated and cleaned TESS data
        print(f"--- Loading translated TESS data from '{input_file}' ---")
        df = pd.read_csv(input_file)
        print("✅ Data loaded successfully.")

        # --- Initial Verification ---
        print("\n--- Dataframe Info ---")
        df.info()

        print("\n--- Translated Dataset Dimensions ---")
        print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

        print("\n--- Descriptive Statistics ---")
        # This gives us a feel for the ranges and distributions of the new data.
        # Are TESS planets typically smaller? On shorter periods? This will tell us.
        print(df.describe())

        # --- Visualization for Analysis ---
        print("\n--- Generating Analysis Plots ---")

        # 1. Countplot of Dispositions
        # How does the class balance in TESS compare to Kepler?
        plt.figure(figsize=(8, 6))
        sns.countplot(
            x="koi_disposition",
            data=df,
            order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
            palette="viridis",
        )
        plt.title("Count of Dispositions in Translated TESS Data")
        plt.xlabel("Disposition")
        plt.ylabel("Count")

        countplot_path = os.path.join(output_dir, "tess_disposition_countplot.png")
        plt.savefig(countplot_path)
        plt.close()
        print(f"✅ Disposition countplot saved to '{countplot_path}'")

        # 2. Boxplot of Planetary Radius by Disposition
        # This is a critical comparison. Does the 'koi_prad' feature show the
        # same strong separation for TESS data as it did for Kepler?
        plt.figure(figsize=(10, 7))
        sns.boxplot(
            x="koi_disposition",
            y="koi_prad",
            data=df,
            order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
            palette="magma",
        )
        plt.title("Planetary Radius by Disposition (TESS Data)")
        plt.xlabel("Disposition")
        plt.ylabel("Planetary Radius (Earth Radii)")
        plt.ylim(0, 30)  # Zoom in, as TESS is better at finding smaller planets

        boxplot_path = os.path.join(output_dir, "tess_radius_boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"✅ Planetary radius boxplot saved to '{boxplot_path}'")

        print("\n✅ Analysis complete!")

    except FileNotFoundError:
        print(
            f"❌ ERROR: The file '{input_file}' was not found. Have you run the 'translate_toi_data.py' script yet?"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    analyze_translated_data()
