import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "datasets/KOI_dataset_1.csv"

try:
    # By adding comment='#', we tell pandas to ignore any lines
    # that start with a hash, which are common in scientific datasets.
    koi_df = pd.read_csv(file_path, comment="#")

    print(f"Successfully loaded '{file_path}'.")

    # --- Initial Exploration and Analysis ---

    print("\n--- First 5 Rows ---")
    # .head() is great for a quick preview of the data's structure and values.
    print(koi_df.head())

    print("\n--- Dataframe Info ---")
    # .info() gives a concise summary of the DataFrame, including column data types
    # and the number of non-null values. It's excellent for spotting missing data.
    koi_df.info()

    print("\n--- Descriptive Statistics ---")
    # .describe() provides statistical data for all numerical columns,
    # like mean, standard deviation, min/max values, and quartiles.
    print(koi_df.describe())

    print("\n--- Dataset Dimensions ---")
    # .shape returns a tuple showing the number of rows and columns.
    print(f"The dataset has {koi_df.shape[0]} rows and {koi_df.shape[1]} columns.")

    print("\n--- Count of Missing Values per Column ---")
    # Chaining .isnull() and .sum() gives a count of missing (NaN) values in each column.
    print(koi_df.isnull().sum())

    # --- Disposition Analysis ---
    # Let's find out what categories are used for classification and which columns are influential.

    print("\n--- Kepler Disposition Categories ---")
    # .value_counts() is perfect for seeing the distribution of categorical data.
    # This shows us the breakdown of Confirmed, Candidate, and False Positive objects.
    print(koi_df["koi_disposition"].value_counts())

    print("\n--- Influence of False Positive Flags ---")
    # The 'koi_fpflag_*' columns are flags set by the Kepler pipeline to identify likely false positives.
    # A value of 1 in these columns strongly suggests the object is not a planet.
    fp_flags = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
    print(koi_df.groupby("koi_disposition")[fp_flags].sum())

    print("\n--- Average Feature Values by Disposition ---")
    # By grouping by the disposition, we can see how the average of key features differs between categories.
    # For example, we expect CONFIRMED planets to have a higher signal-to-noise ratio (koi_model_snr)
    # and FALSE POSITIVES to have different transit characteristics.
    key_features = ["koi_period", "koi_depth", "koi_duration", "koi_model_snr"]
    print(koi_df.groupby("koi_disposition")[key_features].mean())

    # --- Visualizations ---
    # Visualizing the data helps build intuition about the relationships between features.

    print("\n--- Generating Visualizations ---")

    # Set the visual style for the plots
    sns.set_theme(style="whitegrid")

    # 1. Countplot of KOI Dispositions
    # This plot shows the distribution of the target variable. We can see the class imbalance.
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x="koi_disposition",
        data=koi_df,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
    )
    plt.title("Distribution of KOI Dispositions")
    plt.xlabel("Disposition")
    plt.ylabel("Count")
    plt.show()

    # 2. Boxplot of Planetary Radius by Disposition
    # This helps us see if the calculated radius differs between categories.
    # Note how 'FALSE POSITIVE' has a much wider range, including some very large objects
    # that are likely eclipsing binary stars, not planets.
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        x="koi_disposition",
        y="koi_prad",
        data=koi_df,
        order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
    )
    plt.title("Planetary Radius by Disposition")
    plt.xlabel("Disposition")
    plt.ylabel("Planetary Radius (Earth Radii)")
    plt.ylim(0, 50)  # Set a limit to zoom in on the majority of data points
    plt.show()

    # 3. Scatterplot of Orbital Period vs. Planetary Radius
    # This plot can reveal if false positives occupy a different parameter space.
    # We use a log scale for better visualization of wide-ranging data.
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=koi_df,
        x="koi_period",
        y="koi_prad",
        hue="koi_disposition",
        hue_order=["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
        alpha=0.6,
        s=15,
    )
    plt.title("Orbital Period vs. Planetary Radius")
    plt.xlabel("Orbital Period (days) [log scale]")
    plt.ylabel("Planetary Radius (Earth Radii) [log scale]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(title="Disposition")
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print(
        "Please make sure the file is in the correct directory or check the file path."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")

