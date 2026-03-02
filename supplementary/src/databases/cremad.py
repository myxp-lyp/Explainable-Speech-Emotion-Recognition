import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    train_dataset: any
    test_dataset: any
    num_labels: any
    label_list: any


def load_db(crema_d_root: str):
    """
    Load the CREMA-D dataset from the specified path.

    Args:
        crema_d_root (str): Path to the CREMA-D dataset directory.

    Returns:
        pd.DataFrame: DataFrame containing the loaded and processed dataset.
    """
    wav_folder_path = os.path.join(crema_d_root, "AudioWAV")
    video_demographics_path = os.path.join(crema_d_root, "VideoDemographics.csv")
    summary_table_path = os.path.join(crema_d_root, "processedResults/summaryTable.csv")

    # Load into DF
    df = pd.read_csv(summary_table_path)

    # Select Relevant Columns
    df = df[["FileName", "VoiceVote"]]

    # Add New Columns
    df["path"] = wav_folder_path.rstrip("/") + "/" + df["FileName"] + ".wav"
    df["emotion"] = df["FileName"].str.split("_").str[2].str[0].astype("category")
    df["ActorID"] = df["FileName"].str.split("_").str[0].astype(int)

    # Merge with demographics data
    demographics_df = pd.read_csv(video_demographics_path)
    df = pd.merge(df, demographics_df, on="ActorID", how="outer")

    # Create the new age group column
    age_boundaries = [20, 25, 35, 45, 55, 65, 75]
    age_labels = [
        f"{age_boundaries[i]}-{age_boundaries[i+1]}"
        for i in range(len(age_boundaries) - 1)
    ]
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=age_boundaries,
        labels=age_labels,
        right=False,
    )

    # Shuffle the DataFrame
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    return df


def extract_train_test_sets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Load and split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Training and testing DataFrames.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["emotion"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def visualise_distribution(df: pd.DataFrame, output_file: str = None):
    """
    Visualize the distribution of emotions and age groups in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
    """
    cols = ["emotion", "AgeGroup", "Sex", "Ethnicity", "Race"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    axes = axes.flatten()

    for i, col in enumerate(cols):
        value_counts = df[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    if output_file:
        fig.savefig(output_file)
    plt.tight_layout()
    plt.show()
