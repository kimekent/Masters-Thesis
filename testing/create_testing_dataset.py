"""
This scripts splits the web support ticketing dataset into a train, test and validation data set.

To run this script update the 'path' variable to the root project directory.
"""

# Define root directory path for this project
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis"

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV data into a DataFrame
df = pd.read_csv(path + r"\testing\testing_data\cleaned_websupport_questions_with_intents_utf-8.csv", encoding= "utf-8")

# Group the data by the "intent" column,
# ensuring proportional representation of each intent across the train, test, and validation sets
intent_groups = df.groupby("intent")

# Initialize empty DataFrames for training, testing, and validation
train_df = pd.DataFrame()
test_df = pd.DataFrame()
val_df = pd.DataFrame()

# Minimum number of samples required for splitting
min_samples = 3

# Split each intent group into training, testing, and validation subsets
for group_name, group_data in intent_groups:
    if len(group_data) >= min_samples:
        group_train, group_test = train_test_split(group_data, test_size=0.2, random_state=42)
        group_train, group_val = train_test_split(group_train, test_size=0.1, random_state=42)

        train_df = pd.concat([train_df, group_train])
        test_df = pd.concat([test_df, group_test])
        val_df = pd.concat([val_df, group_val])
    else:
        # If the group has too few samples, add it to the training set (adjust as needed)
        train_df = pd.concat([train_df, group_data])

# Shuffle the dataframes
train_df = train_df.sample(frac=1, random_state=42)
test_df = test_df.sample(frac=1, random_state=42)
val_df = val_df.sample(frac=1, random_state=42)

# Save the split datasets to separate CSV files if needed
train_df.to_csv(r"testing_data\websupport_train_dataset.csv", index=False, encoding = "utf-8")
test_df.to_csv(r"testing_data\test_dataset.csv", index=False, encoding = "utf-8")
val_df.to_csv(r"testing_data\validation_dataset.csv", index=False, encoding = "utf-8")