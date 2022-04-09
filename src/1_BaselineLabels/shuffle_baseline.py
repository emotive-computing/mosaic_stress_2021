import os
import numpy as np
import pandas as pd

data_file_path = "tesserae_dataset.csv" # This should point to your downloaded Tesserate data
label_column = "stress.d"
subject_id_column = "snapshot_id"
output_folder = "./results"

def ComputeShuffledBaseline():
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(data_file_path)

    # Random shuffle across all valid labels
    labels = df[label_column]
    label_valid_mask = ~pd.isnull(labels)
    shuffled_df = labels[label_valid_mask].sample(frac=1)

    out_df = df
    out_df.loc[label_valid_mask, label_column + '_shuffled'] = shuffled_df.values
    out_df[label_column + '_shuffledWithinSubject'] = np.nan*np.zeros(out_df.shape[0])

    # Random shuffle within each subject across all valid labels
    subj_ids = np.unique(df[subject_id_column].values)
    for subj_id in subj_ids:
        subj_mask = df[subject_id_column] == subj_id
        valid_subj_labels_mask = np.logical_and(~pd.isnull(df[label_column]), subj_mask)
        subj_valid_labels = df.loc[valid_subj_labels_mask, label_column]
        shuffled_subj_labels = subj_valid_labels.sample(frac=1)
        out_df.loc[valid_subj_labels_mask, label_column + '_shuffledWithinSubject'] = shuffled_subj_labels.values

    out_file_path = os.path.join(output_folder, os.path.basename(data_file_path)[:-4]+"_with_baseline.csv")
    out_df.to_csv(out_file_path, index=False, header=True)

    return

if __name__ == '__main__':
    ComputeShuffledBaseline()
