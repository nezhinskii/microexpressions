import pandas as pd
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)

def prepare_microexpression_datasets(
    data_root: str = 'data/augmented/casme3',
    labels_path: str = 'data/augmented/casme3/labels.xlsx',
    val_size: float = 0.2,
    subject_independent: bool = True,
    include_augmented: bool = False,
    seed: int = 1,
    target_col: str = 'emotion',
):
    set_seed(seed)
    data_root = Path(data_root)
    labels_path = Path(labels_path)

    df_labels = pd.read_excel(labels_path)
    df_labels['Subject'] = df_labels['Subject'].astype(str).str.strip()
    df_labels['Filename'] = df_labels['Filename'].astype(str).str.strip()
    df_labels['Onset'] = df_labels['Onset'].astype(str).str.strip()
    df_labels['folder_name'] = (
        df_labels['Subject'] + '_' + 
        df_labels['Filename'] + '_' + 
        df_labels['Onset']
    )
    
    existing_folders = {p.name for p in data_root.iterdir() if p.is_dir()}
    valid_rows = df_labels['folder_name'].isin(existing_folders)
    df_valid = df_labels[valid_rows].reset_index(drop=True) 
    fragments = df_valid[['folder_name', 'Subject', target_col]].drop_duplicates()
    fragments[target_col] = fragments[target_col].astype(str).str.lower()

    if subject_independent:
        unique_subjects = fragments['Subject'].unique().tolist()
        random.shuffle(unique_subjects)
        total_frags = len(fragments)
        train_subjects = []
        val_subjects = []
        val_full = False
        for i, subj in enumerate(unique_subjects):
            if (i % 2 == 0) or val_full:
                train_subjects.append(subj)
            else:
                val_subjects.append(subj)
                if len(fragments[fragments['Subject'].isin(val_subjects)]) >= val_size * total_frags:
                    val_full = True

        train_fragments = fragments[fragments['Subject'].isin(train_subjects)]
        val_fragments = fragments[fragments['Subject'].isin(val_subjects)]

        print(f"  Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
        print(f"  Train fragments: {len(train_fragments)}, Val fragments: {len(val_fragments)}")
        print("Target distribution train:", train_fragments['emotion'].value_counts(normalize=True).to_dict())
        print("Target distribution val:", val_fragments['emotion'].value_counts(normalize=True).to_dict())
    else:
        train_fragments, val_fragments = train_test_split(
            fragments,
            test_size=val_size,
            random_state=seed,
            stratify=fragments['emotion']
        )
        print(f"  Train fragments: {len(train_fragments)}, Val fragments: {len(val_fragments)}")
        print("Target distribution train:", train_fragments['emotion'].value_counts(normalize=True).to_dict())
        print("Target distribution val:", val_fragments['emotion'].value_counts(normalize=True).to_dict())
    
    def make_dataset(fragments_df, include_augmented_in_train=False):
        samples = []
        for _, row in fragments_df.iterrows():
            folder_name = row['folder_name']
            emotion = row['emotion']

            original_csv = data_root / folder_name / "procrustes_lm_original.csv"
            if original_csv.exists():
                samples.append((str(original_csv), emotion))
    
            if include_augmented_in_train:
                for aug_csv in (data_root / folder_name).glob("procrustes_lm_*.csv"):
                    if aug_csv.name != "procrustes_lm_original.csv":
                        samples.append((str(aug_csv), emotion))
        
        return samples
    
    train_list = make_dataset(train_fragments, include_augmented_in_train=include_augmented)
    val_list = make_dataset(val_fragments, include_augmented_in_train=False)
    random.shuffle(train_list)
    random.shuffle(val_list)

    label_map = { cl_name:i for i, cl_name in enumerate(fragments[target_col].unique())}
    return train_list, val_list, label_map


class MicroExpressionDataset(Dataset):
    def __init__(self, sample_list, label_map):
        self.samples = sample_list
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label_str = self.samples[idx]
        label = self.label_map[label_str]

        df = pd.read_csv(csv_path, delimiter=';')
        num_frames = len(df)

        landmarks_list = []
        for i in range(num_frames):
            row = df.iloc[i]
            points = []
            for j in range(68):
                x = row[f'n_x{j}']
                y = row[f'n_y{j}']
                points.append([x, y])
            points = torch.tensor(points, dtype=torch.float)
            landmarks_list.append(points)

        return landmarks_list, label, num_frames
    

def me_collate_fn(batch):
    landmarks_seqs = []
    labels = []
    lengths = []

    for landmarks_list, label, num_frames in batch:
        landmarks_seqs.append(landmarks_list)
        labels.append(label)
        lengths.append(num_frames)

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return landmarks_seqs, labels, lengths
