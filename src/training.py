import pandas as pd
import random
import numpy as np
import torch
import argparse
import mlflow
from collections import Counter
from dataclasses import dataclass, fields, field
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import FacialGNN, FacialTemporalTransformer, MicroExpressionModel, MILPooling
from focal_loss import FocalLoss
import pytorch_warmup as warmup

@dataclass
class DatasetConfig:
    data_root: str = 'data/augmented/casme3'
    labels_path: str = 'data/augmented/casme3/labels.xlsx'
    val_size: float = 0.2
    subject_independent: bool = True
    include_augmented: bool = False
    seed: int = 1
    target_col: str = 'Objective class'
    drop_others: bool = True
    remap_classes: bool = False

@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    patience_early_stop: int = 10
    grad_clip_max_norm: float = 1.0
    device: str = 'cuda'
    scheduler: str = 'cosine'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 7

@dataclass
class GNNConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    fusion_dim: int | None = None
    dropout: float = 0.2

@dataclass
class TransformerConfig:
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.1
    use_cls_token: bool = True

@dataclass
class MILConfig:
    d_model: int = 512
    kernel_size: int = 3
    num_conv_layers: int = 1
    dropout: float = 0.3
    temperature: float = 1.0
    use_residual: bool = False

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
    target_col: str = 'Objective class',
):
    set_seed(seed)
    data_root = Path(data_root)
    labels_path = Path(labels_path)
    exclude_fragments = set()
    bad_fragments_path = Path(data_root, 'bad_fragments.txt')
    if bad_fragments_path.exists():
        with open(bad_fragments_path, "r") as fd:
            exclude_fragments = {f.strip() for f in fd.readlines()}

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
    existing_folders = existing_folders - exclude_fragments
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
        print("Target distribution train:", train_fragments[target_col].value_counts(normalize=True).to_dict())
        print("Target distribution val:", val_fragments[target_col].value_counts(normalize=True).to_dict())
    else:
        train_fragments, val_fragments = train_test_split(
            fragments,
            test_size=val_size,
            random_state=seed,
            stratify=fragments[target_col]
        )
        print(f"  Train fragments: {len(train_fragments)}, Val fragments: {len(val_fragments)}")
        print("Target distribution train:", train_fragments[target_col].value_counts(normalize=True).to_dict())
        print("Target distribution val:", val_fragments[target_col].value_counts(normalize=True).to_dict())
    
    def make_dataset(target_col, fragments_df, include_augmented_in_train=False):
        samples = []
        for _, row in fragments_df.iterrows():
            folder_name = row['folder_name']
            target_col_value = row[target_col]

            original_csv = data_root / folder_name / "procrustes_lm_original.csv"
            if original_csv.exists():
                samples.append((str(original_csv), target_col_value))
    
            if include_augmented_in_train:
                for aug_csv in (data_root / folder_name).glob("procrustes_lm_*.csv"):
                    if aug_csv.name != "procrustes_lm_original.csv":
                        samples.append((str(aug_csv), target_col_value))
        
        return samples
    
    train_list = make_dataset(target_col, train_fragments, include_augmented_in_train=include_augmented)
    val_list = make_dataset(target_col, val_fragments, include_augmented_in_train=False)
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


def _build_dataloaders(
    data_root, labels_path, val_size, subject_independent,
    include_augmented, seed, target_col, batch_size, drop_others, remap_classes
):
    train_list, val_list, label_map = prepare_microexpression_datasets(
        data_root=data_root,
        labels_path=labels_path,
        val_size=val_size,
        subject_independent=subject_independent,
        include_augmented=include_augmented,
        seed=seed,
        target_col=target_col,
    )

    class_mapping = {
        'happy':        'positive',
        'happiness':    'positive',

        'surprise':     'surprise',

        'disgust':      'negative',
        'fear':         'negative',
        'anger':        'negative',
        'sad':          'negative',
        'sadness':      'negative',
        'contempt':     'negative',

        'others':       'others',
        'repression':   'others',
        'tense':        'others'
    }

    def remap_sample_list(sample_list):
        remapped = []
        for csv_path, label_str in sample_list:
            label_str = label_str.lower().strip()
            new_label = class_mapping.get(label_str, 'others')
            remapped.append((csv_path, new_label))
        return remapped
    
    if remap_classes:
        train_list_remapped = remap_sample_list(train_list)
        val_list_remapped   = remap_sample_list(val_list)
    else:
        train_list_remapped = train_list
        val_list_remapped   = val_list

    if Path(data_root).parts[-1] == 'casmeii':
        train_list_remapped = [s for s in train_list_remapped if (s[1] != 'fear') and (s[1] != 'sadness')]
        val_list_remapped   = [s for s in val_list_remapped   if (s[1] != 'fear') and (s[1] != 'sadness')]

    if drop_others:
        train_list_remapped = [s for s in train_list_remapped if s[1] != 'others']
        val_list_remapped   = [s for s in val_list_remapped   if s[1] != 'others']
        print("Dropped 'others' class after remapping.")
    else:
        print("Kept 'others' class after remapping.")

    unique_labels = sorted(set(label for _, label in train_list_remapped + val_list_remapped))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    train_labels = [label for _, label in train_list_remapped]
    val_labels = [label for _, label in val_list_remapped]
    full_labels = [*train_labels, *val_labels] 
    print(f"Classes after remapping: {unique_labels}")
    print("Full distribution:", {k: (full_labels.count(k), full_labels.count(k) / len(full_labels)) for k in unique_labels})
    print("Train distribution:", {k: (train_labels.count(k), train_labels.count(k) / len(train_labels)) for k in unique_labels})
    print("Val distribution:  ", {k: (val_labels.count(k), val_labels.count(k)   / len(val_labels))   for k in unique_labels})

    train_dataset = MicroExpressionDataset(train_list_remapped, label_map)
    val_dataset = MicroExpressionDataset(val_list_remapped, label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=me_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=me_collate_fn)

    train_labels_str = [sample[1] for sample in train_list_remapped]
    train_labels_int = [label_map[label] for label in train_labels_str]

    val_labels_str = [sample[1] for sample in val_list_remapped]
    val_labels_int = [label_map[label] for label in val_labels_str]

    class_counts = Counter(train_labels_int)
    num_classes = len(label_map)

    weights = []
    for i in range(num_classes):
        weights.append(1.0 / class_counts.get(i, 1))
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * num_classes

    train_class_counts = Counter(train_labels_int)
    val_class_counts = Counter(val_labels_int)

    return (train_loader, val_loader, label_map, weights,
            len(train_dataset), len(val_dataset), class_mapping,
            train_class_counts, val_class_counts)

def _build_model(
    num_classes: int,
    gnn_cfg: GNNConfig,
    temporal_cfg: TransformerConfig | MILConfig,
    temporal_model: str = 'transformer'
):
    # ----- GNN -----
    gnn = FacialGNN(
        hidden_dims=gnn_cfg.hidden_dims,
        fusion_dim=gnn_cfg.fusion_dim,
        dropout=gnn_cfg.dropout
    )

    # ----- Temporal module -----
    if temporal_model == 'transformer':
        assert isinstance(temporal_cfg, TransformerConfig)
        temporal_module = FacialTemporalTransformer(
            embed_dim=temporal_cfg.embed_dim,
            num_layers=temporal_cfg.num_layers,
            num_heads=temporal_cfg.num_heads,
            ff_dim=temporal_cfg.ff_dim,
            dropout=temporal_cfg.dropout,
            use_cls_token=temporal_cfg.use_cls_token
        )
        embed_dim = temporal_cfg.embed_dim
    elif temporal_model == 'mil':
        assert isinstance(temporal_cfg, MILConfig)
        frame_dim = gnn_cfg.fusion_dim if gnn_cfg.fusion_dim is not None else gnn_cfg.hidden_dims[-1]

        temporal_module = MILPooling(
            frame_dim=frame_dim,
            d_model=temporal_cfg.d_model,
            kernel_size=temporal_cfg.kernel_size,
            num_conv_layers=temporal_cfg.num_conv_layers,
            dropout=temporal_cfg.dropout,
            temperature=temporal_cfg.temperature,
            use_residual=temporal_cfg.use_residual
        )
        embed_dim = temporal_cfg.d_model
    else:
        raise ValueError(f"Unknown temporal_model: {temporal_model}")

    model = MicroExpressionModel(
        gnn=gnn,
        temporal_module=temporal_module,
        num_classes=num_classes,
        embed_dim=embed_dim,
        temporal_model=temporal_model
    )
    return model

def _train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm, lr_scheduler=None, warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for lm_list, labels, num_frames in loader:
        landmarks_seqs = [[frame.to(device) for frame in seq] for seq in lm_list]
        labels = labels.to(device)
        num_frames = num_frames.to(device)

        optimizer.zero_grad()
        logits = model(landmarks_seqs, num_frames)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        if (lr_scheduler is not None) and (warmup_scheduler is not None):
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1

def _validate_epoch(model, loader, criterion, device, return_extras=False):
    model.eval()
    val_loss = 0.0
    val_labels = []
    val_preds = []
    
    val_scores = [] if return_extras else None
    val_attn_weights = [] if return_extras else None
    val_lengths = [] if return_extras else None

    with torch.no_grad():
        for landmarks_seqs, labels, lengths in loader:
            landmarks_seqs = [seq.to(device) for seq in landmarks_seqs]
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(landmarks_seqs, lengths, return_extras=return_extras)
            
            if return_extras:
                logits, scores, attn_weights = outputs
            else:
                logits = outputs

            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

            if return_extras:
                val_scores.append(scores.detach().cpu())
                val_attn_weights.append(attn_weights.detach().cpu())
                val_lengths.append(lengths.cpu())

    val_loss /= len(loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    if return_extras:
        val_scores = torch.cat(val_scores, dim=0).numpy()
        val_attn_weights = torch.cat(val_attn_weights, dim=0).numpy()
        val_lengths = torch.cat(val_lengths, dim=0).numpy()
        return val_loss, val_acc, val_f1, np.array(val_labels), np.array(val_preds), \
               val_scores, val_attn_weights, val_lengths
    else:
        return val_loss, val_acc, val_f1, np.array(val_labels), np.array(val_preds)

def train(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    training_cfg: TrainingConfig = TrainingConfig(),
    gnn_cfg: GNNConfig = GNNConfig(),
    transformer_cfg: TransformerConfig = TransformerConfig(),
    mil_cfg: MILConfig = MILConfig(),
    temporal_model: str = 'transformer',
    debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using temporal model: {temporal_model}")

    (train_loader, val_loader, label_map, weights, train_samples, val_samples,
     class_mapping, train_class_counts, val_class_counts) = _build_dataloaders(
        data_root=dataset_cfg.data_root,
        labels_path=dataset_cfg.labels_path,
        val_size=dataset_cfg.val_size,
        subject_independent=dataset_cfg.subject_independent,
        include_augmented=dataset_cfg.include_augmented,
        seed=dataset_cfg.seed,
        target_col=dataset_cfg.target_col,
        batch_size=training_cfg.batch_size,
        drop_others=dataset_cfg.drop_others,
        remap_classes=dataset_cfg.remap_classes
    )
    weights = weights.to(device)
    num_classes = len(label_map)

    temporal_cfg = transformer_cfg if temporal_model == 'transformer' else mil_cfg
    model = _build_model(
        num_classes=num_classes,
        gnn_cfg=gnn_cfg,
        temporal_cfg=temporal_cfg,
        temporal_model=temporal_model
    ).to(device)

    # 3. Optimizer & co
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # criterion = FocalLoss(gamma=2.5, alpha=weights, task_type='multi-class', num_classes=len(weights))
    optimizer = AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    lr_scheduler = None
    if training_cfg.scheduler == 'cosine':
        num_steps = len(train_loader) * training_cfg.num_epochs
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    elif training_cfg.scheduler == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=training_cfg.scheduler_factor, patience=training_cfg.scheduler_patience, verbose=True)
        warmup_scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {training_cfg.scheduler}")

    mlflow.set_experiment('ME_model_training')
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        params_to_log = {
            "temporal_model": temporal_model,
            "num_classes": num_classes,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "classes": ", ".join(sorted(label_map.keys())),
            "class_mapping": class_mapping,
            "loss": "FocalLoss" if isinstance(criterion, FocalLoss) else "CrossEntropyLoss",
        }
        if isinstance(criterion, FocalLoss):
            params_to_log["focal_loss_gamma"] = criterion.gamma

        params_to_log.update({
            "dataset_cfg": dataset_cfg,
            "training_cfg": training_cfg,
            "gnn_cfg": gnn_cfg,
        })

        if temporal_model == 'transformer':
            params_to_log["temporal_cfg"] = transformer_cfg
        else:
            params_to_log["temporal_cfg"] = mil_cfg

        for cls_name, idx in label_map.items():
            params_to_log[f"train_{cls_name}"] = train_class_counts.get(idx, 0)
            params_to_log[f"val_{cls_name}"]   = val_class_counts.get(idx, 0)

        mlflow.log_params(params_to_log)

        save_dir = Path('models/me_model') / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        last_path = save_dir / "last_model.pth"
        best_path = save_dir / "best_model.pth"

        best_val_f1 = 0.0
        early_stop_counter = 0

        for epoch in range(training_cfg.num_epochs):
            train_loss, train_acc, train_f1 = _train_epoch(
                model=model, 
                loader=train_loader, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=device, 
                grad_clip_norm=training_cfg.grad_clip_max_norm, 
                lr_scheduler=lr_scheduler if training_cfg.scheduler == 'cosine' else None, 
                warmup_scheduler=warmup_scheduler
            )

            val_outputs = _validate_epoch(
                model, val_loader, criterion, device, return_extras=debug
            )

            if debug:
                val_loss, val_acc, val_f1, val_labels, val_preds, \
                val_scores, val_attn, val_lengths = val_outputs
            else:
                val_loss, val_acc, val_f1, val_labels, val_preds = val_outputs

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)

            if training_cfg.scheduler == 'plateau':
                lr_scheduler.step(val_loss)

            # Save checkpoints
            torch.save(model.state_dict(), last_path)
            mlflow.log_artifact(str(last_path), "checkpoints")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path), "checkpoints")
                early_stop_counter = 0
                best_cm = confusion_matrix(val_labels, val_preds)
                best_cm_path = save_dir / "best_confusion_matrix.txt"
                np.savetxt(best_cm_path, best_cm, fmt="%d")
                mlflow.log_artifact(str(best_cm_path))
                print(f"*** New best: {val_f1:.4f} ***")
            else:
                early_stop_counter += 1

            if ((val_f1 > best_val_f1) or (epoch == training_cfg.num_epochs - 1)) and debug:
                prefix = f"epoch_{epoch}"
                np.save(save_dir / f"{prefix}_val_scores.npy", val_scores)
                np.save(save_dir / f"{prefix}_val_attn_weights.npy", val_attn)
                np.save(save_dir / f"{prefix}_val_lengths.npy", val_lengths)
                np.save(save_dir / f"{prefix}_val_labels.npy", val_labels)
                np.save(save_dir / f"{prefix}_val_preds.npy", val_preds)

                mlflow.log_artifact(str(save_dir / f"{prefix}_val_scores.npy"))
                mlflow.log_artifact(str(save_dir / f"{prefix}_val_attn_weights.npy"))
                mlflow.log_artifact(str(save_dir / f"{prefix}_val_lengths.npy"))
                mlflow.log_artifact(str(save_dir / f"{prefix}_val_labels.npy"))
                mlflow.log_artifact(str(save_dir / f"{prefix}_val_preds.npy"))

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Best Val F1: {best_val_f1:.4f}")

            # if early_stop_counter >= training_cfg.patience_early_stop:
            #     print("Early stopping")
            #     break

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        cm_path = save_dir / "confusion_matrix.txt"
        np.savetxt(cm_path, cm, fmt="%d")
        mlflow.log_artifact(str(cm_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroExpression model")

    # ==================== DatasetConfig ====================
    parser.add_argument('--data_root', type=str, default='data/augmented/casme3')
    parser.add_argument('--labels_path', type=str, default='data/augmented/casme3/labels.xlsx')
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--subject_independent', action='store_true', default=True)
    parser.add_argument('--no_subject_independent', action='store_false', dest='subject_independent')
    parser.add_argument('--include_augmented', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--target_col', type=str, default='emotion')
    parser.add_argument('--drop_others', action='store_true', default=True, help="Drop samples with 'others' class after remapping to 3-class scheme")
    parser.add_argument('--no_drop_others', action='store_false', dest='drop_others')
    parser.add_argument('--remap_classes', action='store_true', default=False, help="Remap classes to positive, negative, surprise")

    # ==================== TrainingConfig ====================
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience_early_stop', type=int, default=40)
    parser.add_argument('--grad_clip_max_norm', type=float, default=5.0)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau'], default='cosine')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="For plateau scheduler")
    parser.add_argument('--scheduler_patience', type=int, default=7, help="For plateau scheduler")

    # ==================== Debug options ====================
    parser.add_argument('--debug', action='store_true', default=False, help="If set, collect and log MIL attention scores/weights on validation")

    # ==================== GNNConfig ====================
    parser.add_argument('--gnn_hidden_dims', type=int, nargs='+', default=[64, 128, 256], help="Example: --gnn_hidden_dims 32 64 128")
    parser.add_argument('--gnn_fusion_dim', type=int, default=256, help="Use negative value (e.g. -1) to set None")
    parser.add_argument('--gnn_dropout', type=float, default=0.3)

    # ==================== Temporal Model Selection ====================
    parser.add_argument('--temporal_model', type=str, choices=['transformer', 'mil'], default='transformer', help="Choose temporal model: transformer or mil")

    # ==================== TransformerConfig ====================
    parser.add_argument('--transformer_embed_dim', type=int, default=256)
    parser.add_argument('--transformer_num_layers', type=int, default=6)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--transformer_ff_dim', type=int, default=None, help="Use negative value (e.g. -1) to set None")
    parser.add_argument('--transformer_dropout', type=float, default=0.2)
    parser.add_argument('--use_cls_token', action='store_true', default=True)
    parser.add_argument('--no_cls_token', action='store_false', dest='use_cls_token')

    # ==================== MILConfig ====================
    parser.add_argument('--mil_d_model', type=int, default=512)
    parser.add_argument('--mil_kernel_size', type=int, default=3)
    parser.add_argument('--mil_num_conv_layers', type=int, default=1)
    parser.add_argument('--mil_dropout', type=float, default=0.3)
    parser.add_argument('--mil_temperature', type=float, default=1.0)
    parser.add_argument('--mil_use_residual', action='store_true', default=False)
    parser.add_argument('--no_mil_use_residual', action='store_false', dest='mil_use_residual')

    args = parser.parse_args()
    
    dataset_cfg = DatasetConfig(
        data_root=args.data_root,
        labels_path=args.labels_path,
        val_size=args.val_size,
        subject_independent=args.subject_independent,
        include_augmented=args.include_augmented,
        seed=args.seed,
        target_col=args.target_col,
        drop_others=args.drop_others,
        remap_classes=args.remap_classes
    )

    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience_early_stop=args.patience_early_stop,
        grad_clip_max_norm=args.grad_clip_max_norm,
        device=args.device,
        scheduler=args.scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )

    gnn_fusion_dim = args.gnn_fusion_dim
    if (args.gnn_fusion_dim is None) or (args.gnn_fusion_dim < 0):
        gnn_fusion_dim = None 
    gnn_cfg = GNNConfig(
        hidden_dims=args.gnn_hidden_dims,
        fusion_dim=gnn_fusion_dim,
        dropout=args.gnn_dropout,
    )

    transformer_ff_dim = args.transformer_ff_dim
    if (args.transformer_ff_dim is None) or (args.transformer_ff_dim < 0):
        transformer_ff_dim = None 
    transformer_cfg = TransformerConfig(
        embed_dim=args.transformer_embed_dim,
        num_layers=args.transformer_num_layers,
        num_heads=args.transformer_num_heads,
        ff_dim=transformer_ff_dim,
        dropout=args.transformer_dropout,
        use_cls_token=args.use_cls_token,
    )

    mil_cfg = MILConfig(
        d_model=args.mil_d_model,
        kernel_size=args.mil_kernel_size,
        num_conv_layers=args.mil_num_conv_layers,
        dropout=args.mil_dropout,
        temperature=args.mil_temperature,
        use_residual=args.mil_use_residual,
    )

    train(dataset_cfg, training_cfg, gnn_cfg, transformer_cfg, mil_cfg, args.temporal_model, args.debug)