import pandas as pd
import random
import numpy as np
import torch
import argparse
import mlflow
from collections import Counter
from dataclasses import dataclass, fields, field
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import FacialGNN, FacialTemporalTransformer, MicroExpressionModel

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
    include_augmented, seed, target_col, batch_size
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

    train_dataset = MicroExpressionDataset(train_list, label_map)
    val_dataset = MicroExpressionDataset(val_list, label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=me_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=me_collate_fn)

    train_labels_str = [sample[1] for sample in train_list]
    train_labels_int = [label_map[label] for label in train_labels_str]

    class_counts = Counter(train_labels_int)
    num_classes = len(label_map)

    weights = []
    for i in range(num_classes):
        weights.append(1.0 / class_counts.get(i, 1))

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * num_classes

    return train_loader, val_loader, label_map, weights, len(train_dataset), len(val_dataset)

def _build_model(num_classes, transformer_embed_dim, **model_params):
    gnn_hidden_dims = model_params.get("gnn_hidden_dims", [32, 64, 128])

    gnn = FacialGNN(
        hidden_dims=gnn_hidden_dims,
        fusion_dim=model_params["gnn_fusion_dim"],
        dropout=model_params["gnn_dropout"]
    )

    transformer = FacialTemporalTransformer(
        embed_dim=transformer_embed_dim,
        num_layers=model_params["transformer_num_layers"],
        num_heads=model_params["transformer_num_heads"],
        ff_dim=model_params["transformer_ff_dim"],
        dropout=model_params["transformer_dropout"],
        use_cls_token=model_params["transformer_use_cls_token"]
    )

    model = MicroExpressionModel(
        gnn=gnn,
        transformer=transformer,
        num_classes=num_classes,
        embed_dim=transformer_embed_dim
    )
    return model

def _train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm):
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

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1

def _validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for lm_list, labels, num_frames in loader:
            landmarks_seqs = [[frame.to(device) for frame in seq] for seq in lm_list]
            labels = labels.to(device)
            num_frames = num_frames.to(device)

            logits = model(landmarks_seqs, num_frames)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_labels, all_preds

@dataclass
class DatasetConfig:
    data_root: str = 'data/augmented/casme3'
    labels_path: str = 'data/augmented/casme3/labels.xlsx'
    val_size: float = 0.2
    subject_independent: bool = True
    include_augmented: bool = False
    seed: int = 1
    target_col: str = 'Objective class'

@dataclass
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    patience_early_stop: int = 10
    grad_clip_max_norm: float = 1.0
    device: str = 'cuda'

@dataclass
class GNNConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    fusion_dim: int | None = 256
    dropout: float = 0.2

@dataclass
class TransformerConfig:
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.1
    use_cls_token: bool = True

def train(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    training_cfg: TrainingConfig = TrainingConfig(),
    gnn_cfg: GNNConfig = GNNConfig(),
    transformer_cfg: TransformerConfig = TransformerConfig(),
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, label_map, weights, train_samples, val_samples = _build_dataloaders(
        data_root=dataset_cfg.data_root,
        labels_path=dataset_cfg.labels_path,
        val_size=dataset_cfg.val_size,
        subject_independent=dataset_cfg.subject_independent,
        include_augmented=dataset_cfg.include_augmented,
        seed=dataset_cfg.seed,
        target_col=dataset_cfg.target_col,
        batch_size=training_cfg.batch_size,
    )
    weights = weights.to(device)

    num_classes = len(label_map)

    # 2. Model
    model_params = {
        "gnn_hidden_dims": gnn_cfg.hidden_dims,
        "gnn_fusion_dim": gnn_cfg.fusion_dim,
        "gnn_dropout": gnn_cfg.dropout,
        "transformer_num_layers": transformer_cfg.num_layers,
        "transformer_num_heads": transformer_cfg.num_heads,
        "transformer_ff_dim": transformer_cfg.ff_dim,
        "transformer_dropout": transformer_cfg.dropout,
        "transformer_use_cls_token": transformer_cfg.use_cls_token,
    }

    model = _build_model(num_classes, transformer_cfg.embed_dim, **model_params).to(device)

    # 3. Optimizer & co
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    mlflow.set_experiment('ME_model_training')
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params({**locals(), **model_params, "num_classes": num_classes,
                           "train_samples": train_samples, "val_samples": val_samples})

        save_dir = Path('models/me_model') / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        last_path = save_dir / "last_model.pth"
        best_path = save_dir / "best_model.pth"

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(training_cfg.num_epochs):
            train_loss, train_acc, train_f1 = _train_epoch(
                model, train_loader, optimizer, criterion, device, training_cfg.grad_clip_max_norm
            )

            val_loss, val_acc, val_f1, val_labels, val_preds = _validate_epoch(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)

            scheduler.step(val_loss)

            # Save checkpoints
            torch.save(model.state_dict(), last_path)
            mlflow.log_artifact(str(last_path), "checkpoints")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path), "checkpoints")
                early_stop_counter = 0
                print(f"*** New best: {val_loss:.4f} ***")
            else:
                early_stop_counter += 1

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Best Val Loss: {best_val_loss:.4f}")

            if early_stop_counter >= training_cfg.patience_early_stop:
                print("Early stopping")
                break

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
    parser.add_argument('--target_col', type=str, default='Objective class')

    # ==================== TrainingConfig ====================
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience_early_stop', type=int, default=10)
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

    # ==================== GNNConfig ====================
    parser.add_argument('--gnn_hidden_dims', type=int, nargs='+', default=[32, 64, 128], help="Example: --gnn_hidden_dims 32 64 128")
    parser.add_argument('--gnn_fusion_dim', type=int, default=256, help="Use negative value (e.g. -1) to set None")
    parser.add_argument('--gnn_dropout', type=float, default=0.2)

    # ==================== TransformerConfig ====================
    parser.add_argument('--transformer_embed_dim', type=int, default=256)
    parser.add_argument('--transformer_num_layers', type=int, default=4)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--transformer_ff_dim', type=int, default=None, help="Use negative value (e.g. -1) to set None")
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--use_cls_token', action='store_true', default=True)
    parser.add_argument('--no_cls_token', action='store_false', dest='use_cls_token')

    args = parser.parse_args()
    
    dataset_cfg = DatasetConfig(
        data_root=args.data_root,
        labels_path=args.labels_path,
        val_size=args.val_size,
        subject_independent=args.subject_independent,
        include_augmented=args.include_augmented,
        seed=args.seed,
        target_col=args.target_col,
    )

    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience_early_stop=args.patience_early_stop,
        grad_clip_max_norm=args.grad_clip_max_norm,
        device=args.device,
    )

    gnn_cfg = GNNConfig(
        hidden_dims=args.gnn_hidden_dims,
        fusion_dim=args.gnn_fusion_dim,
        dropout=args.gnn_dropout,
    )

    transformer_cfg = TransformerConfig(
        embed_dim=args.transformer_embed_dim,
        num_layers=args.transformer_num_layers,
        num_heads=args.transformer_num_heads,
        ff_dim=args.transformer_ff_dim,
        dropout=args.transformer_dropout,
        use_cls_token=args.use_cls_token,
    )

    train(dataset_cfg, training_cfg, gnn_cfg, transformer_cfg)