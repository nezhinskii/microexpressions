import pandas as pd
import random
import numpy as np
import torch
import mlflow
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import FacialGNN, FacialTemporalTransformer, MicroExpressionModel, MILPooling, FacialRNN
from focal_loss import FocalLoss
import pytorch_warmup as warmup
from train_arguments import create_training_parser, create_train_configs, TrainingConfig, DatasetConfig, GNNConfig, TransformerConfig, MILConfig, RNNConfig

def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    
def filter_bad_fragments(fragments: pd.DataFrame, data_root:Path):
    filtered_fragmetns = fragments
    
    bad_fragments_path = Path(data_root, 'bad_fragments.txt')
    if bad_fragments_path.exists():
        with open(bad_fragments_path, "r") as fo:
            exclude_fragments = {f.strip() for f in fo.readlines()}
            filtered_fragmetns = filtered_fragmetns[~filtered_fragmetns['folder_name'].isin(exclude_fragments)]
    
    bad_images_path = Path(data_root, 'bad_images.txt')
    if bad_images_path.exists():
        with open(bad_images_path, "r") as fo:
            bad_images = {}
            for line in fo.readlines():
                subject, filename, frame = line.split(' ')
                frame = int(frame.strip())
                if subject not in bad_images:
                    bad_images[subject] = {}
                if filename not in bad_images[subject]:
                    bad_images[subject][filename] = set()
                bad_images[subject][filename].add(frame)
            
            valid_rows = np.full(len(filtered_fragmetns), True)
            for idx, row in filtered_fragmetns.iterrows():
                filneame_bad_frames = bad_images.get(row['Subject'], {}).get(row['Filename'], set())
                fragment_onset = int(row['Onset'])
                fragment_offset = int(row['Offset'])
                for bad_frame in filneame_bad_frames:
                    if (fragment_onset < bad_frame) and (bad_frame < fragment_offset):
                        valid_rows[idx] = False
                        print(row['folder_name'], fragment_offset, bad_frame)
                        break
            
            filtered_fragmetns = filtered_fragmetns[valid_rows]
    
    return filtered_fragmetns

def prepare_microexpression_datasets(
    data_root: str = 'data/augmented/casme3',
    labels_path: str = 'data/augmented/casme3/labels.xlsx',
    subject_independent: bool = True,
    include_augmented: bool = False,
    seed: int = 1,
    target_col: str = 'Objective class',
    k_folds: int = 1,
    current_fold: int = 0,
    max_train_samples: int | None = None
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
    fragments = df_valid[['folder_name', 'Subject', 'Filename', 'Onset', 'Offset', target_col]].drop_duplicates()
    fragments[target_col] = fragments[target_col].astype(str).str.lower()
    
    fragments = filter_bad_fragments(fragments, data_root)
    
    print(f"Using {k_folds}-fold CV, current fold: {current_fold}")
    if subject_independent:
        splitter = GroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        split = splitter.split(fragments, groups=fragments['Subject'])
    else:
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        split = splitter.split(fragments, fragments[target_col])
    
    for fold_idx, (train_idx, val_idx) in enumerate(split):
        if fold_idx == current_fold:
            train_fragments = fragments.iloc[train_idx]
            val_fragments = fragments.iloc[val_idx]
            break

    print(f"  Fold {current_fold}: Train fragments: {len(train_fragments)}, Val fragments: {len(val_fragments)}")
    print("Target distribution train:", train_fragments[target_col].value_counts(normalize=True).to_dict())
    print("Target distribution val:", val_fragments[target_col].value_counts(normalize=True).to_dict())
    
    if max_train_samples is not None and max_train_samples > 0 and len(train_fragments) > max_train_samples:
        print(f"Original fragmets num limited from {len(train_fragments)} to {max_train_samples}")
        sampled_fragments, _ = train_test_split(
            train_fragments,
            train_size=max_train_samples,
            stratify=train_fragments[target_col],
            random_state=seed
        )
        train_fragments = sampled_fragments.reset_index(drop=True)
        print(f"  Fold {current_fold}: Train fragments: {len(train_fragments)}, Val fragments: {len(val_fragments)}")
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

        return landmarks_list, label, num_frames, csv_path
    

def me_collate_fn(batch):
    landmarks_seqs = []
    labels = []
    lengths = []
    csv_paths = []

    for landmarks_list, label, num_frames, batch_csv_paths in batch:
        landmarks_seqs.append(landmarks_list)
        labels.append(label)
        lengths.append(num_frames)
        csv_paths.append(batch_csv_paths)

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return landmarks_seqs, labels, lengths, csv_paths


def _build_dataloaders(
    data_root, labels_path, subject_independent,
    include_augmented, seed, target_col, batch_size, drop_others, remap_classes,
    k_folds, current_fold, max_train_samples
):
    train_list, val_list, label_map = prepare_microexpression_datasets(
        data_root=data_root,
        labels_path=labels_path,
        subject_independent=subject_independent,
        include_augmented=include_augmented,
        seed=seed,
        target_col=target_col,
        k_folds=k_folds,
        current_fold=current_fold,
        max_train_samples=max_train_samples
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
    rnn_cfg: RNNConfig,
    temporal_model: str = 'transformer'
):
    # ----- GNN -----
    gnn = FacialGNN(
        hidden_dims=gnn_cfg.hidden_dims,
        fusion_dim=gnn_cfg.fusion_dim,
        dropout=gnn_cfg.dropout,
        pool=gnn_cfg.pool
    )
    
    gnn_output_dim = (
        gnn_cfg.fusion_dim
        if gnn_cfg.fusion_dim is not None
        else gnn_cfg.hidden_dims[-1]
    )

    # ----- Temporal module -----
    if temporal_model == 'transformer':
        assert isinstance(temporal_cfg, TransformerConfig)
        temporal_module = FacialTemporalTransformer(
            input_dim=gnn_output_dim,
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

        temporal_module = MILPooling(
            frame_dim=gnn_output_dim,
            d_model=temporal_cfg.d_model,
            kernel_size=temporal_cfg.kernel_size,
            num_conv_layers=temporal_cfg.num_conv_layers,
            dropout=temporal_cfg.dropout,
            temperature=temporal_cfg.temperature,
            use_residual=temporal_cfg.use_residual
        )
        embed_dim = temporal_cfg.d_model
    elif temporal_model in ['gru', 'lstm']:
        assert isinstance(rnn_cfg, RNNConfig), "For GRU/LSTM use rnn_cfg"
        
        temporal_module = FacialRNN(
            input_size=gnn_output_dim,
            hidden_size=rnn_cfg.hidden_size,
            num_layers=rnn_cfg.num_layers,
            dropout=rnn_cfg.dropout,
            bidirectional=rnn_cfg.bidirectional,
            rnn_type=temporal_model
        )
        embed_dim = temporal_module.output_dim
        
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

    for lm_list, labels, num_frames, _ in loader:
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

        if warmup_scheduler:
            with warmup_scheduler.dampening():
                if lr_scheduler and training_cfg.scheduler in ['cosine', 'cosine-restarts']:
                    lr_scheduler.step()
        else:
            if lr_scheduler and training_cfg.scheduler in ['cosine', 'cosine-restarts']:
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
    val_csv_paths = [] if return_extras else None
    global_max_len = 0

    with torch.no_grad():
        for lm_list, labels, lengths, csv_paths in loader:
            landmarks_seqs = [[frame.to(device) for frame in seq] for seq in lm_list]
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
                batch_max_len = lengths.max().item()
                global_max_len = max(global_max_len, batch_max_len)
                val_scores.append(scores.detach().cpu())
                val_attn_weights.append(attn_weights.detach().cpu())
                val_lengths.append(lengths.cpu())
                val_csv_paths.append(np.array(csv_paths))

    val_loss /= len(loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    if return_extras:
        padded_scores = []
        padded_attn = []
        for scores, attn in zip(val_scores, val_attn_weights):
            batch_size, current_len = scores.shape
            if current_len < global_max_len:
                pad_width = global_max_len - current_len
                scores_padded = torch.nn.functional.pad(scores, (0, pad_width), value=float('-inf'))
                attn_padded = torch.nn.functional.pad(attn, (0, pad_width), value=0.0)
            else:
                scores_padded = scores
                attn_padded = attn
            
            padded_scores.append(scores_padded)
            padded_attn.append(attn_padded)

        val_scores = torch.cat(padded_scores, dim=0).numpy()           # [N_val, global_max_len]
        val_attn_weights = torch.cat(padded_attn, dim=0).numpy()       # [N_val, global_max_len]
        val_lengths = torch.cat(val_lengths, dim=0).numpy()            # [N_val]
        val_csv_paths = np.concatenate(val_csv_paths, axis=0)          # [N_val]

        return val_loss, val_acc, val_f1, np.array(val_labels), np.array(val_preds), \
               val_scores, val_attn_weights, val_lengths, val_csv_paths
    else:
        return val_loss, val_acc, val_f1, np.array(val_labels), np.array(val_preds)

def train(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    training_cfg: TrainingConfig = TrainingConfig(),
    gnn_cfg: GNNConfig = GNNConfig(),
    transformer_cfg: TransformerConfig = TransformerConfig(),
    mil_cfg: MILConfig = MILConfig(),
    rnn_cfg: RNNConfig = RNNConfig(),
    temporal_model: str = 'transformer',
    debug: bool = False,
    is_cv_fold: bool = False,
    fold_idx: int = -1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using temporal model: {temporal_model}")

    (train_loader, val_loader, label_map, weights, train_samples, val_samples,
     class_mapping, train_class_counts, val_class_counts) = _build_dataloaders(
        data_root=dataset_cfg.data_root,
        labels_path=dataset_cfg.labels_path,
        subject_independent=dataset_cfg.subject_independent,
        include_augmented=dataset_cfg.include_augmented,
        seed=dataset_cfg.seed,
        target_col=dataset_cfg.target_col,
        batch_size=training_cfg.batch_size,
        drop_others=dataset_cfg.drop_others,
        remap_classes=dataset_cfg.remap_classes,
        k_folds=dataset_cfg.k_folds,
        current_fold=dataset_cfg.current_fold,
        max_train_samples=dataset_cfg.max_train_samples
    )
    weights = weights.to(device)
    num_classes = len(label_map)

    temporal_cfg = transformer_cfg if temporal_model == 'transformer' else mil_cfg
    model = _build_model(
        num_classes=num_classes,
        gnn_cfg=gnn_cfg,
        temporal_cfg=temporal_cfg,
        rnn_cfg=rnn_cfg,
        temporal_model=temporal_model
    ).to(device)

    # 3. Optimizer & co
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=training_cfg.label_smoothing)
    # criterion = FocalLoss(gamma=2.5, alpha=weights, task_type='multi-class', num_classes=len(weights))
    optimizer = AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

    lr_scheduler = None
    warmup_scheduler = None

    total_batches = len(train_loader) * training_cfg.num_epochs
    if training_cfg.scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_batches)
    elif training_cfg.scheduler == 'cosine-restarts':
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20 * len(train_loader),
            T_mult=2,
            eta_min=1e-6
        )
    elif training_cfg.scheduler == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_cfg.scheduler_factor,
            patience=training_cfg.scheduler_patience,
        )
    else:
        raise ValueError(f"Unknown scheduler: {training_cfg.scheduler}")
    
    if training_cfg.use_warmup:
        warmup_period = len(train_loader) * training_cfg.warmup_epochs
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

    mlflow.set_experiment('ME_model_training')
    run_name = f"Fold_{fold_idx}" if is_cv_fold else "Single_Run"
    with mlflow.start_run(run_name=run_name, nested=is_cv_fold) as run:
        run_id = run.info.run_id

        params_to_log = {
            "temporal_model": temporal_model,
            "num_classes": num_classes,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "classes": label_map,
            "class_mapping": class_mapping,
            "class_weights": weights,
            "loss": "FocalLoss" if isinstance(criterion, FocalLoss) else "CrossEntropyLoss",
        }
        if isinstance(criterion, FocalLoss):
            params_to_log["focal_loss_gamma"] = criterion.gamma

        if is_cv_fold:
            params_to_log["fold"] = fold_idx
            params_to_log["mode"] = "cv_fold"
        else:
            params_to_log.update({
                "dataset_cfg": dataset_cfg,
                "training_cfg": training_cfg,
                "gnn_cfg": gnn_cfg,
            })

            if temporal_model == 'transformer':
                params_to_log["temporal_cfg"] = transformer_cfg
            elif temporal_model == 'mil':
                params_to_log["temporal_cfg"] = mil_cfg
            else:
                params_to_log["temporal_cfg"] = rnn_cfg

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
        
        best_f1_after_warmup = 0.0
        best_labels_after_warmup = None
        best_preds_after_warmup = None
        last_f1 = 0.0
        last_labels = None
        last_preds = None

        for epoch in range(training_cfg.num_epochs):
            train_loss, train_acc, train_f1 = _train_epoch(
                model=model, 
                loader=train_loader, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=device, 
                grad_clip_norm=training_cfg.grad_clip_max_norm, 
                lr_scheduler=lr_scheduler,
                warmup_scheduler=warmup_scheduler
            )

            val_outputs = _validate_epoch(
                model, val_loader, criterion, device, return_extras=debug
            )

            if debug:
                val_loss, val_acc, val_f1, val_labels, val_preds, \
                val_scores, val_attn, val_lengths, val_csv_paths = val_outputs
            else:
                val_loss, val_acc, val_f1, val_labels, val_preds = val_outputs

            mlflow.log_metrics({
                f"train_loss": train_loss,
                f"train_acc": train_acc,
                f"train_f1": train_f1,
                f"val_loss": val_loss,
                f"val_acc": val_acc,
                f"val_f1": val_f1,
                f"lr": optimizer.param_groups[0]['lr']
            }, step=epoch)

            if training_cfg.scheduler == 'plateau':
                if warmup_scheduler:
                    with warmup_scheduler.dampening():
                        lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step(val_loss)
            elif training_cfg.scheduler in ['cosine', 'cosine-restarts']:
                pass

            last_f1 = val_f1
            last_labels = val_labels.copy()
            last_preds = val_preds.copy()
            
            if (epoch >= (training_cfg.num_epochs // 2)) and (val_f1 > best_f1_after_warmup):
                best_f1_after_warmup = val_f1
                best_labels_after_warmup = val_labels.copy()
                best_preds_after_warmup = val_preds.copy()

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

            if (epoch % 5 == 0) and debug:
                prefix = f"epoch_{epoch}"
                np.save(save_dir / f"{prefix}_val_scores.npy", val_scores)
                np.save(save_dir / f"{prefix}_val_attn_weights.npy", val_attn)
                np.save(save_dir / f"{prefix}_val_lengths.npy", val_lengths)
                np.save(save_dir / f"{prefix}_val_labels.npy", val_labels)
                np.save(save_dir / f"{prefix}_val_preds.npy", val_preds)
                val_csv_paths_file = Path(save_dir / f"val_csv_paths.npy")
                if not val_csv_paths_file.exists():
                    np.save(val_csv_paths_file, val_csv_paths)
                    mlflow.log_artifact(str(val_csv_paths_file))

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
            
        mlflow.log_metrics({
            f"best_f1_after_warmup": best_f1_after_warmup,
            f"last_f1": last_f1,
        })

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        cm_path = save_dir / "confusion_matrix.txt"
        np.savetxt(cm_path, cm, fmt="%d")
        mlflow.log_artifact(str(cm_path))
        
        return (
            best_f1_after_warmup, best_labels_after_warmup, best_preds_after_warmup,
            last_f1, last_labels, last_preds
        )

if __name__ == "__main__":
    parser = create_training_parser()
    args = parser.parse_args()
    dataset_cfg, training_cfg, gnn_cfg, transformer_cfg, mil_cfg, rnn_cfg = create_train_configs(args)

    if args.k_folds > 1 and args.current_fold is None:
        print(f"Starting {args.k_folds}-fold cross-validation")

        mlflow.set_experiment('ME_model_training')
        with mlflow.start_run(run_name="CV_Parent") as parent_run:
            cv_id = parent_run.info.run_id
            mlflow.set_tag("cv_id", cv_id)
            mlflow.set_tag("mode", "cross_validation")
            
            temporal_cfg = transformer_cfg
            if args.temporal_model == 'mil':
                temporal_cfg = mil_cfg
            else:
                temporal_cfg = rnn_cfg

            mlflow.log_params({
                "k_folds": args.k_folds,
                "temporal_model": args.temporal_model,
                "seed": args.seed,
                "dataset_cfg": dataset_cfg,
                "training_cfg": training_cfg,
                "gnn_cfg": gnn_cfg,
                "temporal_cfg": temporal_cfg
            })

            all_best_labels = []
            all_best_preds = []
            all_last_labels = []
            all_last_preds = []
            fold_best_f1s = []
            fold_last_f1s = []

            for fold in range(args.k_folds):
                print(f"\nFold {fold}/{args.k_folds-1}")
                dataset_cfg.current_fold = fold

                (
                    fold_best_f1, fold_best_labels, fold_best_preds,
                    fold_last_f1, fold_last_labels, fold_last_preds
                ) = train(
                    dataset_cfg=dataset_cfg,
                    training_cfg=training_cfg,
                    gnn_cfg=gnn_cfg,
                    transformer_cfg=transformer_cfg,
                    mil_cfg=mil_cfg,
                    rnn_cfg=rnn_cfg,
                    temporal_model=args.temporal_model,
                    debug=args.debug,
                    is_cv_fold=True,
                    fold_idx=fold
                )

                all_best_labels.extend(fold_best_labels)
                all_best_preds.extend(fold_best_preds)
                all_last_labels.extend(fold_last_labels)
                all_last_preds.extend(fold_last_preds)

                fold_best_f1s.append(fold_best_f1)
                fold_last_f1s.append(fold_last_f1)

            global_best_f1 = f1_score(all_best_labels, all_best_preds, average='macro')
            global_best_acc = accuracy_score(all_best_labels, all_best_preds)
            global_last_f1 = f1_score(all_last_labels, all_last_preds, average='macro')
            global_last_acc = accuracy_score(all_last_labels, all_last_preds)

            mlflow.log_metrics({
                "cv_global_best_f1": global_best_f1,
                "cv_global_best_acc": global_best_acc,
                "cv_global_last_f1": global_last_f1,
                "cv_global_last_acc": global_last_acc,
                "cv_mean_best_f1": np.mean(fold_best_f1s),
                "cv_std_best_f1": np.std(fold_best_f1s),
                "cv_mean_last_f1": np.mean(fold_last_f1s),
                "cv_std_last_f1": np.std(fold_last_f1s),
            })
    else:
        if args.k_folds > 1:
            print(f"Training only fold {dataset_cfg.current_fold} / {args.k_folds-1}")
        else:
            print("Training single split (no cross-validation)")
        
        train(
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            gnn_cfg=gnn_cfg,
            transformer_cfg=transformer_cfg,
            mil_cfg=mil_cfg,
            rnn_cfg=rnn_cfg,
            temporal_model=args.temporal_model,
            debug=args.debug,
            is_cv_fold=False,
            fold_idx=-1
        )