import argparse
from argparse import Namespace
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    data_root: str = 'data/augmented/casme3'
    labels_path: str = 'data/augmented/casme3/labels.xlsx'
    subject_independent: bool = True
    online_aug: bool = False
    aug_num: int = 0
    seed: int = 1
    target_col: str = 'Objective class'
    drop_others: bool = True
    remap_classes: bool = False
    k_folds: int = 1
    current_fold: int = 0
    max_train_samples: int | None = None

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
    use_warmup: bool = False
    warmup_epochs: int = 8
    label_smoothing: float = 0.0

@dataclass
class GNNConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    fusion_dim: int | None = None
    dropout: float = 0.2
    pool: str = 'mean'

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

@dataclass
class RNNConfig:
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    
def create_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MicroExpression model")

    # ==================== DatasetConfig ====================
    parser.add_argument('--data_root', type=str, default='data/augmented/casme3_spotting')
    parser.add_argument('--labels_path', type=str, default='data/augmented/casme3_spotting/labels.xlsx')
    parser.add_argument('--subject_independent', action='store_true', default=True)
    parser.add_argument('--no_subject_independent', action='store_false', dest='subject_independent')
    parser.add_argument('--online_aug', action='store_true', default=False, help="Enable online augmentation (noise, rotate, scale)")
    parser.add_argument('--aug_num', type=int, default=0, help="Number of augment fragmetns for 1 original")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--target_col', type=str, default='ME')
    parser.add_argument('--drop_others', action='store_true', default=True, help="Drop samples with 'others' class after remapping to 3-class scheme")
    parser.add_argument('--no_drop_others', action='store_false', dest='drop_others')
    parser.add_argument('--remap_classes', action='store_true', default=False, help="Remap classes to positive, negative, surprise")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--current_fold', type=int, default=None, help="If set — train only this fold. If None and k_folds > 1 — train all folds sequentially")
    parser.add_argument('--max_train_samples', type=int, default=None, help="Max samples in train (None — all train).")

    # ==================== TrainingConfig ====================
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience_early_stop', type=int, default=40)
    parser.add_argument('--grad_clip_max_norm', type=float, default=5.0)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau', 'cosine-restarts'], default='cosine')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="For plateau scheduler")
    parser.add_argument('--scheduler_patience', type=int, default=7, help="For plateau scheduler")
    parser.add_argument('--use_warmup', action='store_true', default=False)
    parser.add_argument('--warmup_epochs', type=int, default=8)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # ==================== Debug options ====================
    parser.add_argument('--debug', action='store_true', default=False, help="If set, collect and log MIL attention scores/weights on validation")

    # ==================== GNNConfig ====================
    parser.add_argument('--gnn_hidden_dims', type=int, nargs='+', default=[64, 128, 256], help="Example: --gnn_hidden_dims 32 64 128")
    parser.add_argument('--gnn_fusion_dim', type=int, default=256, help="Use negative value (e.g. -1) to set None")
    parser.add_argument('--gnn_dropout', type=float, default=0.3)
    parser.add_argument('--gnn_pool', type=str, choices=['mean', 'max'], default='mean')

    # ==================== Temporal Model Selection ====================
    parser.add_argument('--temporal_model', type=str, choices=['transformer', 'mil', 'gru', 'lstm'], 
                    default='transformer', help="Choose temporal model: transformer, mil, gru or lstm")
    
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
    
    # ==================== RNNConfig ====================
    parser.add_argument("--rnn_hidden_size", type=int, default=256, help="Hidden size for RNN")
    parser.add_argument("--rnn_num_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--rnn_dropout", type=float, default=0.3, help="Dropout for RNN")
    parser.add_argument("--rnn_bidirectional", action='store_true', default=False, help="Use bidirectional RNN")
    
    return parser

def create_train_configs(args: Namespace):
    dataset_cfg = DatasetConfig(
        data_root=args.data_root,
        labels_path=args.labels_path,
        subject_independent=args.subject_independent,
        online_aug=args.online_aug,
        aug_num=args.aug_num,
        seed=args.seed,
        target_col=args.target_col,
        drop_others=args.drop_others,
        remap_classes=args.remap_classes,
        k_folds=args.k_folds,
        current_fold=args.current_fold if args.current_fold is not None else 0,
        max_train_samples=args.max_train_samples
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
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing
    )

    gnn_fusion_dim = args.gnn_fusion_dim
    if (args.gnn_fusion_dim is None) or (args.gnn_fusion_dim < 0):
        gnn_fusion_dim = None 
    gnn_cfg = GNNConfig(
        hidden_dims=args.gnn_hidden_dims,
        fusion_dim=gnn_fusion_dim,
        dropout=args.gnn_dropout,
        pool=args.gnn_pool
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
    
    rnn_cfg = RNNConfig(
        hidden_size=args.rnn_hidden_size,
        num_layers=args.rnn_num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidirectional
    )
    
    return dataset_cfg, training_cfg, gnn_cfg, transformer_cfg, mil_cfg, rnn_cfg