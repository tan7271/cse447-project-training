#!/usr/bin/env python
"""
Train a character-level Transformer language model on preprocessed_data.pkl.

This mirrors the "training" section of training_generation_cse447_project.ipynb:
- Uses CharDatasetFull (RIGHT-padding, autoregressive LM over x+y)
- Uses CharTransformer (TransformerEncoder with explicit causal mask)
- Trains with mixed-precision (AMP) and GradScaler

Input:
  - A preprocessed pickle created by training_generation_cse447_project.run_pipeline,
    containing:
      data["pairs"]: list of dicts with at least "x" and "y" fields
      data["vocab"]: dict with "char2idx", "idx2char", "vocab_size"

Output:
  - A directory with:
      char_transformer.pt  (model state_dict)
      vocab.pkl            (vocab dict for decoding / inference)
"""

import argparse
import math
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "train" / "preprocessed_data_10M.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "work" / "char-transformer-10M"


def load_preprocessed(path: Path):
    """Load preprocessed pairs and vocab from pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    pairs = data["pairs"]
    vocab = data["vocab"]
    print(f"Loaded {len(pairs):,} pairs and vocabulary of size {vocab['vocab_size']}")
    return pairs, vocab


class CharDatasetFull(Dataset):
    """
    Dataset for character-level autoregressive LM.
    For each pair, we build full = (x + y)[-max_len:], then train on
    all next-character positions within that window.
    """

    def __init__(self, pairs, char2idx, max_len: int = 64):
        self.pairs = pairs
        self.char2idx = char2idx
        self.pad = char2idx["<PAD>"]
        self.unk = char2idx["<UNK>"]
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        # Expect dict with "x" and "y" fields from run_pipeline
        x = p.get("x") if isinstance(p, dict) else p[0]
        y = p.get("y") if isinstance(p, dict) else p[1]
        full = (x + y)[-self.max_len :]  # keep last max_len chars
        enc = [self.char2idx.get(c, self.unk) for c in full]

        # RIGHT pad
        if len(enc) < self.max_len:
            enc = enc + [self.pad] * (self.max_len - len(enc))

        # Autoregressive LM: input is all but last, target is all but first
        x_ids = torch.tensor(enc[:-1], dtype=torch.long)  # (max_len-1,)
        y_ids = torch.tensor(enc[1:], dtype=torch.long)   # (max_len-1,)
        return x_ids, y_ids


class CharTransformer(nn.Module):
    """
    Single-stack Transformer encoder with causal mask for next-character LM.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        max_len: int = 63,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

        # BOOL causal mask (T,T): True = disallow attending
        causal = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        B, T = x.shape
        pad_mask = x == self.pad_id  # bool (B,T)

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) * math.sqrt(self.d_model) + self.pos_embed(pos)
        h = self.drop(h)

        attn_mask = self.causal_mask[:T, :T]  # bool (T,T)
        h = self.transformer(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.head(h)  # (B,T,vocab_size)


def train_with_validation(
    train_pairs,
    val_pairs,
    vocab,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1.2e-4,
    max_len: int = 128,
    log_steps: int = 100,
    output_dir: Optional[Path] = None,
    wandb_run=None,
):
    """
    Train CharTransformer with train/val split, tracking loss per token.

    If wandb_run is provided, logs training/validation loss and percent done.
    If output_dir is provided, saves a checkpoint at the end of every epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pad_id = vocab["char2idx"]["<PAD>"]

    train_ds = CharDatasetFull(train_pairs, vocab["char2idx"], max_len)
    val_ds = CharDatasetFull(val_pairs, vocab["char2idx"], max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CharTransformer(
        vocab_size=vocab["vocab_size"],
        pad_id=pad_id,
        d_model=256,
        nhead=8,
        num_layers=6,
        max_len=max_len - 1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "val_loss": []}

    num_steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * num_steps_per_epoch
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=True,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )

            # Count only non-pad positions as tokens
            mask = y != pad_id
            tokens = mask.sum().item()

            scaler.scale(loss / tokens).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_tokens += tokens
            pbar.set_postfix(loss=f"{total_loss/total_tokens:.4f}")

            global_step += 1
            if (
                wandb_run is not None
                and log_steps > 0
                and (global_step % log_steps == 0 or global_step == 1)
            ):
                step_loss = loss.item() / max(tokens, 1)
                percent_done = (
                    100.0 * global_step / total_steps if total_steps > 0 else 0.0
                )
                wandb_run.log(
                    {
                        "train/loss_step": step_loss,
                        "train/loss_running": total_loss / max(total_tokens, 1),
                        "train/percent_done": percent_done,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                mask = y != pad_id
                val_loss += loss.item()
                val_tokens += mask.sum().item()

        train_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        val_loss_norm = val_loss / val_tokens if val_tokens > 0 else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_norm)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss_norm:.4f}"
        )

        if wandb_run is not None:
            epoch_percent_done = 100.0 * (epoch + 1) / max(epochs, 1)
            wandb_run.log(
                {
                    "train/loss_epoch": train_loss,
                    "val/loss_epoch": val_loss_norm,
                    "epoch": epoch + 1,
                    "train/percent_done_epoch": epoch_percent_done,
                },
                step=global_step,
            )

        if output_dir is not None:
            # Save a rich checkpoint with model + vocab + arch so it can be
            # evaluated \"out of the box\" without separate metadata.
            ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "max_len": max_len,
                "epoch": epoch + 1,
                "vocab_size": vocab["vocab_size"],
                "pad_id": pad_id,
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "model_max_len": max_len - 1,
                "dropout": 0.1,
            }
            epoch_path = output_dir / f"char_transformer_epoch{epoch+1}.pt"
            latest_path = output_dir / "char_transformer.pt"
            torch.save(ckpt, epoch_path)
            torch.save(ckpt, latest_path)

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train a character-level Transformer LM on preprocessed_data.pkl"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to preprocessed_data.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save trained Transformer model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Max total sequence length (x+y window) used in CharDatasetFull",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log metrics to Weights & Biases (requires `pip install wandb` and `wandb login`)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cse447-char-transformer",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=100,
        help="Log training metrics to W&B every N training steps",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=10_000_000,
        help="Max training examples to use (shuffle then take first N). 0 = no cap",
    )
    args = parser.parse_args()

    data_path = args.data
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using seed: {SEED}")
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")

    # Load pairs and vocab
    pairs, vocab = load_preprocessed(data_path)

    # Shuffle and split: val and test each = 10% of train size (train + 0.2*train used)
    random.seed(SEED)
    random.shuffle(pairs)
    total_count = len(pairs)
    uncapped_train = int(total_count / 1.2)
    train_size = (
        min(uncapped_train, args.max_train_examples)
        if args.max_train_examples > 0
        else uncapped_train
    )
    val_size = int(0.1 * train_size)
    test_size = int(0.1 * train_size)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size : train_size + val_size]
    test_pairs = pairs[train_size + val_size : train_size + val_size + test_size]

    used = train_size + val_size + test_size
    if used < total_count:
        print(f"Using first {used:,} pairs (val/test = 10% of train each; {total_count - used:,} unused)")

    print(f"Total pairs: {total_count:,}")
    print(f"Train pairs: {len(train_pairs):,} (val/test = 10% of train)")
    print(f"Val pairs:   {len(val_pairs):,}")
    print(f"Test pairs:  {len(test_pairs):,}")

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise SystemExit(
                "wandb is not installed. Install it with `pip install wandb`."
            )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_len": args.max_len,
                "train_size": len(train_pairs),
                "val_size": len(val_pairs),
            },
        )

    # Train model (also saves checkpoints each epoch if output_dir is set)
    model, history = train_with_validation(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        vocab=vocab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        log_steps=args.log_steps,
        output_dir=output_dir,
        wandb_run=wandb_run,
    )

    # Save final rich checkpoint and separate vocab file for convenience
    model_path = output_dir / "char_transformer.pt"
    vocab_path = output_dir / "vocab.pkl"
    final_ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "max_len": args.max_len,
        "epoch": args.epochs,
        "vocab_size": vocab["vocab_size"],
        "pad_id": vocab["char2idx"]["<PAD>"],
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "model_max_len": args.max_len - 1,
        "dropout": 0.1,
    }
    torch.save(final_ckpt, model_path)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nModel checkpoint saved to: {model_path}")
    print(f"Vocab saved to:           {vocab_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

