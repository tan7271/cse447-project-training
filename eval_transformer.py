#!/usr/bin/env python
"""
Evaluate the character-level Transformer LM on the test set.

Follows transformer_training_cse447_project.ipynb evaluation:
- Loads model (state_dict from train_transformer.py output, or full checkpoint from notebook)
- Uses same train/val/test split as train_transformer.py for comparable test set
- Reports: test loss, perplexity, top-1 and top-3 accuracy, optional per-language breakdown

Usage:
  python eval_transformer.py --checkpoint work/char-transformer/char_transformer.pt --data data/train/preprocessed_data-2.pkl
  python eval_transformer.py --checkpoint path/to/best.pt  # full checkpoint with model_state + arch
"""
import argparse
import math
import pickle
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from train_transformer import (
    SEED,
    load_preprocessed,
    CharDatasetFull,
    CharTransformer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "train" / "preprocessed_data_5M.pkl"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "work" / "char-transformer-10M" / "char_transformer_epoch9.pt"


def load_vocab_from_path_or_data(vocab_path: Path | None, data_path: Path):
    """
    Load vocab from:
      - vocab_path, if provided:
          * if it looks like a preprocessed_data.pkl (dict with 'pairs' and 'vocab'),
            return obj['vocab']
          * otherwise assume it's already a vocab dict
      - otherwise, from data_path via load_preprocessed.
    """
    if vocab_path is not None and vocab_path.exists():
        with open(vocab_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "vocab" in obj and "pairs" in obj:
            return obj["vocab"]
        return obj  # assume it's already a vocab dict

    # Fallback: vocab from the eval data file
    _, vocab = load_preprocessed(data_path)
    return vocab


def load_model_and_vocab(checkpoint_path: Path, vocab_path: Path, data_path: Path, max_len: int):
    """Load vocab from checkpoint, vocab_path, or data pkl; load model (state_dict or full ckpt)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Full checkpoint: model_state present
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # Notebook BEST save: vocab + max_len (no arch keys)
        if "vocab" in ckpt:
            vocab = ckpt["vocab"]
            ckpt_max_len = ckpt["max_len"]
            model = CharTransformer(
                vocab_size=vocab["vocab_size"],
                pad_id=vocab["char2idx"]["<PAD>"],
                d_model=256,
                nhead=8,
                num_layers=6,
                max_len=ckpt_max_len - 1,
                dropout=0.1,
            )
        else:
            # Training ckpt: vocab_size, pad_id, d_model, nhead, num_layers, model_max_len
            vocab = None
            model = CharTransformer(
                vocab_size=ckpt["vocab_size"],
                pad_id=ckpt["pad_id"],
                d_model=ckpt["d_model"],
                nhead=ckpt["nhead"],
                num_layers=ckpt["num_layers"],
                max_len=ckpt["model_max_len"],
                dropout=ckpt.get("dropout", 0.1),
            )
        model.load_state_dict(ckpt["model_state"], strict=True)
        if vocab is None:
            vocab = load_vocab_from_path_or_data(vocab_path, data_path)
        return model, vocab

    # State_dict only (train_transformer.py output)
    vocab = load_vocab_from_path_or_data(vocab_path, data_path)

    # Infer vocab size from checkpoint embed weights and sanity check
    ckpt_vocab_size = None
    if isinstance(ckpt, dict):
        for name, tensor in ckpt.items():
            if isinstance(tensor, torch.Tensor) and name.endswith("embed.weight") and tensor.ndim == 2:
                ckpt_vocab_size = tensor.shape[0]
                break

    if ckpt_vocab_size is not None and vocab["vocab_size"] != ckpt_vocab_size:
        raise SystemExit(
            "Vocab size mismatch between checkpoint and provided vocab.\n"
            f"  Checkpoint expects vocab_size={ckpt_vocab_size}, "
            f"but vocab has vocab_size={vocab['vocab_size']}.\n"
            "  To evaluate this checkpoint, pass --vocab pointing to either:\n"
            "    (a) the original vocab.pkl saved with training, or\n"
            "    (b) the original preprocessed_data*.pkl used for training.\n"
        )

    pad_id = vocab["char2idx"]["<PAD>"]
    model = CharTransformer(
        vocab_size=vocab["vocab_size"],
        pad_id=pad_id,
        d_model=256,
        nhead=8,
        num_layers=6,
        max_len=max_len - 1,
        dropout=0.1,
    )
    model.load_state_dict(ckpt, strict=True)
    return model, vocab


def get_test_split(pairs, max_train_examples: int):
    """Same split as train_transformer.py: val/test each 10% of train size."""
    random.seed(SEED)
    random.shuffle(pairs)
    total_count = len(pairs)
    uncapped_train = int(total_count / 1.2)
    train_size = (
        min(uncapped_train, max_train_examples)
        if max_train_examples > 0
        else uncapped_train
    )
    val_size = int(0.1 * train_size)
    test_size = int(0.1 * train_size)
    test_pairs = pairs[
        train_size + val_size : train_size + val_size + test_size
    ]
    return test_pairs


def evaluate(
    model,
    test_pairs,
    vocab,
    max_len: int = 256,
    batch_size: int = 256,
    device=None,
    per_lang: bool = True,
):
    """Compute test loss, perplexity, top-1/top-3 accuracy; optional per-language breakdown."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pad_id = vocab["char2idx"]["<PAD>"]
    test_ds = CharDatasetFull(test_pairs, vocab["char2idx"], max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top3_correct = 0

    lang_loss = defaultdict(float)
    lang_tokens = defaultdict(int)
    lang_top1 = defaultdict(int)
    lang_top3 = defaultdict(int)

    pair_idx = 0
    has_lang = test_pairs and isinstance(test_pairs[0], dict) and "lang" in test_pairs[0]

    with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            mask = y != pad_id
            tokens = mask.sum().item()
            total_loss += loss.item()
            total_tokens += tokens

            top3_preds = logits.topk(3, dim=-1).indices
            top1_hit = (top3_preds[:, :, 0] == y) & mask
            top3_hit = (top3_preds == y.unsqueeze(-1)).any(dim=-1) & mask

            top1_correct += top1_hit.sum().item()
            top3_correct += top3_hit.sum().item()

            if per_lang and has_lang:
                for i in range(B):
                    if pair_idx + i < len(test_pairs):
                        lang = test_pairs[pair_idx + i]["lang"]
                        m = mask[i].sum().item()
                        lang_tokens[lang] += m
                        lang_loss[lang] += criterion(
                            logits[i][mask[i]], y[i][mask[i]]
                        ).item()
                        lang_top1[lang] += top1_hit[i].sum().item()
                        lang_top3[lang] += top3_hit[i].sum().item()

            pair_idx += B

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Test pairs:    {len(test_pairs):,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Avg Loss:      {avg_loss:.4f}")
    print(f"  Perplexity:    {perplexity:.2f}")
    print(f"  Top-1 Acc:     {top1_correct/total_tokens:.4f} ({100*top1_correct/total_tokens:.2f}%)")
    print(f"  Top-3 Acc:     {top3_correct/total_tokens:.4f} ({100*top3_correct/total_tokens:.2f}%)")

    if per_lang and has_lang and lang_tokens:
        print("\n" + "=" * 60)
        print("PER-LANGUAGE BREAKDOWN (top 15)")
        print("=" * 60)
        print(f"{'Lang':<6} {'Tokens':>8} {'Loss':>8} {'PPL':>8} {'Top1':>8} {'Top3':>8}")
        print("-" * 50)
        sorted_langs = sorted(
            lang_tokens.keys(),
            key=lambda l: lang_tokens[l],
            reverse=True,
        )
        for lang in sorted_langs[:15]:
            t = lang_tokens[lang]
            l = lang_loss[lang] / t
            p = math.exp(l)
            t1 = lang_top1[lang] / t
            t3 = lang_top3[lang] / t
            print(f"{lang:<6} {t:>8,} {l:>8.4f} {p:>8.2f} {t1:>7.2%} {t3:>7.2%}")

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate character-level Transformer LM on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to model .pt (state_dict or full checkpoint)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to preprocessed_data.pkl (for vocab if no vocab.pkl, and for test split)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Path to vocab.pkl or preprocessed_data.pkl used for TRAINING "
        "(default: checkpoint_dir/vocab.pkl or vocab from --data)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Max sequence length (must match training)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=10_000_000,
        help="Must match value used in train_transformer.py for same test set (0 = no cap)",
    )
    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=100_000,
        help="Max test examples to evaluate on (0 = use full test split)",
    )
    parser.add_argument(
        "--no-per-lang",
        action="store_true",
        help="Skip per-language breakdown",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")

    vocab_path = args.vocab or (args.checkpoint.parent / "vocab.pkl")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data}")
    print(f"Max len:    {args.max_len}")

    pairs, _ = load_preprocessed(args.data)
    test_pairs = get_test_split(pairs, args.max_train_examples)
    if args.max_test_examples > 0 and len(test_pairs) > args.max_test_examples:
        test_pairs = test_pairs[: args.max_test_examples]
        print(f"Test pairs: {len(test_pairs):,} (capped at --max-test-examples)")
    else:
        print(f"Test pairs: {len(test_pairs):,}")

    model, vocab = load_model_and_vocab(
        args.checkpoint, vocab_path, args.data, args.max_len
    )
    print("Model loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        model,
        test_pairs,
        vocab,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=device,
        per_lang=not args.no_per_lang,
    )


if __name__ == "__main__":
    main()
