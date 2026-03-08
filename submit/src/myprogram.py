#!/usr/bin/env python
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


class MyModel:
    """
    Wrapper around a fine-tuned ByT5 model for next-character prediction.

    Assumptions:
    - A HuggingFace-style checkpoint directory named 'byt5-finetuned'
      already exists under `work_dir`, e.g. `work/byt5-finetuned/`.
    - This directory was created by `T5ForConditionalGeneration.save_pretrained(...)`
      and `AutoTokenizer.save_pretrained(...)` in your Colab notebook.
    """

    CHECKPOINT_SUBDIR = "byt5-finetuned"

    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    # ------------------------
    # Data I/O helpers
    # ------------------------
    @classmethod
    def load_training_data(cls):
        """
        Training is done offline in the notebook; this CLI training step is a no-op.
        We keep this method to preserve the original interface.
        """
        return []

    @classmethod
    def load_test_data(cls, fname: str) -> List[str]:
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                inp = line.rstrip("\n")  # strip trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds: List[str], fname: str) -> None:
        with open(fname, "wt", encoding="utf-8") as f:
            for p in preds:
                f.write(f"{p}\n")

    # ------------------------
    # Training / saving
    # ------------------------
    def run_train(self, data, work_dir: str) -> None:
        """
        No-op in this script: we assume you've already fine-tuned ByT5 in Colab
        and copied the 'byt5-finetuned' directory into `work_dir`.
        """
        print(
            "run_train: skipping training. "
            "Assuming fine-tuned checkpoint already exists in "
            f"'{os.path.join(work_dir, self.CHECKPOINT_SUBDIR)}'."
        )

    def save(self, work_dir: str) -> None:
        """
        No-op here because training/fine-tuning is handled externally.
        We keep this method to satisfy the original interface.
        """
        checkpoint_dir = os.path.join(work_dir, self.CHECKPOINT_SUBDIR)
        print(f"save: assuming checkpoint is already in {checkpoint_dir}; nothing to do.")

    # ------------------------
    # Loading
    # ------------------------
    @classmethod
    def load(cls, work_dir: str) -> "MyModel":
        """
        Load the fine-tuned ByT5 checkpoint from `work_dir/byt5-finetuned`.
        """
        checkpoint_dir = os.path.join(work_dir, cls.CHECKPOINT_SUBDIR)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"Expected checkpoint directory '{checkpoint_dir}' not found.\n"
                "Make sure you've copied your fine-tuned 'byt5-finetuned' folder into 'work/'."
            )

        print(f"Loading tokenizer and model from '{checkpoint_dir}'")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        return cls(tokenizer=tokenizer, model=model, device=device)

    # ------------------------
    # Prediction
    # ------------------------
    def run_pred(self, data: List[str]) -> List[str]:
        """
        Given a list of input prefixes, predict a 3-character string for each,
        where each character is a likely next character.

        Logic mirrors the notebook's `predict_next_chars`:
        - Use beam search with num_beams=3, num_return_sequences=3
        - Decode each beam and take the first character
        - Deduplicate while preserving order
        - Pad with '?' if fewer than 3 unique characters
        """
        preds: List[str] = []
        self.model.eval()

        for i, prefix in enumerate(data):
            # Tokenize the prefix
            inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)

            # Generate with beam search — get 3 different sequences
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_beams=3,
                    num_return_sequences=3,
                    early_stopping=True,
                )

            # Decode each beam's output and extract the first character
            candidates = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded:
                    candidates.append(decoded[0])  # first character
                else:
                    # Fallback if the model only produced special tokens
                    candidates.append("?")

            # Deduplicate while preserving order
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            # Ensure exactly 3 characters
            pred = "".join(unique_candidates[:3])
            if len(pred) < 3:
                pred += "?" * (3 - len(pred))

            preds.append(pred)

            if (i + 1) % 100 == 0:
                print(f"  Predicted {i + 1}/{len(data)}")

        return preds


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save / load checkpoints", default="work")
    parser.add_argument("--test_data", help="path to test data", default="example/input.txt")
    parser.add_argument("--test_output", help="path to write test predictions", default="pred.txt")
    args = parser.parse_args()

    if args.mode == "train":
        # Keep interface but do not retrain inside Docker; we rely on the existing checkpoint.
        if not os.path.isdir(args.work_dir):
            print(f"Making working directory {args.work_dir}")
            os.makedirs(args.work_dir)
        print("Instantiating model")
        model = MyModel.load(args.work_dir)
        print("Skipping training (model assumed pre-finetuned).")
        model.run_train([], args.work_dir)
        print("Training step completed (no-op).")
        model.save(args.work_dir)
    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print(f"Loading test data from {args.test_data}")
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print(f"Writing predictions to {args.test_output}")
        assert len(pred) == len(
            test_data
        ), f"Expected {len(test_data)} predictions but got {len(pred)}"
        MyModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f"Unknown mode {args.mode}")
