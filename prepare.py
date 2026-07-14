"""
One-time data preparation for autoresearch experiments.
Downloads the IrishMAN ABC-notation dataset and builds a char-level tokenizer.

Usage:
    python prepare.py

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # xet transfer backend hangs/times out on Colab, plain HTTP works fine

import math
import json
import random

from huggingface_hub import hf_hub_download
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024        # max context length (covers ~99% of tunes uncropped, p99=813 chars)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 500_000     # approx. number of tokens for val eval (~most of the 2162-tune val set)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
REPO_ID = "sander-wood/irishman"
TRAIN_FILENAME = "train.json"
VAL_FILENAME = "validation.json"
ABC_FIELD = "abc notation"  # the field in each record holding the tune text

BOS_TOKEN = "<BOS>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download the (pre-split) IrishMAN train/validation files. Skips files already cached."""
    print("Data: downloading IrishMAN train/validation splits...")
    for filename in (TRAIN_FILENAME, VAL_FILENAME):
        hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=DATA_DIR)
    print(f"Data: ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training (char-level)
# ---------------------------------------------------------------------------

def load_tunes(split):
    """Return list of ABC notation strings for split ('train' or 'val')."""
    filename = TRAIN_FILENAME if split == "train" else VAL_FILENAME
    with open(os.path.join(DATA_DIR, filename)) as f:
        records = json.load(f)
    return [r[ABC_FIELD] for r in records]


def train_tokenizer():
    """Build a char-level vocab from the training split, save as JSON."""
    vocab_path = os.path.join(TOKENIZER_DIR, "vocab.json")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(vocab_path) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already built at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: scanning training data for unique characters...")
    tunes = load_tunes("train")
    chars = sorted(set("".join(tunes)))
    print(f"Tokenizer: found {len(chars)} unique characters")

    with open(vocab_path, "w") as f:
        json.dump({"chars": chars, "bos_token": BOS_TOKEN, "unk_token": UNK_TOKEN, "pad_token": PAD_TOKEN}, f)
    print(f"Tokenizer: saved vocab to {vocab_path}")

    # token_bytes: byte length per token id, for BPB eval. ABC notation is ASCII, so every
    # real character is 1 byte; specials (BOS, UNK, PAD) are 0.
    token_bytes_list = [1] * len(chars) + [0, 0, 0]
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    tok = Tokenizer.from_directory()
    test = tunes[0][:200]
    assert tok.decode(tok.encode(test)) == test, f"Tokenizer roundtrip failed: {test!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={tok.get_vocab_size()})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Char-level tokenizer. One token id per character, plus BOS/UNK/PAD. Training is handled above."""

    def __init__(self, chars, bos_token, unk_token, pad_token):
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = list(chars) + [bos_token, unk_token, pad_token]
        self.bos_token_id = len(chars)
        self.unk_token_id = len(chars) + 1
        self.pad_token_id = len(chars) + 2

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "vocab.json")) as f:
            vocab = json.load(f)
        return cls(vocab["chars"], vocab["bos_token"], vocab["unk_token"], vocab["pad_token"])

    def get_vocab_size(self):
        return len(self.id_to_char)

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_pad_token_id(self):
        return self.pad_token_id

    def encode(self, text, prepend=None):
        ids = [self.char_to_id.get(c, self.unk_token_id) for c in text]
        if prepend is not None:
            ids.insert(0, prepend)
        return ids

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def make_dataloader(tokenizer, B, T, split, device):
    """
    Each row is one BOS-prefixed tune, cropped to at most T+1 tokens.
    Rows are padded with PAD up to the longest tune in the batch (not always T)
    PAD positions carry zero byte-length, so evaluate_bpb and the training loss
    (ignore_index=pad_token_id) both skip them automatically.
    Tunes are reshuffled (fixed seed) at the start and on every epoch wrap, so
    multi-epoch runs (common on faster GPUs) don't repeat identical batches.
    """
    assert split in ["train", "val"]
    tunes = load_tunes(split)
    assert len(tunes) > 0, f"No tunes found for split={split}. Run prepare.py first."
    rng = random.Random(42)
    rng.shuffle(tunes)
    bos, pad = tokenizer.get_bos_token_id(), tokenizer.get_pad_token_id()
    epoch = 1
    i = 0

    while True:
        rows = []
        for _ in range(B):
            if i >= len(tunes):
                i, epoch = 0, epoch + 1
                rng.shuffle(tunes)
            rows.append(tokenizer.encode(tunes[i], prepend=bos)[:T + 1])
            i += 1

        width = max(len(row) for row in rows)
        for row in rows:
            row.extend([pad] * (width - len(row)))

        batch = torch.tensor(rows, dtype=torch.long).to(device)
        yield batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data()
    print()

    # Step 2: Build tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
