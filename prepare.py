"""
One-time data preparation for autoresearch experiments.
Downloads the IrishMAN ABC-notation dataset and builds a char-level tokenizer.

Usage:
    python prepare.py

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import json

import requests
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
BASE_URL = "https://huggingface.co/datasets/sander-wood/irishman/resolve/main"
TRAIN_FILENAME = "train.json"
VAL_FILENAME = "validation.json"
ABC_FIELD = "abc notation"  # the field in each record holding the tune text

BOS_TOKEN = "<BOS>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_file(filename):
    """Download one dataset file with retries. Exits on repeated failure."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    print(f"  Failed to download {filename} after {max_attempts} attempts")
    sys.exit(1)


def download_data():
    """Download the (pre-split) IrishMAN train/validation files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = [f for f in (TRAIN_FILENAME, VAL_FILENAME) if os.path.exists(os.path.join(DATA_DIR, f))]
    if len(existing) == 2:
        print(f"Data: train/validation already downloaded at {DATA_DIR}")
        return
    print("Data: downloading IrishMAN train/validation splits...")
    download_file(TRAIN_FILENAME)
    download_file(VAL_FILENAME)
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
    chars = set()
    for tune in tunes:
        chars.update(tune)
    chars = sorted(chars)
    print(f"Tokenizer: found {len(chars)} unique characters")

    with open(vocab_path, "w") as f:
        json.dump({"chars": chars, "bos_token": BOS_TOKEN, "unk_token": UNK_TOKEN, "pad_token": PAD_TOKEN}, f)
    print(f"Tokenizer: saved vocab to {vocab_path}")

    # token_bytes: byte length per token id, for BPB eval. Specials (BOS, UNK, PAD) are 0.
    token_bytes_list = [len(c.encode("utf-8")) for c in chars] + [0, 0, 0]
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
    """
    assert split in ["train", "val"]
    tunes = load_tunes(split)
    assert len(tunes) > 0, f"No tunes found for split={split}. Run prepare.py first."
    bos, pad = tokenizer.get_bos_token_id(), tokenizer.get_pad_token_id()
    epoch = 1
    i = 0

    while True:
        rows = []
        for _ in range(B):
            if i >= len(tunes):
                i, epoch = 0, epoch + 1
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
