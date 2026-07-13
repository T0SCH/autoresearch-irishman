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

MAX_SEQ_LEN = 1024        # context length (covers ~99% of tunes uncropped, p99=813 chars)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 500_000     # number of tokens for val eval (~most of the 2162-tune val set)

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
        json.dump({"chars": chars, "bos_token": BOS_TOKEN, "unk_token": UNK_TOKEN}, f)
    print(f"Tokenizer: saved vocab to {vocab_path}")

    # token_bytes: byte length per token id, for BPB eval. Specials (BOS, UNK) are 0.
    token_bytes_list = [len(c.encode("utf-8")) for c in chars] + [0, 0]
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
    """Char-level tokenizer. One token id per character, plus BOS/UNK. Training is handled above."""

    def __init__(self, chars, bos_token, unk_token):
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = list(chars) + [bos_token, unk_token]
        self.bos_token_id = len(chars)
        self.unk_token_id = len(chars) + 1

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "vocab.json")) as f:
            vocab = json.load(f)
        return cls(vocab["chars"], vocab["bos_token"], vocab["unk_token"])

    def get_vocab_size(self):
        return len(self.id_to_char)

    def get_bos_token_id(self):
        return self.bos_token_id

    def _encode_one(self, text):
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def encode(self, text, prepend=None):
        if isinstance(text, str):
            ids = self._encode_one(text)
            if prepend is not None:
                ids.insert(0, prepend)
        elif isinstance(text, list):
            ids = [self._encode_one(t) for t in text]
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from the loaded tune list."""
    tunes = load_tunes(split)
    assert len(tunes) > 0, f"No tunes found for split={split}. Run prepare.py first."
    epoch = 1
    while True:
        for i in range(0, len(tunes), tokenizer_batch_size):
            yield tunes[i:i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="mps")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="mps")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
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
