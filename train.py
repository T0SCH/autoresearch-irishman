"""
Autoresearch training script: char-level RNN for Irish tune (ABC notation) generation.
Single-GPU, single-file.
Usage: uv run train.py
"""

import math
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, load_tunes, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# RNN model
# ---------------------------------------------------------------------------

@dataclass
class RNNConfig:
    vocab_size: int = 100
    embed_size: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    pad_token_id: int = 0


class CharRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.rnn = nn.RNN(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.h0 = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_size))

    def forward(self, idx, targets=None, reduction='mean'):
        x = self.embed(idx)
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        x, _ = self.rnn(x, h0)  # learned initial hidden state (shared across batch), no state carry across steps
        x = self.drop(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=self.config.pad_token_id, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Length-bucketed training dataloader
# ---------------------------------------------------------------------------

def make_bucketed_dataloader(tokenizer, token_budget, max_items, T, device, seed=42):
    """
    Length-bucketed TRAIN-only dataloader: greedily packs tunes into a batch until
    items_in_batch * max_len_in_batch would exceed token_budget (capped at max_items),
    instead of make_dataloader's fixed batch size + pad-to-longest-in-random-batch.
    Measured padding waste under random batch_size=32 batching was ~59% of tokens
    (x.numel() includes padding, which the loss/eval both mask out) -- short tunes
    landing with a long one in the same random batch means most of that batch's
    compute goes to PAD. Bucketing keeps real compute per step roughly constant
    instead: many short tunes per batch, few long ones, ~10% padding waste measured.
    A small random jitter is added to lengths before sorting, and buckets are rebuilt
    (rejittered) every epoch, so tunes of similar length don't always land in the same
    batch together epoch after epoch (which would correlate their gradients).
    Only replaces the TRAIN loader -- evaluate_bpb's internal loader (prepare.py,
    unmodified) and the quick_eval val_loader both keep fixed-batch_size make_dataloader
    so the metrics stay comparable across configs.
    """
    tunes = load_tunes("train")
    bos, pad = tokenizer.get_bos_token_id(), tokenizer.get_pad_token_id()
    encoded = [tokenizer.encode(t, prepend=bos)[:T + 1] for t in tunes]
    rng = random.Random(seed)
    epoch = 1

    while True:
        order = sorted(range(len(encoded)), key=lambda idx: len(encoded[idx]) + rng.uniform(-20, 20))
        i = 0
        while i < len(order):
            bucket = [order[i]]
            max_len = len(encoded[order[i]])
            i += 1
            while i < len(order) and len(bucket) < max_items:
                cand_len = max(max_len, len(encoded[order[i]]))
                if (len(bucket) + 1) * cand_len > token_budget:
                    break
                bucket.append(order[i])
                max_len = cand_len
                i += 1

            rows = [list(encoded[j]) for j in bucket]
            width = max(len(row) for row in rows)
            for row in rows:
                row.extend([pad] * (width - len(row)))

            batch = torch.tensor(rows, dtype=torch.long).to(device)
            yield batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), epoch
        epoch += 1

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
EMBED_SIZE = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
DROPOUT = 0.0             # helps once training does multiple epochs (confirmed on a fast GPU: 3 epochs
                          # in 300s overfits without it); on slower hardware a run may not even finish one

# Optimization
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0            # RNNs are prone to exploding gradients, clip by global norm
WARMUP_STEPS = 20          # linear warmup, then cosine decay over the wall-clock time budget
                           # (time-based, not step-based -- step count varies a lot across configs)

BATCH_SIZE = 32            # only used for the val_loader/evaluate_bpb (fixed-batch, must stay
                           # comparable across configs) -- training uses TOKEN_BUDGET below instead
TOKEN_BUDGET = 32768       # length-bucketed training batches: pack items until items*max_len_in_bucket
                           # hits this -- doubled again from 16384, which beat 8192; testing if the
                           # larger-less-noisy-batch trend continues before calling it settled
MAX_BUCKET_ITEMS = 128     # cap so very-short-tune buckets don't get absurdly large
EVAL_EVERY = 50            # steps between quick val checks (loss/top1/top5) for wandb charts

SAVE_CHECKPOINT = False    # off by default -- every kept experiment would otherwise add a multi-MB
                           # blob to git history. Flip to True only for the one deliberate final run.

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed(42)
elif device.type == "mps":
    torch.mps.manual_seed(42)
print(f"Device: {device.type}")
torch.set_float32_matmul_precision("high")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = RNNConfig(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS, dropout=DROPOUT, pad_token_id=tokenizer.get_pad_token_id())
print(f"Model config: {config}")

model = CharRNN(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Num params: {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loader = make_bucketed_dataloader(tokenizer, TOKEN_BUDGET, MAX_BUCKET_ITEMS, MAX_SEQ_LEN, device)
val_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "val", device)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")

# offline mode: no network calls during the run (avoids stalls in the unattended overnight loop),
# `wandb sync wandb/offline-run-*` uploads everything afterward
wandb.init(project="autoresearch-irishman", mode="offline", config={
    "device": device.type, "embed_size": EMBED_SIZE, "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
    "token_budget": TOKEN_BUDGET, "max_bucket_items": MAX_BUCKET_ITEMS, "num_params": num_params,
})


@torch.no_grad()
def quick_eval():
    """Cheap single-batch val check (loss/top1/top5) for wandb charts — not the final val_bpb metric."""
    model.eval()
    x_val, y_val, _ = next(val_loader)
    logits = model(x_val)
    targets_flat = y_val.view(-1)
    mask = targets_flat != config.pad_token_id
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_flat, ignore_index=config.pad_token_id)
    top5 = logits.view(-1, logits.size(-1)).topk(5, dim=-1).indices
    top1_correct = (top5[:, 0] == targets_flat) & mask
    top5_correct = (top5 == targets_flat.unsqueeze(-1)).any(dim=-1) & mask
    model.train()
    return loss.item(), (top1_correct.sum() / mask.sum()).item(), (top5_correct.sum() / mask.sum()).item()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

total_training_time = 0
total_tokens = 0
step = 0

while True:
    sync()
    t0 = time.time()

    loss = model(x, y)
    train_loss_f = loss.item()
    tokens_this_step = x.numel()
    loss.backward()
    x, y, epoch = next(train_loader)

    if step < WARMUP_STEPS:
        lr_scale = (step + 1) / WARMUP_STEPS
    else:
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = LEARNING_RATE * lr_scale

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    sync()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt
        total_tokens += tokens_this_step

    pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    tok_per_sec = int(tokens_this_step / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {train_loss_f:.6f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
    log = {"loss": train_loss_f, "tok_per_sec": tok_per_sec, "epoch": epoch,
           "grad_norm": grad_norm.item(), "lr": optimizer.param_groups[0]["lr"]}
    if step % EVAL_EVERY == 0:
        val_loss, top1_acc, top5_acc = quick_eval()
        log.update({"val_loss": val_loss, "top1_acc": top1_acc, "top5_acc": top5_acc})
    wandb.log(log, step=step)

    step += 1

    # Time's up — but only stop after warmup steps so we don't count startup
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE, device)
_, final_top1_acc, final_top5_acc = quick_eval()

# Final summary
t_end = time.time()
if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device.type == "mps":
    peak_vram_mb = torch.mps.current_allocated_memory() / 1024 / 1024  # MPS has no peak tracker, current is the closest signal
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"top1_acc:         {final_top1_acc:.4f}")
print(f"top5_acc:         {final_top5_acc:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.3f}")
print(f"num_layers:       {NUM_LAYERS}")
print(f"hidden_size:      {HIDDEN_SIZE}")

wandb.log({"val_bpb": val_bpb, "top1_acc": final_top1_acc, "top5_acc": final_top5_acc,
           "peak_vram_mb": peak_vram_mb, "total_tokens_M": total_tokens / 1e6})
wandb.finish()

if SAVE_CHECKPOINT:
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, "checkpoint.pt")
    print("Saved checkpoint.pt (git add + commit it manually -- this is a deliberate one-off, not part of the loop)")
