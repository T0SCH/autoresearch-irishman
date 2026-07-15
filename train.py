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
        self.rnn = nn.GRU(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        # Weight tying: share the embedding and output-head weight matrix (free win in both
        # compute and params, confirmed on lstm-improve). Requires embed_size == hidden_size
        # so the two matrices have the same shape (vocab x hidden).
        if config.embed_size == config.hidden_size:
            self.head.weight = self.embed.weight  # tied; head.bias stays an independent param

    def forward(self, idx, targets=None, reduction='mean'):
        x = self.embed(idx)
        x, _ = self.rnn(x)  # zero-initialized hidden state per batch, no state carry across steps
        x = self.drop(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=self.config.pad_token_id, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
RNN_TYPE = "gru"  # not a hyperparameter — set per worktree to tag the architecture family in
                          # wandb (rnn/lstm/gru/birnn); the agent updates this when it swaps the recurrent cell
EMBED_SIZE = 384  # = HIDDEN_SIZE, to enable weight tying. Raised with hidden for the capacity bracket
                # (GRU 3 gates < LSTM 4 -> cheaper/step -> expect higher ceiling than LSTM's 256). run-1817b08
                # showed embed-up alone hurts at 256; if 384 wins anyway, hidden capacity dominates.
HIDDEN_SIZE = 384  # bracket above LSTM's 256 ceiling; GRU is cheaper/step so the ceiling may sit higher.
NUM_LAYERS = 2
DROPOUT = 0.1             # GRU-side test: LSTM found 0.1 a clean repeatable win (confirmed twice), RNN
                          # the opposite (dropout hurt 4x). Priors disagree -> let GRU's own result decide.

# Optimization
LEARNING_RATE = 0.003      # was 0.004 (optimum at TRAIN_SEQ_LEN=1024). Re-bracket at TRAIN_SEQ_LEN=512:
                          # 1.53x more update steps may allow a lower LR to converge better.
WEIGHT_DECAY = 0.1        # was 0.05 (tuned at hidden=256). Re-bracket at hidden=384: bigger model ->
                          # more overfitting risk -> maybe more weight decay helps. One variable.
GRAD_CLIP = 1.0            # RNNs are prone to exploding gradients, clip by global norm

BATCH_SIZE = 64            # reduce if OOM
TRAIN_SEQ_LEN = 512        # training crop length (val stays at MAX_SEQ_LEN=1024 for comparability).
                          # throughput-vs-context tradeoff: shorter crops -> ~2x tokens but less long-range.
assert TRAIN_SEQ_LEN <= MAX_SEQ_LEN
EVAL_EVERY = 50            # steps between quick val checks (loss/top1/top5) for wandb charts

SAMPLE_CHECK = True      # bar-structure spot-check (program.md every-10th-keep mechanism): when True,
                           # generate 3 samples (temp 0.8, ~512 tok, seeded from a val tune) after eval
                           # and write them to samples.txt. Runs after eval so it never affects val_bpb.
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

# Mixed precision: T4 is Turing (no native bf16) -> fp16 + GradScaler. The #1 transferable
# learning from lstm-improve (~2.5x throughput, no accuracy cost). scaler.unscale_(optimizer)
# MUST run before clip_grad_norm_ (else clipping acts on still-scaled grads by mistake).
USE_AMP = device.type == "cuda"
AMP_DTYPE = torch.float16
scaler = torch.amp.GradScaler(device=device.type, enabled=USE_AMP)

train_loader = make_dataloader(tokenizer, BATCH_SIZE, TRAIN_SEQ_LEN, "train", device)
val_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "val", device)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")

# offline mode: no network calls during the run (avoids stalls in the unattended overnight loop),
# `wandb sync wandb/offline-run-*` uploads everything afterward
wandb.init(project="autoresearch-irishman", mode="offline",
    tags=[RNN_TYPE], config={
    "device": device.type, "rnn_type": RNN_TYPE, "embed_size": EMBED_SIZE, "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
    "num_params": num_params, "use_amp": USE_AMP, "amp_dtype": str(AMP_DTYPE),
    "weight_tying": (config.embed_size == config.hidden_size),
})


@torch.no_grad()
def quick_eval():
    """Cheap single-batch val check (loss/top1/top5) for wandb charts — not the final val_bpb metric."""
    model.eval()
    x_val, y_val, _ = next(val_loader)
    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
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

    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
        loss = model(x, y)
    train_loss_f = loss.item()  # real (unscaled) loss: autocast keeps loss at true scale, scaler only scales for backward
    tokens_this_step = x.numel()
    scaler.scale(loss).backward()  # no-op when USE_AMP=False (scaler enabled=False passes through)
    x, y, epoch = next(train_loader)

    scaler.unscale_(optimizer)  # must precede clip_grad_norm_ so clipping sees real grad magnitudes
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
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

if SAMPLE_CHECK:
    # Bar-structure spot-check (program.md): 3 samples at temp 0.8, ~512 tokens, seeded from a val
    # tune's first chars so the model has a realistic ABC header to continue. Writes to samples.txt
    # (not run.log) to avoid flooding the training log. Purely qualitative - never affects val_bpb.
    model.eval()
    val_tunes = load_tunes("val")
    srng = random.Random(0)
    bos = tokenizer.get_bos_token_id()
    with open("samples.txt", "w") as _f:
        for _si in range(3):
            seed_tune = srng.choice(val_tunes)
            prompt = tokenizer.encode(seed_tune[:64], prepend=bos)
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.no_grad():
                for _ in range(512):
                    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                        logits = model(ids)
                    logits = logits[:, -1, :] / 0.8  # temperature 0.8
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                    ids = torch.cat([ids, nxt], dim=1)
            _f.write(f"=== sample {_si+1} (seed tune head: {seed_tune[:32]!r}) ===\n")
            _f.write(tokenizer.decode(ids[0].tolist()) + "\n\n")
    print("wrote samples.txt (SAMPLE_CHECK)")

if SAVE_CHECKPOINT:
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, "checkpoint.pt")
    print("Saved checkpoint.pt (git add + commit it manually -- this is a deliberate one-off, not part of the loop)")
