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
        self.rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.h0 = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_size))
        self.last_hidden = None  # side channel: (h_n, c_n) detached, set by every forward() call --
                                 # lets the training loop thread state across steps without changing
                                 # forward's public return value (quick_eval/evaluate_bpb stay untouched)

    def forward(self, idx, targets=None, reduction='mean', init_state=None):
        x = self.embed(idx)
        if init_state is None:
            h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
            c0 = self.c0.expand(-1, x.size(0), -1).contiguous()
        else:
            h0, c0 = init_state
        x, (hn, cn) = self.rnn(x, (h0, c0))
        self.last_hidden = (hn.detach(), cn.detach())
        x = self.drop(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=self.config.pad_token_id, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Stateful truncated-BPTT training dataloader (train.py-only, prepare.py untouched)
# ---------------------------------------------------------------------------

def make_stateful_windowed_dataloader(tokenizer, seq_len, batch_size, T, device, seed=42):
    """
    batch_size parallel lanes, each stepping sequentially through one tune's tokens in
    consecutive seq_len-length chunks (no shuffling within a tune -- order matters here,
    unlike the stateless windowed loader). When a lane's tune is exhausted it picks up
    the next tune from a shuffled per-epoch queue, and that lane's first chunk of the new
    tune is flagged in the returned reset_mask so the training loop resets that lane's
    carried (h, c) back to the model's learned (h0, c0) instead of leaking state from an
    unrelated tune. Detached truncated BPTT: the gradient horizon per step stays at
    seq_len, but forward-pass context now persists across a tune's own windows -- this is
    the LSTM-specific capability the stateless windowed loader (discarded: wash vs.
    whole-tune) couldn't exercise, since it reset to (h0, c0) every window.
    Only replaces the TRAIN loader -- evaluate_bpb (prepare.py, unmodified) and
    quick_eval's val_loader keep fixed-batch full-sequence make_dataloader, so val_bpb
    stays comparable across every run in this ledger.
    """
    tunes = load_tunes("train")
    bos, pad = tokenizer.get_bos_token_id(), tokenizer.get_pad_token_id()
    encoded = [tokenizer.encode(t, prepend=bos)[:T + 1] for t in tunes]

    rng = random.Random(seed)
    epoch = 1
    queue = list(range(len(encoded)))
    rng.shuffle(queue)
    qpos = 0

    def next_tune_idx():
        nonlocal qpos, epoch, queue
        if qpos >= len(queue):
            queue = list(range(len(encoded)))
            rng.shuffle(queue)
            qpos = 0
            epoch += 1
        idx = queue[qpos]
        qpos += 1
        return idx

    remaining = [list(encoded[next_tune_idx()]) for _ in range(batch_size)]

    while True:
        x_batch, y_batch, reset_mask = [], [], []
        for lane in range(batch_size):
            is_reset = False
            if len(remaining[lane]) < 2:
                remaining[lane] = list(encoded[next_tune_idx()])
                is_reset = True
            chunk = remaining[lane][:seq_len + 1]
            if len(chunk) < seq_len + 1:
                chunk = chunk + [pad] * (seq_len + 1 - len(chunk))
            remaining[lane] = remaining[lane][seq_len:]
            x_batch.append(chunk[:-1])
            y_batch.append(chunk[1:])
            reset_mask.append(is_reset)
        x = torch.tensor(x_batch, dtype=torch.long).to(device)
        y = torch.tensor(y_batch, dtype=torch.long).to(device)
        reset_mask_t = torch.tensor(reset_mask, dtype=torch.bool, device=device)
        yield x, y, epoch, reset_mask_t

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
RNN_TYPE = "lstm"  # not a hyperparameter — set per worktree to tag the architecture family in
                          # wandb (rnn/lstm/gru/birnn); the agent updates this when it swaps the recurrent cell
EMBED_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.0             # helps once training does multiple epochs (confirmed on a fast GPU: 3 epochs
                          # in 300s overfits without it); on slower hardware a run may not even finish one

# Optimization -- held at run e980c7a's confirmed values: this experiment isolates
# stateful vs. stateless windowing alone (run 5b26872 already tested stateless windowing
# in isolation and found it a wash vs. whole-tune; this changes exactly one more thing --
# carrying (h,c) across a tune's windows -- on top of that).
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0            # RNNs are prone to exploding gradients, clip by global norm

BATCH_SIZE = 64            # only used for the val_loader/evaluate_bpb (fixed-batch, must stay
                           # comparable across configs) -- training uses the stateful windowed loader below
TRAIN_SEQ_LEN = 256        # truncated-BPTT window length; doubled from the confirmed-keep 128 now
                           # that stateful carry works, to see whether a longer single BPTT horizon
                           # (fewer, bigger windows/tune) helps further or whether 128 already captured
                           # most of the benefit
WINDOW_BATCH_SIZE = 128    # halved vs. seq_len=128's 256 to hold batch_size*seq_len ~= 32768 constant,
                           # isolating window length from total tokens/step
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

train_loader = make_stateful_windowed_dataloader(tokenizer, TRAIN_SEQ_LEN, WINDOW_BATCH_SIZE, MAX_SEQ_LEN, device)
val_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "val", device)
x, y, epoch, reset_mask = next(train_loader)  # prefetch first batch
carried_h, carried_c = None, None

print(f"Time budget: {TIME_BUDGET}s")

# offline mode: no network calls during the run (avoids stalls in the unattended overnight loop),
# `wandb sync wandb/offline-run-*` uploads everything afterward
wandb.init(project="autoresearch-irishman", mode="offline",
    tags=[RNN_TYPE], config={
    "device": device.type, "rnn_type": RNN_TYPE, "embed_size": EMBED_SIZE, "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
    "train_seq_len": TRAIN_SEQ_LEN, "window_batch_size": WINDOW_BATCH_SIZE, "num_params": num_params,
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

    if carried_h is None:
        init_state = None
    else:
        h0_learned = model.h0.expand(-1, x.size(0), -1).contiguous()
        c0_learned = model.c0.expand(-1, x.size(0), -1).contiguous()
        rm = reset_mask.view(1, -1, 1)
        init_state = (torch.where(rm, h0_learned, carried_h), torch.where(rm, c0_learned, carried_c))

    loss = model(x, y, init_state=init_state)
    train_loss_f = loss.item()
    tokens_this_step = x.numel()
    loss.backward()
    carried_h, carried_c = model.last_hidden
    x, y, epoch, reset_mask = next(train_loader)

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
