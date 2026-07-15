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

from prepare import MAX_SEQ_LEN, Tokenizer, load_tunes, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Sampling (for the mandatory every-10th-keep bar-line/meter spot-check)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(model, tokenizer, device, seed_text, max_new_tokens=512, temperature=0.8):
    model.eval()
    ids = tokenizer.encode(seed_text, prepend=tokenizer.get_bos_token_id())
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x[:, -MAX_SEQ_LEN:])
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    model.train()
    return tokenizer.decode(x[0].tolist())

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
        assert config.embed_size == config.hidden_size, "weight tying requires embed_size == hidden_size"
        assert config.num_layers == 3, "residual connection below is hardcoded for exactly 3 layers"
        # own idea: extend the confirmed 2-layer residual/skip connection (b31523f) to a 3rd layer,
        # now that the residual path has already shown it eases the gradient-flow problem that made
        # a plain (non-residual) NUM_LAYERS=3 fail earlier in this worktree. Same ResNet-style running-
        # sum pattern as before, just one more step: each layer's raw output is added to the running
        # sum, and that running sum (not the raw output) feeds the next layer -- x2 = x1 + rnn2(x1),
        # x3 = x2 + rnn3(x2). Same per-layer parameter count as before, one extra LSTM layer's worth
        # of params overall.
        self.rnn1 = nn.LSTM(input_size=config.embed_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.rnn3 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.head.weight = self.embed.weight  # weight tying (own idea) -- embed_size==hidden_size
                                              # makes the shapes match (both [vocab_size, hidden_size]).
                                              # Classic char-rnn/LM trick: shares the input/output token
                                              # representation, cuts ~vocab_size*hidden_size duplicate
                                              # params and often regularizes/improves generalization.
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
        x1, (h1, c1) = self.rnn1(x, (h0[0:1], c0[0:1]))
        x2, (h2, c2) = self.rnn2(x1, (h0[1:2], c0[1:2]))
        y2 = x1 + x2  # running residual sum after layer 2
        x3, (h3, c3) = self.rnn3(y2, (h0[2:3], c0[2:3]))
        x = y2 + x3  # running residual sum after layer 3
        self.last_hidden = (torch.cat([h1, h2, h3], dim=0).detach(), torch.cat([c1, c2, c3], dim=0).detach())
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
EMBED_SIZE = 256          # human steer: retest w/ hidden_size, now that fp16 (16abb36) gives ~2.5x
                          # throughput -- the earlier c255ddc attempt at this size was keep-prov
                          # (only 38% of median tokens under fp32), this run should finally land
                          # near the fp16-era step count and give a fair capacity comparison
HIDDEN_SIZE = 256
NUM_LAYERS = 3            # own idea: extend the confirmed residual connection (b31523f) to a 3rd
                          # layer. This worktree only ever directly tested NUM_LAYERS=1 on the LSTM
                          # (49c8d09, discard, too shallow) -- 3-layer without residuals was assumed
                          # to hurt by transferring baseline-improve's RNN finding, never directly
                          # tested here. Residuals may change that calculus by easing the gradient-
                          # flow / optimization-depth problem that made extra depth costly before.
DROPOUT = 0.1             # human steer: the one LSTM-allowed dropout check per program.md (RNN failed
                          # 4x across every config there) -- now is the right moment: strong config
                          # settled, hidden_size/weight_decay give real capacity+regularization headroom.
                          # If it hurts like it did for RNN, settle DROPOUT=0.0 for good, no further sweep.

# Optimization -- held at run e980c7a's confirmed values: this experiment isolates
# stateful vs. stateless windowing alone (run 5b26872 already tested stateless windowing
# in isolation and found it a wash vs. whole-tune; this changes exactly one more thing --
# carrying (h,c) across a tune's windows -- on top of that).
LEARNING_RATE = 0.003     # own idea: continue the fresh HIDDEN_SIZE=256 bracket downward -- 0.008->0.005
                          # (f9e43b5) was a real, matched-throughput win, opposite direction from the
                          # hidden_size=128 bracket's optimum. Checking whether the trend continues to
                          # 0.003 (the pre-bracket original default) or 0.005 is already the local optimum.
WEIGHT_DECAY = 0.05        # human steer: isolated re-test -- never tested alone on the LSTM (only in
                           # 5fc6b30's 5-variable bundle). At ~0.29M params with multiple epochs of
                           # exposure (9000+ steps), more plausible as a real regularizer here than
                           # it would have been on a much smaller/undertrained early run.
GRAD_CLIP = 1.0            # RNNs are prone to exploding gradients, clip by global norm

BATCH_SIZE = 64            # only used for the val_loader/evaluate_bpb (fixed-batch, must stay
                           # comparable across configs) -- training uses the stateful windowed loader below
TRAIN_SEQ_LEN = 256        # truncated-BPTT window length; doubled from the confirmed-keep 128 now
                           # that stateful carry works, to see whether a longer single BPTT horizon
                           # (fewer, bigger windows/tune) helps further or whether 128 already captured
                           # most of the benefit
WINDOW_BATCH_SIZE = 256    # own idea: WINDOW_BATCH_SIZE has always moved together with TRAIN_SEQ_LEN
                           # to hold tokens/step constant, never swept independently at fixed seq_len.
                           # Doubling here (holding TRAIN_SEQ_LEN=256) tests pure batch diversity/size.
EVAL_EVERY = 50            # steps between quick val checks (loss/top1/top5) for wandb charts

NUM_EPOCHS = 50            # unbounded full run (no TIME_BUDGET): trains for exactly this many
                           # full passes over the ~214k-tune training set (~62M chars/epoch).
                           # Final submission run.

SAVE_CHECKPOINT = True     # flipped True for the one deliberate final run.
SAMPLE_CHECK = True        # flipped True for the mandatory every-10th-keep bar-line/meter
                           # spot-check (program.md's mechanism) -- prints 3 samples to stdout.

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
# human steer: fp16 mixed precision (NOT bf16 -- T4 is Turing, no native bf16 tensor cores,
# bf16 would run emulated/slow there and falsely look like "doesn't help"). GradScaler is
# mandatory with fp16 (unlike bf16) since fp16's narrow exponent range underflows small
# gradients without loss scaling. Goal: cut per-step wall time so step-starved keep-prov
# configs (e.g. HIDDEN_SIZE=256) can reach a fair, throughput-matched comparison.
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

train_loader = make_stateful_windowed_dataloader(tokenizer, TRAIN_SEQ_LEN, WINDOW_BATCH_SIZE, MAX_SEQ_LEN, device)
val_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "val", device)
x, y, epoch, reset_mask = next(train_loader)  # prefetch first batch
carried_h, carried_c = None, None

print(f"Epoch budget: {NUM_EPOCHS} epochs (no time limit)")

# offline mode: no network calls during the run (avoids stalls in the unattended overnight loop),
# `wandb sync wandb/offline-run-*` uploads everything afterward
wandb.init(project="autoresearch-irishman", mode="offline",
    tags=[RNN_TYPE], config={
    "device": device.type, "rnn_type": RNN_TYPE, "embed_size": EMBED_SIZE, "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
    "train_seq_len": TRAIN_SEQ_LEN, "window_batch_size": WINDOW_BATCH_SIZE, "num_params": num_params,
    "num_epochs": NUM_EPOCHS,
    "use_amp": use_amp,
})


QUICK_EVAL_BATCHES = 8

@torch.no_grad()
def quick_eval():
    """Val check over QUICK_EVAL_BATCHES batches (loss/top1/top5), token-weighted across
    batches (not a naive mean of per-batch means), for wandb charts -- not the final val_bpb metric."""
    model.eval()
    loss_sum = 0.0
    top1_correct_sum = 0
    top5_correct_sum = 0
    token_count = 0
    for _ in range(QUICK_EVAL_BATCHES):
        x_val, y_val, _ = next(val_loader)
        logits = model(x_val)
        targets_flat = y_val.view(-1)
        mask = targets_flat != config.pad_token_id
        loss_sum += F.cross_entropy(logits.view(-1, logits.size(-1)), targets_flat,
                                     ignore_index=config.pad_token_id, reduction="sum").item()
        top5 = logits.view(-1, logits.size(-1)).topk(5, dim=-1).indices
        top1_correct_sum += ((top5[:, 0] == targets_flat) & mask).sum().item()
        top5_correct_sum += ((top5 == targets_flat.unsqueeze(-1)).any(dim=-1) & mask).sum().item()
        token_count += mask.sum().item()
    model.train()
    return loss_sum / token_count, top1_correct_sum / token_count, top5_correct_sum / token_count

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

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss = model(x, y, init_state=init_state)
    train_loss_f = loss.item()
    tokens_this_step = x.numel()
    scaler.scale(loss).backward()
    # carried (h, c) kept as the fp32 "master" state across steps, regardless of the fp16
    # compute inside this step's forward -- avoids accumulating fp16 rounding error into the
    # long-lived recurrent state over ~thousands of steps, and keeps dtype consistent with
    # the fp32 learned h0/c0 it gets torch.where-blended against at the top of the next step.
    carried_h, carried_c = model.last_hidden
    carried_h = carried_h.float()
    carried_c = carried_c.float()
    x, y, epoch, reset_mask = next(train_loader)

    scaler.unscale_(optimizer)
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

    tok_per_sec = int(tokens_this_step / dt)

    print(f"\rstep {step:05d} | epoch {epoch}/{NUM_EPOCHS} | loss: {train_loss_f:.6f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | elapsed: {total_training_time:.0f}s    ", end="", flush=True)
    log = {"loss": train_loss_f, "tok_per_sec": tok_per_sec, "epoch": epoch,
           "grad_norm": grad_norm.item(), "lr": optimizer.param_groups[0]["lr"]}
    if step % EVAL_EVERY == 0:
        val_loss, top1_acc, top5_acc = quick_eval()
        log.update({"val_loss": val_loss, "top1_acc": top1_acc, "top5_acc": top5_acc})
    wandb.log(log, step=step)

    step += 1

    # epoch ticks over once the shuffled tune queue wraps (see make_stateful_windowed_dataloader) --
    # epoch > NUM_EPOCHS means we've started consuming NUM_EPOCHS+1's tunes, i.e. NUM_EPOCHS full
    # passes are done. Only stop after warmup steps so we don't count startup.
    if step > 10 and epoch > NUM_EPOCHS:
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

if SAMPLE_CHECK:
    val_tunes = load_tunes("val")
    print("\n=== SAMPLE CHECK (temp=0.8, ~512 tokens, seeded from val tunes) ===")
    for i in range(3):
        seed = val_tunes[i][:40]
        sample = generate_sample(model, tokenizer, device, seed, max_new_tokens=512, temperature=0.8)
        print(f"\n--- sample {i+1} (seed: {seed!r}) ---")
        print(sample)
