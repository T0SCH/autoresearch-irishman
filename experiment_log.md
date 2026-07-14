# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 1eaa8e2 — keep
**Source:** agent (initial baseline, per program.md setup)
**Change:** none — train.py as-is: plain `nn.RNN`, embed_size=128, hidden_size=256, num_layers=2, dropout=0.0, AdamW lr=3e-3, batch_size=64.
**Result:** val_bpb 2.093672, memory 1.4GB
**Notes:** Establishes the baseline this worktree (RNN-only scope) will iterate from. top1_acc 0.5528, top5_acc 0.8652, 8180 steps, 421.5M tokens in 300s train time. Next experiments to try per program.md goal: larger hidden_size, more layers, dropout tuning, learned initial hidden state, LR schedule — no cell-type swap (that's lstm-improve/gru-improve's job).

## 0465f3d — discard
**Source:** agent
**Change:** `DROPOUT` 0.0 → 0.2 (baseline's own code comment flagged that fast-GPU runs completing 3 epochs in 300s overfit without dropout — baseline run had epoch=3, so this looked like a direct match).
**Result:** val_bpb 2.174009 (worse than baseline 2.093672), memory 1.4GB
**Notes:** top1/top5 ticked up slightly (0.5536/0.8725 vs 0.5528/0.8652) but val_bpb — the ground-truth metric — got worse. Possibly 0.2 is too aggressive for a model this small (0.268M params) and this short a training run (fewer effective updates seen due to regularization noise), or the baseline's "overfits without it" comment was about a different (larger/longer) setup than what this worktree's baseline actually ran. Worth retrying with a smaller dropout (e.g. 0.05–0.1) later rather than abandoning the idea entirely.

## 10b1c9b — keep
**Source:** agent
**Change:** `HIDDEN_SIZE` 256 → 384 (0.268M → 0.543M params).
**Result:** val_bpb 1.839949 (down from baseline 2.093672), memory 1.9GB (up from 1.4GB)
**Notes:** Big val_bpb win, but with an important caveat: throughput cratered from baseline's ~1.4M tok/sec average (421.5M tokens / 8180 steps / 300s) to ~0.18M tok/sec (53.6M tokens / 1045 steps / 300s) — a ~7.8x slowdown for only a 1.5x hidden_size increase, far more than FLOP scaling would predict ((384/256)^2 ≈ 2.25x). Likely a cuDNN RNN kernel-selection cliff at this hidden_size (persistent-kernel eligibility can be size-sensitive), not a fundamental compute cost. Consequence: this run only got partway through 1 epoch, vs. baseline's 3 full epochs — so part of the val_bpb win may be "accidental early stopping" (much less data seen ⇒ much less overfitting) rather than a pure capacity win. Worth testing other hidden_size values (512, 640, 768 — round numbers that may or may not hit the same cliff) to separate the capacity effect from the throughput effect, and worth retrying dropout now that the effective epoch count is much lower.

## 17b792f — discard
**Source:** agent
**Change:** `HIDDEN_SIZE` 384 → 512 (0.543M → 0.917M params), following up on the 384 win to test the cuDNN-cliff theory.
**Result:** val_bpb 1.858794 (worse than 384's 1.839949), memory 1.3GB (down from 384's 1.9GB)
**Notes:** Confirms the cliff theory, at least partially: throughput recovered a lot (125.8M tokens / 2448 steps ≈ 419k tok/sec, vs. 384's ~180k tok/sec) and peak VRAM actually *dropped* below 384's despite more params — consistent with 384 hitting a slow, memory-hungry cuDNN kernel path while 512 (nicer for tensor cores) lands on a leaner, faster one. But more tokens seen (125.8M vs 53.6M) and more capacity still produced a slightly worse val_bpb than 384. So it's not simply "avoid the cliff and things improve" — 384's apparent advantage over 512 may itself be partly the "less data seen ⇒ less overfitting" effect from the previous entry, now working in 384's favor instead of against it. Net effect so far: 384 is still the best point found on this axis. Not chasing hidden_size further for now — moving to other levers (dropout, weight_decay, more layers) with 384 as the base.

## 49edaa4 — keep
**Source:** agent
**Change:** Added `self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))`, expanded per-batch and passed into `nn.RNN` instead of relying on the implicit zero-init.
**Result:** val_bpb 1.833233 (down from 1.839949), memory 1.9GB, throughput unchanged (53.6M tokens / 1045 steps — same cuDNN kernel path as plain hidden_size=384)
**Notes:** Small but real improvement (~0.0067) for 3 lines of code — good complexity/benefit ratio per program.md's simplicity criterion. Matches the program.md goal framing: a learned starting state can encode a prior about typical tune openings (e.g. anacrusis/pickup notes, common opening intervals) instead of forcing the model to infer "nothing has happened yet" from a zero vector every single sequence. New best: val_bpb 1.833233.
