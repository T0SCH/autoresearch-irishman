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
