# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 1eaa8e2 — keep
**Source:** agent (initial baseline, per program.md setup)
**Change:** none — train.py as-is: plain `nn.RNN`, embed_size=128, hidden_size=256, num_layers=2, dropout=0.0, AdamW lr=3e-3, batch_size=64.
**Result:** val_bpb 2.093672, memory 1.4GB
**Notes:** Establishes the baseline this worktree (RNN-only scope) will iterate from. top1_acc 0.5528, top5_acc 0.8652, 8180 steps, 421.5M tokens in 300s train time. Next experiments to try per program.md goal: larger hidden_size, more layers, dropout tuning, learned initial hidden state, LR schedule — no cell-type swap (that's lstm-improve/gru-improve's job).
