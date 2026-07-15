# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 699b4f0 — keep
**Source:** agent (program.md setup protocol — establish unmodified baseline as worktree ablation-pair anchor)
**Change:** none — ran train.py as-is (nn.RNN, embed128, hidden256, num_layers2, dropout0, lr0.003, wd0, grad_clip1, bs64, seq=MAX_SEQ_LEN1024, fp32).
**Result:** val_bpb 2.066833, memory 1.4GB, top1 0.5579, top5 0.8660, 8339 steps, 429.8M tokens, 0.268M params, 300.0s train / 326.2s total.
**Notes:** Baseline anchor for this worktree. Cross-worktree context from program.md: baseline-improve (nn.RNN) ended at 1.526216, lstm-improve (nn.LSTM) at 1.189357 — those are tuned-trajectory endpoints, this 2.066833 is the raw unmodified starting point (matches the ~2.09 starting points both prior worktrees report). GPU is Tesla T4 (Turing) → fp16+GradScaler is the correct precision path for later runs (no native bf16). Throughput baseline for keep-prov median: 429.8M tokens. Next experiment per worktree scope: raw nn.RNN→nn.GRU swap, identical config, to form the clean ablation pair (only cell type changed) — record explicitly separately from the tuning trajectory. Sample-check skipped (no sampler wired up yet).
