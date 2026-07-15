# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 699b4f0 — keep
**Source:** agent (program.md setup protocol — establish unmodified baseline as worktree ablation-pair anchor)
**Change:** none — ran train.py as-is (nn.RNN, embed128, hidden256, num_layers2, dropout0, lr0.003, wd0, grad_clip1, bs64, seq=MAX_SEQ_LEN1024, fp32).
**Result:** val_bpb 2.066833, memory 1.4GB, top1 0.5579, top5 0.8660, 8339 steps, 429.8M tokens, 0.268M params, 300.0s train / 326.2s total.
**Notes:** Baseline anchor for this worktree. Cross-worktree context from program.md: baseline-improve (nn.RNN) ended at 1.526216, lstm-improve (nn.LSTM) at 1.189357 — those are tuned-trajectory endpoints, this 2.066833 is the raw unmodified starting point (matches the ~2.09 starting points both prior worktrees report). GPU is Tesla T4 (Turing) → fp16+GradScaler is the correct precision path for later runs (no native bf16). Throughput baseline for keep-prov median: 429.8M tokens. Next experiment per worktree scope: raw nn.RNN→nn.GRU swap, identical config, to form the clean ablation pair (only cell type changed) — record explicitly separately from the tuning trajectory. Sample-check skipped (no sampler wired up yet).

## 2303e40 — keep-prov
**Source:** agent (program.md worktree scope — first experiment: raw nn.RNN->nn.GRU swap to form the clean ablation pair)
**Change:** train.py CharRNN.rnn: nn.RNN -> nn.GRU. RNN_TYPE "rnn"->"gru" (wandb tag). Everything else identical: embed128, hidden256, num_layers2, dropout0, lr0.003, wd0, grad_clip1, bs64, seq=MAX_SEQ_LEN1024, fp32.
**Result:** val_bpb 1.373228, memory 1.4GB, top1 0.6836, top5 0.9318, 2770 steps, 142.5M tokens, 0.729M params, 300.0s train / 311.9s total.
**Notes:** ABLATION PAIR (run-1 baseline 699b4f0 nn.RNN fp32 vs this nn.GRU fp32, only cell changed) — record explicitly for the write-up's three-way RNN/LSTM/GRU comparison. val_bpb 2.066833 -> 1.373228, a large drop, BUT throughput-confounded: GRU is ~3x slower/step on T4 fp32 (3 gates vs 1, dt ~100ms vs ~30ms, ~530k vs ~1.6M tok/s), so it saw 142.5M tokens = 33% of the keep/keep-prov median (429.8M), below the 60% keep-prov threshold -> status keep-prov. The val_bpb win is so large (0.69 bpb better at 1/3 the exposure) that a genuine GRU advantage is essentially certain, but the *magnitude* is not fair to quote until a comparable-throughput run confirms it. Per program.md the throughput lever for a cell-swap slowdown is NOT hidden_size (seq_len/hidden unchanged here) — it's precision: adopt fp16+GradScaler (the #1 transferable learning, T4 is Turing so fp16+scaler not bf16). Run 3 = GRU + fp16+GradScaler to restore throughput and promote this to keep. params 0.268M->0.729M (GRU 3 gates vs RNN 1, consistent). Sample-check skipped (no sampler wired up yet).
