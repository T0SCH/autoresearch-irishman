# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 5b01fb6 — keep
**Source:** agent
**Change:** none — this is the unmodified `nn.RNN` baseline (`CharRNN` as committed, `RNN_TYPE="rnn"`, hidden_size=256, embed_size=128, num_layers=2, dropout=0.0, lr=0.003, weight_decay=0.0, batch_size=64, whole-tune padded loader via `make_dataloader`).
**Result:** val_bpb 2.087648, top1 0.5471, top5 0.8635, memory 1.4GB, 7898 steps, 407.0M tokens, 0.268M params
**Notes:** Matches `baseline-improve`'s reported baseline (2.093672) closely, as expected — same architecture, same data. This confirms the environment/setup is sound before the LSTM swap. Per worktree scope in program.md, the mandatory next experiment is swapping `nn.RNN` → `nn.LSTM` with everything else unchanged, to get a clean cell-type ablation pair.

## 27c841c — keep-prov
**Source:** agent
**Change:** `CharRNN.rnn`: `nn.RNN` → `nn.LSTM`, `RNN_TYPE` "rnn"→"lstm". Nothing else touched — same hidden_size=256, embed_size=128, num_layers=2, lr=0.003, whole-tune padded loader as the run-1 baseline.
**Result:** val_bpb 1.447489 (vs. 2.087648 baseline, −31% relative), top1 0.6762, top5 0.9317, memory 1.3GB, 1928 steps, 99.2M tokens, 0.959M params
**Notes:** Big, expected win — LSTM's gated cell state directly addresses the vanishing-gradient problem the assignment's long-range bar-line/meter structure needs. However `tokens_M` (99.2M) is only 24% of the run-1 median (407.0M), well under the `keep-prov` 60% threshold: at hidden_size=256 the LSTM's 4 gates cost ~4x an RNN's per-step recurrent compute (1928 vs 7898 steps, ratio 4.1x, matches theory), so part of this win could be "less overfitting from less data exposure" rather than a pure capacity/memory win. Marking `keep-prov` per program.md's mechanism; next experiment re-tests at a smaller `HIDDEN_SIZE` to roughly restore per-run token throughput and confirm the win holds at comparable exposure, before adopting the rest of `baseline-improve`'s transferable stack (windowed loader, learned (h0,c0), lr schedule, weight_decay, embed_size=256) and moving to the worktree's real target — long-sequence training.

## e980c7a — keep
**Source:** agent
**Change:** `HIDDEN_SIZE` 256→128 (LSTM cell unchanged from run 27c841c). Throughput-restoring re-test lever per the `keep-prov` mechanism — LSTM's 4 gates cost ~4x an RNN's per-step compute at the same hidden_size, so shrinking hidden_size is the right lever here (not `TRAIN_SEQ_LEN`, since no windowed/long-sequence loader is in play yet).
**Result:** val_bpb 1.441281 (slightly better than the 256-hidden run), top1 0.6914, top5 0.9424, memory 1.3GB, 6523 steps, 335.5M tokens (82% of the run-1 median — comfortably above the 60% threshold), 0.289M params
**Notes:** Confirms/promotes run 27c841c's `keep-prov`: the LSTM win is not a throughput-exposure artifact — at near-comparable token exposure it still beats the RNN baseline (2.087648) by a wide margin, and even improves slightly over the larger hidden_size, suggesting hidden_size=256 was already oversized for what this budget/loader can update in 300s. hidden_size=128 becomes the new working default going forward. Next: adopt `baseline-improve`'s transferable stack (windowed loader, learned (h0,c0), weight_decay=0.05, lr=0.0015 warmup+cosine, embed_size=256) — this worktree's actual point, per program.md's "Sequence-length stance" — is to move training to *long* sequences (TRAIN_SEQ_LEN 128/256/512) rather than inherit the RNN-tuned TRAIN_SEQ_LEN=64.
