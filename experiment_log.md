# Experiment Log

One entry per experiment (kept, discarded, or crashed), appended by the loop's ledger-commit step. See program.md's "Logging results" section for the entry format.

## 5b01fb6 — keep
**Source:** agent
**Change:** none — this is the unmodified `nn.RNN` baseline (`CharRNN` as committed, `RNN_TYPE="rnn"`, hidden_size=256, embed_size=128, num_layers=2, dropout=0.0, lr=0.003, weight_decay=0.0, batch_size=64, whole-tune padded loader via `make_dataloader`).
**Result:** val_bpb 2.087648, top1 0.5471, top5 0.8635, memory 1.4GB, 7898 steps, 407.0M tokens, 0.268M params
**Notes:** Matches `baseline-improve`'s reported baseline (2.093672) closely, as expected — same architecture, same data. This confirms the environment/setup is sound before the LSTM swap. Per worktree scope in program.md, the mandatory next experiment is swapping `nn.RNN` → `nn.LSTM` with everything else unchanged, to get a clean cell-type ablation pair.
