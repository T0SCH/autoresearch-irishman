# autoresearch

This is an experiment to have the LLM do its own research. This run is scoped to a university deep learning assignment: train a char-level RNN to generate Irish folk tunes in ABC notation (IrishMAN dataset). Baseline starts as a plain `nn.RNN` (simplest option the assignment allows) — deliberately, to see how far it gets before reaching for LSTM/GRU.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AUFGABENSTELLUNG.md` — the actual assignment (submission requirements, bonus points, deadline). The autoresearch loop below only covers the training/architecture-search part of it, not evaluation notebooks or the write-up.
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, IrishMAN data download, char-level tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. RNN model (`CharRNN`), optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains the IrishMAN data and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row and commit it (tracked, not gitignored — we run on ephemeral cloud VMs, so the ledger needs to survive disconnects the same way the code does). The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Goal

Beyond "lower val_bpb" in the abstract, the human's current research direction for this run:

> Optimiere das LSTM so, dass es langfristige Rhythmen (wie die typischen 4/4- oder 6/8-Taktstrukturen von irischen Jigs und Reels) besser im Gedächtnis behält. Versuche Architektur-Anpassungen in `train.py`, um den Loss zu senken.

Concretely: `val_bpb` rewards good next-character prediction everywhere, but bar-line (`|`) placement and meter consistency (the `M:4/4` / `M:6/8` header vs. the actual bar lengths) are where long-range memory shows up most clearly in ABC notation. When judging whether an architecture change is working, don't just watch the aggregate `val_bpb` number — spot-check a few generated samples for whether bar structure stays coherent over a full tune, not just locally. The baseline is a plain `nn.RNN`, which is known to struggle with exactly this kind of long-range dependency (vanishing gradients) — if `val_bpb`/sample quality plateaus and bar structure still falls apart on longer tunes, swapping `nn.RNN` for `nn.LSTM` or `nn.GRU` in `CharRNN` is the most direct next lever, not just a hyperparameter tweak. Other ideas worth trying: larger `hidden_size` (more memory capacity), more `NUM_LAYERS` (hierarchical structure), tuning `DROPOUT`, or architectural tweaks (e.g. a learned initial hidden state instead of zero-init, residual/skip connections between layers). This section reflects the human's current framing — if they redirect the goal, update this section rather than the loop mechanics below.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: `CharRNN` architecture (including swapping `nn.RNN` for `nn.LSTM`/`nn.GRU`), `NUM_LAYERS`, `HIDDEN_SIZE`, `EMBED_SIZE`, `DROPOUT`, optimizer (AdamW, SGD, ...), learning rate / schedule, `GRAD_CLIP`, batch size, training loop, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     1024.3
total_tokens_M:   38.6
num_steps:        589
num_params_M:     0.847
num_layers:       2
hidden_size:      256
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	1.0	keep	baseline
b2c3d4e	0.993200	1.0	keep	increase hidden_size to 512
c3d4e5f	1.005000	1.0	discard	switch optimizer to SGD
d4e5f6g	0.000000	0.0	crash	hidden_size 4096 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv, and `git add results.tsv` along with it — unlike the original upstream design, we run on ephemeral cloud VMs that can disconnect anytime, so the ledger needs the same push-on-keep protection as the code (see step 8). Amend it into the same commit rather than creating a separate one.
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit — then `git push origin <branch>` (`-u` on the first push). The machine running training may be ephemeral (e.g. a cloud/Colab VM); anything not pushed is lost the moment the session dies, only the pushed history survives.
9. If val_bpb is equal or worse, you git reset back to where you started — nothing to push, the commit never left local history

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
