# autoresearch

This is an experiment to have the LLM do its own research. This run is scoped to a university deep learning assignment: train a char-level RNN to generate Irish folk tunes in ABC notation (IrishMAN dataset). Baseline starts as a plain `nn.RNN` (simplest option the assignment allows) — deliberately, to see how far it gets before reaching for LSTM/GRU.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag> dev`. From `dev`, not `master` — `master` still mirrors upstream's original GPT/Muon template, none of this repo's actual IrishMAN/RNN work.
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

Concretely: `val_bpb` rewards good next-character prediction everywhere, but bar-line (`|`) placement and meter consistency (the `M:4/4` / `M:6/8` header vs. the actual bar lengths) are where long-range memory shows up most clearly in ABC notation. When judging whether an architecture change is working, don't just watch the aggregate `val_bpb` number — bar-line (`|`) placement and meter consistency are the real long-range-memory signal. **Mechanism (mandatory)**: every 10th kept run, before starting the next experiment, generate 3 samples (temp 0.8, seed from a val tune, ~512 tokens each) from the current model and append a one-sentence bar-structure/meter verdict to that run's `experiment_log.md` entry. A kept run past the 10-keep threshold with no verdict blocks the next experiment — this is mechanism, not prose. The baseline is a plain `nn.RNN`, which is known to struggle with exactly this kind of long-range dependency (vanishing gradients) — if `val_bpb`/sample quality plateaus and bar structure still falls apart on longer tunes, swapping `nn.RNN` for `nn.LSTM` or `nn.GRU` in `CharRNN` is the most direct next lever, not just a hyperparameter tweak. Other ideas worth trying: larger `hidden_size` (more memory capacity), more `NUM_LAYERS` (hierarchical structure), tuning `DROPOUT`, or architectural tweaks (e.g. a learned initial hidden state instead of zero-init, residual/skip connections between layers). This section reflects the human's current framing — if they redirect the goal, update this section rather than the loop mechanics below.

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
top1_acc:         0.5820
top5_acc:         0.8833
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

Two files record every attempt, kept or not: `results.tsv` (terse, machine-parseable table) and `experiment_log.md` (narrative log with more detail). Both are tracked in git — see "Committing the ledger" below for why they need their own commit, separate from the experimental code commit.

`results.tsv` is tab-separated, NOT comma-separated (commas break in descriptions). Header row and 7 columns:

```
commit	val_bpb	memory_gb	num_steps	tokens_M	status	description
```

1. git commit hash (short, 7 chars) — the code commit this row describes, even if that commit was later reset away (see below)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. num_steps — from train.py's final printout, use 0 for crashes
5. tokens_M — total_tokens_M from train.py's final printout, use 0.0 for crashes
6. status: `keep`, `keep-prov`, `discard`, or `crash` (see `keep-prov` rule below)
7. short text description of what this experiment tried

Columns 4/5 exist because TIME_BUDGET is fixed (300s), not step count — two runs can differ hugely in how much data they saw in that window (e.g. a slower architecture might do 1000 steps where a faster one does 8000). A val_bpb "win" from a run that saw far fewer tokens may just be less overfitting from less exposure, not a real improvement. **Mechanism (`keep-prov`)**: before keeping a run, compare its `tokens_M` to the median `tokens_M` of all prior `keep`/`keep-prov` rows. Below 60% of that median → status `keep-prov` (provisional), note the confound in the `experiment_log.md` entry, and re-test the same idea at a smaller `HIDDEN_SIZE` (or whichever lever restores throughput) as the very next experiment. A `keep-prov` promotes to `keep` only once a comparable-throughput follow-up confirms the win; if the follow-up discards, the `keep-prov` row stays in the ledger as history but the code reverts to the last real `keep`. Mention any `keep-prov` that survived to end-of-session in the one-pager.

Example:

```
commit	val_bpb	memory_gb	num_steps	tokens_M	status	description
a1b2c3d	0.997900	1.0	8000	410.0	keep	baseline
b2c3d4e	0.993200	1.0	1100	55.0	keep	increase hidden_size to 512
c3d4e5f	1.005000	1.0	8000	410.0	discard	switch optimizer to SGD
d4e5f6g	0.000000	0.0	0	0.0	crash	hidden_size 4096 (OOM)
```

`experiment_log.md` is one entry per experiment, appended (never edited/rewritten), most recent last:

```
## <commit-hash> — <status>
**Source:** agent | human (<one-line summary of what the human asked for, if this experiment came from `human_input.md`>)
**Change:** what in train.py went from what to what
**Result:** val_bpb <value>, memory <value>GB
**Notes:** anything worth remembering — why it did/didn't work, if known
```

`Source` matters for the write-up (which ideas were the agent's own vs. steered by the human).

## Receiving input from the human mid-loop

`human_input.md` (repo root, gitignored, not part of the ledger) is a mailbox. The human may edit it at any time — while you're mid-experiment, between experiments, whenever — since you're not watching the chat continuously during an overnight run, this is the reliable channel, not a chat message that might arrive while you're deep in a 5-minute run.

Check it at the start of every loop iteration (step 0 below), before touching `train.py`. If it doesn't exist or is empty, there's nothing to do, carry on to step 1.

If it exists and has content:
- **Content is exactly `STOP` (case-insensitive, ignoring surrounding whitespace)**: this is not a tip, it's an instruction to end the autonomous loop. Do not start a new experiment. If you're reading this between experiments (the normal case, since you only check at iteration boundaries), you're already done — finish any in-progress ledger commit for the last experiment if you haven't, then stop and summarize what happened across the session for the human. Do not discard/reset the last kept state to do this.
- **Any other content**: treat it as a human-supplied idea or steer for your *next* experiment (this iteration's step 2). Fold it into what you try. Log it with `Source: human (<summary>)` in `experiment_log.md` once you've acted on it. Clear the file (empty it, or delete it) after reading, so you don't re-process the same note forever.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

0. Check `human_input.md` (see above) — this can end the loop before you start another experiment.
1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit (train.py only — not the ledger files, they come later, see step 8)
4. Run the experiment in the background, don't block on it: `uv run train.py > run.log 2>&1 &` (or your environment's background-execution mechanism for Bash) — redirect to a file, do NOT use `tee` or otherwise stream the output, that floods your context for no benefit. Don't poll it every few seconds either, same reason: the budget is fixed at 5 minutes plus a few seconds of startup/eval overhead, so there's nothing to gain from checking early. Either use whatever "notify me when this finishes" / scheduled-wakeup mechanism your environment offers, or a single check-back around the 05:10 mark. If your environment gives you idle time while it runs, that's fine to spend thinking about the *next* experiment, not on watching this one.
5. Sync wandb, regardless of what step 4 did — offline runs live under `wandb/offline-run-*` and are otherwise as ephemeral as the VM itself (unlike the ledger, `wandb/` is gitignored, nothing protects it from a disconnect). Attempt each not-yet-synced run individually, with a timeout so a stuck sync can't stall the loop the way the HF Xet hang once did:
   ```bash
   for d in wandb/offline-run-*; do
     [ -d "$d" ] || continue
     if timeout 30 uv run wandb sync "$d"; then rm -rf "$d"; fi
   done
   ```
   Only delete a run's local directory once its own sync succeeded. A failed or timed-out sync just leaves that directory in place — the same loop picks it up again next iteration, so nothing needs retry bookkeeping.
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Decide keep or discard/crash, then act in this order:
   - **Keep**: leave the step-3 commit in place. `git push origin <branch>` (`-u` on the first push).
   - **Discard or crash**: `git reset --hard` back to the commit from before step 3 — this throws away the train.py change, on purpose.
   - **Either way, now append to `results.tsv` and `experiment_log.md`** describing what just happened, and commit *only these two files* as a fresh commit on top of wherever HEAD ended up (the kept train.py commit, or the clean pre-experiment state after a reset). Push this commit too.

### Committing the ledger separately (why step 8 is ordered this way)

The ledger commit must never be bundled into the step-3 experimental commit, and must always happen *after* the keep/discard decision — otherwise a discard's `git reset --hard` deletes its own ledger entry along with the code, and you lose the record that the attempt ever happened. Committing the ledger update as its own commit, always last, means:
- A **kept** experiment: two commits land — the code change, then the ledger update.
- A **discarded/crashed** experiment: the code commit is reset away entirely; only the ledger commit survives, so `results.tsv`/`experiment_log.md` still show the attempt even though train.py itself shows no trace of it.

Both cases push, so the ledger survives disconnects the same way kept code does (we run on ephemeral cloud VMs — see the push note in step 8).

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until stopped. You are autonomous. If you run out of ideas on your own: re-read the in-scope files, re-sweep axes prior runs only sampled once (LR/grad-clip/warmup often shift after a dataloader change), and combine prior near-misses. Architecture *ideas* — new dataloaders, new cell types, structural changes — are the human's job and arrive via `human_input.md`; your job is to evaluate and sweep around them. Don't promise "reading papers" you won't read. The loop runs until `human_input.md` tells you `STOP` (see above) — that is the sanctioned way to end it, not you deciding you've done enough.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
