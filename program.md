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

Concretely: `val_bpb` rewards good next-character prediction everywhere, but bar-line (`|`) placement and meter consistency (the `M:4/4` / `M:6/8` header vs. the actual bar lengths) are where long-range memory shows up most clearly in ABC notation. When judging whether an architecture change is working, don't just watch the aggregate `val_bpb` number — bar-line (`|`) placement and meter consistency are the real long-range-memory signal. **Mechanism (mandatory)**: every 10th kept run, before starting the next experiment, generate 3 samples (temp 0.8, seed from a val tune, ~512 tokens each) from the current model and append a one-sentence bar-structure/meter verdict to that run's `experiment_log.md` entry. A missing verdict does NOT block the next experiment — if no sampler is wired up yet, log a one-line 'sample-check skipped' note in that run's `experiment_log.md` entry and move on. The point is the verdict when it's cheap, not a hard gate that could stall the loop. The baseline is a plain `nn.RNN`, which is known to struggle with exactly this kind of long-range dependency (vanishing gradients) — if `val_bpb`/sample quality plateaus and bar structure still falls apart on longer tunes, swapping `nn.RNN` for `nn.LSTM` or `nn.GRU` in `CharRNN` is the most direct next lever, not just a hyperparameter tweak. Other ideas worth trying: larger `hidden_size` (more memory capacity), more `NUM_LAYERS` (hierarchical structure), tuning `DROPOUT`, or architectural tweaks (e.g. a learned initial hidden state instead of zero-init, residual/skip connections between layers). This section reflects the human's current framing — if they redirect the goal, update this section rather than the loop mechanics below.

**Worktree scope**: swap `nn.RNN` for `nn.GRU` in `CharRNN` as your first experiment (baseline run is still the unmodified `nn.RNN` per the setup protocol below). After that, **stay on `nn.GRU` — do NOT swap to `nn.LSTM`, bidirectional, plain `nn.RNN`, or any other cell type.** That is what the parallel `lstm-improve` / `bidirectional-improve` / `architecture-search` worktrees are for; one fixed cell type per worktree is exactly what makes the cross-worktree ablation comparison valid. Your mandate to originate your own architectural ideas (see "Driving your own research") operates *within the GRU family* — gating tweaks, layer structure, dataloader, initialization, optimizer, LR schedule, hidden_size, embed_size are all fair game; the GRU cell-type identity of this worktree is the one thing that is fixed.

## Transferable learnings from baseline-improve and lstm-improve

Two prior worktrees already ran 33 experiments each: `baseline-improve` (plain `nn.RNN`, val_bpb 2.093672 → **1.526216**) and `lstm-improve` (`nn.LSTM`, val_bpb 2.087648 → **1.189357**, 22% better than the RNN). Their full `results.tsv`/`experiment_log.md` are the source of truth. **`lstm-improve` is the more relevant prior** — GRU and LSTM are both gated (unlike vanilla RNN), so most of its findings are a much better starting prior than RNN's. Use both to skip already-answered questions; don't re-run either worktree's dead ends from scratch.

**Adopt immediately as your starting stack — high-confidence, cell-type-agnostic wins:**
- **fp16 mixed precision** (`torch.autocast(dtype=torch.float16)` + `torch.amp.GradScaler`, with `scaler.unscale_(optimizer)` called *before* `clip_grad_norm_` — get this order right, it's easy to clip on the still-scaled gradients by mistake). This was `lstm-improve`'s single highest-leverage discovery: ~2.5x throughput with no accuracy tradeoff, and it's what made a fair `hidden_size` capacity comparison possible at all (see below). Purely a systems win, has nothing to do with cell type — build it in from the start instead of rediscovering it after several step-starved `keep-prov` dead ends like `lstm-improve` did. **Verify your GPU is Turing (e.g. T4) before defaulting to fp16+GradScaler over bf16** — Turing has no native bf16; if you land on an Ampere+ GPU (A100/L4), bf16 without a scaler is the better/simpler choice.
- **Weight tying** (share the embedding and output-head weight matrix). Free win in both compute and params in `lstm-improve`, no reason to expect otherwise here.
- **Stateful windowed loader as your first loader** (carry hidden state across a tune's own consecutive windows, truncated-BPTT gradient horizon per window, `reset_mask` at tune boundaries). Implement the *stateful* variant directly — stateless windowing (no carry across windows) was independently a wash on both the RNN and the LSTM, so there's no need to re-derive and test that intermediate step a third time; compare your stateful loader straight against the non-windowed (whole-tune) baseline instead. **But don't assume the stateful win itself transfers** — GRU has no separate cell state (LSTM's `C_t = f_t⊙C_{t-1} + i_t⊙C̃_t` additive-memory path isn't present in GRU; its update gate `z_t⊙h_{t-1} + (1-z_t)⊙h̃_t` is structurally similar but not identical), so it's plausible cross-window carry helps GRU less than it helped LSTM. Treat the loader choice as implemented-by-default but the *result* as still an open question to confirm.
- **Residual/skip connections between layers.** Small but real win for LSTM at 2 layers, and specifically what made a 3rd layer viable (plain `NUM_LAYERS=3` without residual was untested on LSTM, but `NUM_LAYERS=1` without residual was clearly worse than 2 — the capacity-per-layer curve is steep, residual is what let `lstm-improve` push deeper safely). Worth building in from the start rather than discovering it mid-sweep.
- **Methodology, non-negotiable:** one variable per experiment. Both prior worktrees independently hit the same failure mode — bundling multiple pre-validated changes into a single "adopt the whole stack" experiment (`baseline-improve`'s early ledger-schema tangent aside, `lstm-improve`'s `5fc6b30` bundled 5 changes and regressed, un-diagnosable, had to be reverted and re-tested one variable at a time). Adopting the *items above* as a starting point is fine (they're each independently confirmed), but any *new* idea gets its own isolated experiment.

**Skip or deprioritize — settled negatives on the only other gated cell tested:**
- **The IRNN recipe** (ReLU + identity-init) — vanilla-RNN-specific vanishing-gradient hack, irrelevant to any gated cell, don't even consider it.
- **`NUM_LAYERS=1`** — confirmed worse than 2 on both RNN and LSTM (lacks capacity), don't retest at the bottom of this axis.
- **Time-based LR warmup+cosine schedule** — this is the one genuine surprise: it was `baseline-improve`'s *second-biggest* win on the RNN, but `lstm-improve` isolated it and found it made things *worse* than constant LR. Don't assume "gated = behaves like LSTM" here either — GRU is its own architecture — but given it already failed on the one other gated cell tested, this is low-priority: worth one quick isolated check, not worth defending or re-tuning if it doesn't help immediately.
- **`GRAD_CLIP` above 1.0** — confirmed not binding (clipping rarely triggers) on both RNN and LSTM; unlikely to matter here either, don't spend an experiment on it unless something else suggests instability.

**Re-tune fresh, don't copy the number — cell-type-specific unknowns:**
- **`hidden_size` capacity ceiling.** `nn.LSTM` has 4 gates (`weight_ih`/`weight_hh` shaped `4*hidden`), `nn.GRU` has 3 (`3*hidden`) — meaningfully cheaper per step than LSTM at the same `hidden_size`, so **expect GRU's throughput-limited ceiling to sit higher than LSTM's confirmed 256** (LSTM's own ceiling took real bracketing to find — 384 was tested and lost — do the same bracketing here, don't assume a specific number, just don't be surprised if it's not 256). This is exactly the kind of thing `keep-prov` exists for: build fp16 in from the start so you're not fighting the same throughput-starved `keep-prov` limbo `lstm-improve` spent several experiments escaping.
- **`TRAIN_SEQ_LEN` (window length)** — LSTM settled at 256 (bracketed against 128/512). Reasonable starting point, but re-bracket rather than assume it transfers exactly.
- **`LEARNING_RATE`** — LSTM's bracket moved *down* as `hidden_size` grew (0.008 at hidden=128 → 0.003 at hidden=256), opposite direction from the RNN's own tuning. Re-tune fresh at whatever `hidden_size` you settle on; don't inherit either prior worktree's number.
- **`DROPOUT`** — LSTM found `0.1` a clean, repeatable win (confirmed twice, at two different `hidden_size` values); this is the one axis where LSTM and RNN gave *opposite* answers (RNN: dropout hurt 4x). Worth testing early since it's cheap and the two priors disagree — don't assume either direction, let GRU's own result decide.
- **`WEIGHT_DECAY`** — LSTM confirmed `0.05` a small isolated win (never bundled). Reasonable starting value, cheap to re-confirm once, not worth an extended sweep given how marginal it was there too.

**For the ablation write-up (bonus point):** you now have a three-way comparison forming — RNN 1.526216, LSTM 1.189357, GRU (yours). The cleanest single ablation pair is still "identical config, only the cell swapped" (your run-1 baseline vs. run-2 raw GRU swap) — record that pair explicitly, separately from your own tuning trajectory, same as `lstm-improve` did.

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

`results.tsv` is tab-separated, NOT comma-separated (commas break in descriptions). Header row and 10 columns:

```
commit	val_bpb	top1_acc	top5_acc	memory_gb	num_steps	tokens_M	num_params_M	status	description
```

1. git commit hash (short, 7 chars) — the code commit this row describes, even if that commit was later reset away (see below)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes. Stays the primary keep/discard signal.
3. top1_acc — from train.py's final printout (e.g. 0.6462), use 0.0 for crashes. The assignment's headline metric, logged every run. NOTE: single-batch `quick_eval` number, a noisy rough signal only — see the top1/top5 caveat below.
4. top5_acc — same source and caveat as top1_acc, use 0.0 for crashes
5. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
6. num_steps — from train.py's final printout, use 0 for crashes
7. tokens_M — total_tokens_M from train.py's final printout, use 0.0 for crashes
8. num_params_M — from train.py's final printout (e.g. 3.540), use 0.0 for crashes. Essential context for any capacity/hidden_size comparison — especially across cell types, where an LSTM cell is ~4x an RNN cell's params at the same hidden_size.
9. status: `keep`, `keep-prov`, `discard`, or `crash` (see `keep-prov` rule below)
10. short text description of what this experiment tried

**top1/top5 caveat (matters for the one-pager, not the loop):** the per-run top1_acc/top5_acc are `quick_eval`'s single val-batch numbers — fine as a rough per-run signal (they track val_bpb closely, r≈-0.96 across baseline-improve), but NOT the assignment-grade metric, which wants top-1/top-5 "over all sequences in the test set." Compute that once, aggregated over the full val split, on the final chosen model, for the write-up — don't quote the noisy single-batch ledger numbers as your reported result. Keep making keep/discard calls on val_bpb (the aggregated ground-truth metric); top1/top5 in the ledger is for tracking, not decisions.

Columns 6/7 (num_steps/tokens_M) exist because TIME_BUDGET is fixed (300s), not step count — two runs can differ hugely in how much data they saw in that window (e.g. a slower architecture might do 1000 steps where a faster one does 8000). A val_bpb "win" from a run that saw far fewer tokens may just be less overfitting from less exposure, not a real improvement. **Mechanism (`keep-prov`)**: before keeping a run, compare its `tokens_M` to the median `tokens_M` of all prior `keep`/`keep-prov` rows. Below 60% of that median → status `keep-prov` (provisional), note the confound in the `experiment_log.md` entry, and re-test the same idea with whichever lever restores throughput as the very next experiment. The throughput lever depends on what changed: a vanilla-RNN run that grew `HIDDEN_SIZE` should re-test at smaller `HIDDEN_SIZE`; an LSTM/GRU run on long sequences that grew `TRAIN_SEQ_LEN` (or window size) should re-test at smaller `TRAIN_SEQ_LEN`/`WINDOW_BATCH_SIZE` — `HIDDEN_SIZE` is rarely the right lever there. A `keep-prov` promotes to `keep` only once a comparable-throughput follow-up confirms the win; if the follow-up discards, the `keep-prov` row stays in the ledger as history but the code reverts to the last real `keep`. Mention any `keep-prov` that survived to end-of-session in the one-pager.

Example:

```
commit	val_bpb	top1_acc	top5_acc	memory_gb	num_steps	tokens_M	num_params_M	status	description
a1b2c3d	0.997900	0.5820	0.8833	1.0	8000	410.0	0.847	keep	baseline
b2c3d4e	0.993200	0.5910	0.8890	1.0	1100	55.0	3.396	keep	increase hidden_size to 512
c3d4e5f	1.005000	0.5790	0.8810	1.0	8000	410.0	0.847	discard	switch optimizer to SGD
d4e5f6g	0.000000	0.0	0.0	0.0	0	0.0	0.0	crash	hidden_size 4096 (OOM)
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
- **Any other content**: an *optional* steer from the human — a hint, a direction, an idea. Fold it into your next experiment (this iteration's step 2); it overrides your own pick for that round. But it is a bonus, not your lifeline: you generate your own directions every round now (see "Driving your own research" below), so an empty mailbox is the normal case, not a problem to solve. Log it with `Source: human (<summary>)` in `experiment_log.md` once you've acted on it, and clear the file after reading so you don't re-process the same note forever.

## Driving your own research (every round)

The biggest wins in this project did NOT come from greedy one-knob sweeps — they came from stepping back and reframing the problem (spotting that padding waste was inflating apparent "update starvation," that a vanilla RNN backpropagating over 1024 steps was self-inflicting vanishing gradients, that the fixed-time budget makes throughput the hidden variable behind most "wins"). Historically those reframes arrived from outside via `human_input.md`. **That is now your job, every round — not the human's.** You are a researcher, not a hyperparameter-sweep script.

Each round, before picking a change and during the ~5 min of idle time while the run trains (loop step 4):

1. **Reflect (bird's-eye, every round).** Re-read the *whole* `results.tsv`/`experiment_log.md`, not just the last row. Name the current binding constraint (throughput? capacity? gradient flow? regularization? update count?) and the mechanism behind the last few results. When a result surprises you, that surprise is the most valuable thing on the table — chase *why* it happened, don't just log the number and reach for the next knob.

2. **Research actively.** Validate what you're seeing against real sources and mine them for techniques you haven't tried. If you have web search/fetch tools, USE them — reference implementations (e.g. `karpathy/char-rnn`, folk-rnn, nanoGPT-style repos), blog posts, papers on char-level RNNs, truncated BPTT, LSTM/GRU cells, initialization, optimizers, regularization. Cite the actual URL you read in the `experiment_log.md` entry.
   - **Honesty guard (hard rule):** cite only what you actually fetched. If you have no web access, or a fetch fails, say so plainly in the log and reason from the in-scope files + first principles instead. NEVER fabricate a citation, invent what "the literature says," or report a result you didn't get — a made-up reference is far worse than none, it poisons every decision built on it.

3. **Turn it into your own concrete improvement — architectural changes included.** New dataloaders, cell-type changes (within this worktree's scope), structural tweaks, optimizer swaps: all fair game and all *yours* to originate now. You are no longer restricted to sweeping around ideas the human hands you. Keep the discipline though: **one change per experiment**, and the simplicity criterion still holds.

Time-box it — one or two targeted lookups that actually inform the next concrete change, inside the run's idle window. Don't stall the loop reading for an hour, and don't research in the abstract. The rhythm is: reflect → research → hypothesis → one change → run → keep/discard → reflect.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

0. Check `human_input.md` (see above) — this can end the loop before you start another experiment.
1. **Orient.** Look at the git state (current branch/commit), then do the bird's-eye reflection from "Driving your own research" above — re-read the *whole* ledger, name the binding constraint, and decide what the *data* says to try next, not just the next unswept knob. Every round, not only round 1.
2. Pick ONE concrete idea — informed by step 1's reflection and your own research, architectural changes included — and implement it by directly hacking `train.py`. One change per experiment.
3. git commit (train.py only — not the ledger files, they come later, see step 8)
4. Run the experiment in the background, don't block on it: `uv run train.py > run.log 2>&1 &` (or your environment's background-execution mechanism for Bash) — redirect to a file, do NOT use `tee` or otherwise stream the output, that floods your context for no benefit. Don't poll it every few seconds either, same reason: the budget is fixed at 5 minutes plus a few seconds of startup/eval overhead, so there's nothing to gain from checking early. Either use whatever "notify me when this finishes" / scheduled-wakeup mechanism your environment offers, or a single check-back around the 05:10 mark. Spend the idle time on "Driving your own research" above — reflecting on the full ledger and actually researching the next move — not on watching this run.
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

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to run *indefinitely* until stopped — fire and forget. You are autonomous, and that now means autonomous in *ideas*, not just execution: generating the next direction — architectural reframes and new techniques from your own research (see "Driving your own research" above) included — is your job, not something you wait on `human_input.md` to supply. If you feel out of ideas, you have not reflected or researched hard enough: re-read the full ledger for a pattern you haven't named, look up how reference implementations handle the constraint you're stuck on, re-sweep axes that shift after a structural change (LR/grad-clip/warmup often move after a dataloader change), and combine prior near-misses. The loop runs until `human_input.md` tells you `STOP` (see above) — that is the only sanctioned way to end it, not you deciding you've done enough.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
