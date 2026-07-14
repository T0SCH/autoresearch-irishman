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

## 4f5a853 — discard
**Source:** agent
**Change:** `DROPOUT` 0.0 → 0.1 on top of hidden_size=384 + learned h0 (retrying the dropout idea at a milder rate, reasoning that this base only gets through part of 1 epoch vs. baseline's 3, so overfitting pressure should be much lower).
**Result:** val_bpb 1.839526 (worse than current best 1.833233), memory 1.9GB
**Notes:** Still slightly worse, same as the 0.2 attempt at baseline. Two data points now both say dropout hurts in this fixed-time-budget setup — likely because the model here is throughput/data-limited rather than overfit-limited (partial epoch, not multiple), so dropout just removes useful signal without a matching overfitting benefit to offset it. Dropping this lever; not worth a third attempt at yet another rate.

## 7e4cea3 — keep
**Source:** agent
**Change:** `WEIGHT_DECAY` 0.0 → 0.01 (AdamW decoupled weight decay) on top of hidden_size=384 + learned h0.
**Result:** val_bpb 1.811846 (down from 1.833233), memory 1.9GB, throughput unchanged (53.5M tokens / 1044 steps)
**Notes:** Clear improvement (~0.021), unlike both dropout attempts which hurt. Makes sense given the data/throughput-limited hypothesis from the dropout notes: weight decay shrinks parameter norms directly rather than injecting training-time noise, so it doesn't cost useful gradient signal the way dropout does under a low-step-count regime — it's a much gentler regularizer here. New best: val_bpb 1.811846.

**Operational note (not an experiment):** Human pushed a `results.tsv`/`program.md` schema change directly to the branch (commit `2bcee80`, not via `human_input.md`) while the loop was running: `results.tsv` now has 7 columns (`num_steps`/`tokens_M` inserted between `memory_gb` and `status`), motivated by exactly the throughput-cliff confound flagged in the `10b1c9b`/`17b792f` entries above. Caused a rebase conflict on the `7e4cea3` ledger commit; resolved by rebasing the code commit cleanly onto the new tip and re-adding the ledger row in the new format (see `af2d1dd`, `866975b` — the first attempt at this lost the `experiment_log.md` entry in a `git reset --hard`, fixed in the second commit). Human then left a `human_input.md` note confirming the same thing after the fact — already handled by the time it was read, so no new experiment came out of it. All ledger entries from here on use the 7-column format.

## 1b6bde9 — keep
**Source:** agent
**Change:** Added `WARMUP_STEPS = 20` (linear LR warmup) followed by cosine decay of the LR down to 0, scheduled over wall-clock progress (`total_training_time / TIME_BUDGET`) rather than step count, applied every step before `optimizer.step()`. On top of hidden_size=384 + learned h0 + weight_decay=0.01.
**Result:** val_bpb 1.763448 (down from 1.811846), memory 1.9GB, throughput unchanged (53.5M tokens / 1044 steps)
**Notes:** Best result yet, and the biggest single-experiment improvement since the initial hidden_size jump (~0.048). Time-based (not step-based) scheduling was a deliberate choice: step count varies wildly across configs in this fixed-wall-clock setup (baseline ~8000 steps vs. this config's ~1044), so a step-count-tuned schedule wouldn't transfer. Standard technique, cheap (10 lines), clear win — good fit for the simplicity criterion. New best: val_bpb 1.763448.

## aa5f65e — discard
**Source:** agent
**Change:** `NUM_LAYERS` 2 → 3 on top of the current best (hidden_size=384 + learned h0 + weight_decay=0.01 + LR warmup/cosine).
**Result:** val_bpb 1.833008 (worse than 1.763448), memory 2.1GB (up from 1.9GB), throughput down further (36.0M tokens / 709 steps, vs. 53.5M / 1044 for the 2-layer version)
**Notes:** Extra layer adds more compute per step on top of an already throughput-limited config (hidden_size=384 already sits in the slow cuDNN kernel regime), so it saw even less data than the 2-layer run in the same 300s, and val_bpb got worse. Consistent with num_layers being a straightforward cost here rather than a useful hierarchical-structure win at this data/time budget — not chasing this further for now.

## cfd7f13 — keep
**Source:** human (idea via `human_input.md`: test `hidden_size=1024` to find out whether the throughput cliff at 384 is size-specific or a general "large = slow" pattern; also asked to always record num_steps/tokens_M regardless of outcome, and to check the loss curve for overfitting signs before discarding if val_bpb came out worse)
**Change:** `HIDDEN_SIZE` 384 → 1024 (0.543M → 3.396M params) on top of hidden_size=384's prior slot (still + learned h0 + weight_decay=0.01 + LR warmup/cosine).
**Result:** val_bpb 1.722063 (down from 1.763448 — new best), memory 2.5GB (up from 1.9GB), throughput 43.1M tokens / 844 steps (~144k tok/sec avg) — *lower* than 384's 53.5M/1044 (~178k tok/sec avg)
**Notes:** Answers the human's question: 1024 is not fast like baseline (~1.4M tok/sec) — it's in the same slow regime as 384, slightly slower even. So the cliff isn't "384 specifically unlucky," it looks like a general property of hidden_size past some threshold (probably cuDNN falling back off the persistent/fused RNN kernel path once weights don't fit whatever size/alignment class the fast path wants). More importantly: this result strengthens the case for a real capacity win, since 1024 improved val_bpb *while seeing fewer tokens* than 384 — the earlier confound ("bigger hidden_size wins partly because it's throughput-crippled into seeing less data, i.e. less overfitting") doesn't explain this one; more capacity beat less exposure this time. New best: val_bpb 1.722063. Following up with the human's optional sweep idea (320/448/576/640) to map the cliff boundary and find a better throughput/capacity trade-off point.

## 54058d8 — discard
**Source:** agent
**Change:** `HIDDEN_SIZE` 1024 → 1536 (3.396M → 7.447M params), testing whether the capacity trend from 384→1024 continues further up.
**Result:** val_bpb 2.176296 (much worse than 1.722063, back near baseline territory), memory 3.7GB (up from 2.5GB), only 468 steps / 23.5M tokens (vs. 1024's 844 steps / 43.1M)
**Notes:** Trend reversed hard. Params roughly doubled from 1024→1536 but steps nearly halved (844→468), so this run got severely undertrained relative to its size — top1_acc dropped from 0.6201 to 0.5368, the clearest sign of undertraining rather than overfitting (no train/val divergence pattern to speak of, just not enough updates for a model this large). This maps the top of the capacity curve found so far: somewhere between 1024 and 1536 is too big for a 300s budget at this throughput. 1024 remains the sweet spot.

## 402716f — discard
**Source:** agent
**Change:** `HIDDEN_SIZE` 1024 → 1280 (midpoint), trying to narrow the sweet spot between 1024's win and 1536's overshoot.
**Result:** val_bpb 2.007747 (worse than 1.722063), memory 3.1GB, 589 steps / 29.8M tokens (vs. 1024's 844 / 43.1M)
**Notes:** Surprisingly bad already — top1_acc 0.5625, closer to the 1536 failure (0.5368) than to 1024's 0.6201. The capacity curve falls off much faster right above 1024 than expected; 1024 isn't just "near the peak," it looks like it's sitting right at a fairly sharp optimum for this time budget/throughput regime. Not narrowing further (e.g. 1152) — treating hidden_size=1024 as settled and moving to other levers: this model is now much bigger than when weight_decay=0.01/LR schedule were tuned at hidden_size=384, so those may be worth revisiting, plus batch size, embed_size, and architectural tweaks (residual connections).

## 3f4ea49 — keep
**Source:** agent
**Change:** `WEIGHT_DECAY` 0.01 → 0.05 on top of hidden_size=1024 + learned h0 + LR warmup/cosine. 0.01 was originally tuned at hidden_size=384 (0.543M params); model is now 3.396M params (~6x).
**Result:** val_bpb 1.712790 (down from 1.722063), memory 2.5GB, throughput unchanged (43.1M tokens / 843 steps)
**Notes:** Small further improvement, confirming that the earlier weight_decay tuning didn't fully transfer to the larger model — makes sense since AdamW's decoupled decay scales with parameter count/norm, a fixed decay coefficient regularizes a 6x-bigger model relatively less unless increased to match. New best: val_bpb 1.712790.

## 90c60eb — keep
**Source:** agent
**Change:** `EMBED_SIZE` 128 → 256 on top of hidden_size=1024 + learned h0 + weight_decay=0.05 + LR warmup/cosine.
**Result:** val_bpb 1.692497 (down from 1.712790), memory 2.7GB, 816 steps / 41.6M tokens
**Notes:** Launched before a human `human_input.md` note arrived saying embed_size wasn't worth pursuing (reasoning: vocab is only 98 tokens, so embedding params are negligible next to the 1024×1024 recurrence matrices) — but the result came back a real win anyway (~0.02), on the same order as other kept changes. Parameter-count negligibility doesn't mean representational negligibility: a richer per-character embedding can still sharpen the input signal the RNN has to work with, independent of how small it is relative to total params. Keeping it since it empirically won, but not chasing this axis further (e.g. 384/512) per the human's steer — their broader point (focus on update-count-limited regime instead) still stands, this was just already in flight. New best: val_bpb 1.692497.

**Operational note (not an experiment):** Human left a `human_input.md` steer (received while this run was already in flight) reinterpreting the ledger: the failures above hidden_size=1024 (1280, 1536) show a clear undertraining signature (top1_acc falling 0.620→0.563→0.537 with no train/val divergence), meaning the binding constraint at this time budget is gradient-update count, not model capacity per se — at hidden_size=1024 the model only gets 844 updates with a fully-decayed cosine LR. Concluded the capacity sweep is exhausted (1024 stands) and steered the next experiment toward `BATCH_SIZE` 64→32 (roughly 2x the updates on similar total data, near-pure upside if GPU throughput holds), with a fallback to batch 48 if 32 proves too noisy. Also queued (not immediate): IRNN-style `nn.RNN(nonlinearity='relu')` + identity-initialized recurrent weights (Le/Jaitly/Hinton 2015) to fight tanh saturation over long ABC sequences — keep GRAD_CLIP=1.0, and a destabilized/NaN outcome there is a valid, loggable finding rather than a failure to hide. Explicitly said not to pursue further hidden_size sweeps or embed_size tuning.

## b1b2d1e — keep
**Source:** human (idea via `human_input.md`: reduce `BATCH_SIZE` to test whether the model is update-count-limited rather than data-limited at hidden_size=1024)
**Change:** `BATCH_SIZE` 64 → 32 on top of hidden_size=1024 + embed_size=256 + learned h0 + weight_decay=0.05 + LR warmup/cosine.
**Result:** val_bpb 1.690385 (down from 1.692497 — modest but real new best), memory 1.4GB (down sharply from 2.7GB), num_steps 1554 (vs. 816 — nearly doubled, as predicted), total_tokens_M 35.1 (vs. 41.6 — roughly the same order of magnitude, slightly lower)
**Notes:** Confirms the human's hypothesis directly: GPU throughput held up fine at the smaller batch size (steps nearly doubled while total tokens stayed in the same ballpark), and the extra updates translated into a real val_bpb improvement plus a solid top1_acc bump (0.6293→0.6351). Also a big incidental VRAM win (1.4GB vs 2.7GB) — useful headroom for future capacity experiments. Won cleanly, so no need for the batch_size=48 fallback. New best: val_bpb 1.690385.

## d1a3c35 — crash
**Source:** human (idea via `human_input.md`, queued after the batch_size test: IRNN recipe, Le/Jaitly/Hinton 2015)
**Change:** `nn.RNN(..., nonlinearity="relu")` instead of default tanh, plus `nn.init.eye_` on every `weight_hh_l*` (identity-initialized recurrent weights, `weight_ih` left at default init). `GRAD_CLIP=1.0` kept per the human's instruction. On top of current best (hidden_size=1024 + embed_size=256 + learned h0 + weight_decay=0.05 + LR warmup/cosine + batch_size=32).
**Result:** Crash — loss NaN/exploding within the first step(s), caught by the fast-fail check (no exception, no partial training).
**Notes:** The human explicitly flagged this as a possible, valid outcome rather than a hidden failure. Likely mechanism: with identity-initialized `weight_hh` and ReLU (which only clips negative values, unlike tanh's bounded output), the hidden state can pass through and accumulate largely unchecked across a single long ABC sequence during the *forward* pass — activations can blow up before gradient clipping ever gets a chance to act (clipping only bounds the backward-pass gradient norm, not forward activations). The original IRNN paper trained with a much smaller LR (orders of magnitude below AdamW-typical); this run used the existing LR=0.003 (with 20-step warmup), which was almost certainly too aggressive for this init/nonlinearity combination. Treating this as informative rather than retrying with a swept-down LR right now — noting it as a follow-up idea (e.g. LR ~1e-4 to 1e-5, or clamp/scale the identity init below 1.0) rather than pursuing immediately.

## e3f05ed — discard
**Source:** agent
**Change:** `BATCH_SIZE` 32 → 16, pushing the update-count-limited insight further since 64→32 won cleanly with VRAM headroom to spare.
**Result:** val_bpb 1.753826 (worse than 1.690385), memory 0.7GB (down further from 1.4GB), num_steps 2573 (up from 1554), total_tokens_M 25.0 (down from 35.1)
**Notes:** The trend reversed: despite even more gradient steps, val_bpb got worse. top1_acc actually ticked up slightly (0.6390 vs 0.6351 — the single-batch `quick_eval` metric is noisier and less reliable at this scale), but the ground-truth val_bpb disagrees. Total tokens seen dropped further (25.0M vs 35.1M), and each step's gradient estimate is noisier with only 16 samples — at some point noisier-but-more-frequent updates stop being a net win over fewer-but-more-reliable ones. batch_size=32 sits at the sweet spot on this axis; not going smaller.

## c30583f — crash
**Source:** agent (follow-up on d1a3c35's crash, testing the hypothesis that LR was the cause)
**Change:** Same IRNN recipe (relu + identity-init `weight_hh`) as `d1a3c35`, but `LEARNING_RATE` 0.003 → 0.0003 (10x lower), matching the much smaller LRs the original IRNN paper used.
**Result:** Crash again — loss NaN/exploding within the first step(s), same failure mode as before.
**Notes:** Rules out learning rate as the cause. Since it fails this early (before step 10, likely at/near step 0, still inside LR warmup where the effective LR is even smaller than 0.0003), the instability is almost certainly a *forward-pass* activation explosion — identity-init recurrent weights + ReLU (unbounded above, unlike tanh) can let the hidden state grow across the steps of a single long ABC sequence regardless of how small the optimizer's LR is, since no weight update has even happened yet when the first loss is computed. A smaller LR can't fix an already-unstable forward pass. This is now a settled negative result for this repo's `MAX_SEQ_LEN`/hidden_size combination — not retrying again without a different mitigation (e.g. scaling the identity init below 1.0, LayerNorm/activation clamping inside the recurrence, or truncated BPTT), none of which fit cleanly into the "everything is fair game but keep it simple" spirit right now. Dropping the IRNN idea.

## 40040ea — keep
**Source:** agent
**Change:** `LEARNING_RATE` 0.003 → 0.0015 (peak, before warmup/cosine schedule scales it) on top of hidden_size=1024 + embed_size=256 + learned h0 + weight_decay=0.05 + batch_size=32. The 0.003 value predates most of the current stack — it was last implicitly validated back at batch_size=64.
**Result:** val_bpb 1.658872 (down from 1.690385 — clear new best), memory 1.4GB, throughput essentially unchanged (35.4M tokens / 1572 steps)
**Notes:** Solid win, and the batch-size/LR scaling heuristic intuition paid off — top1_acc also jumped nicely (0.6351→0.6462). New best: val_bpb 1.658872. Worth bracketing with a higher LR (e.g. 0.006) too, to confirm 0.0015 isn't just "lower is always better" but an actual optimum.

## c0fe683 — discard
**Source:** agent
**Change:** `LEARNING_RATE` 0.0015 → 0.0008, testing whether lowering further continues to help.
**Result:** val_bpb 1.683150 (worse than 1.658872), memory 1.4GB, throughput unchanged (35.1M tokens / 1556 steps)
**Notes:** Worse than 0.0015, and worse than not lowering at all past that point — nicely brackets the optimum: 0.003 (worse) → 0.0015 (best) → 0.0008 (worse again). Treating LR≈0.0015 as settled for this stack.

## ce2bfc9 — keep
**Source:** human (idea via `human_input.md`: a larger, deliberately separate dataloader rework — length-bucketed/token-budget training batches instead of `make_dataloader`'s fixed batch size + random padding)
**Change:** Added `make_bucketed_dataloader` (train.py-only, `prepare.py` untouched) — sorts train tunes by length (with small random jitter, rejittered every epoch to avoid correlated same-length batches recurring), greedily packs into buckets until `items * max_len_in_bucket` hits `TOKEN_BUDGET=8192` (capped at `MAX_BUCKET_ITEMS=128`), pads only to each bucket's own max length. `TOKEN_BUDGET=8192` derived from median tune length (258) × 32 (the batch_size sweet spot). Only replaces the train loader — `val_loader`/`evaluate_bpb` keep the fixed `BATCH_SIZE=32` loader so val_bpb stays comparable across configs. On top of current best (hidden_size=1024 + embed_size=256 + learned h0 + weight_decay=0.05 + LR warmup/cosine peak=0.0015).
**Result:** val_bpb 1.637362 (down from 1.658872 — clear new best), memory 0.5GB (down sharply from 1.4GB), num_steps 3953 (vs. 1572 — ~2.5x more gradient updates in the same 300s), startup overhead ~25s (vs. baseline's typical ~25–34s, not meaningfully worse despite the one-time tune-encoding cost)
**Notes:** Measured *before* building anything, per the human's explicit instruction: random `batch_size=32` batching wastes ~59.2% of counted tokens on PAD (2.79M real / 6.85M total over 300 batches) — well above their ~20% "not worth it" threshold, so the full rework was justified rather than a guess. Standalone measurement of the bucketed scheme (before integration) showed padding waste falling to ~10.3%. Decisively confirms the human's core hypothesis: the "update-limited" regime found by the batch_size=64→32 experiment was partly self-inflicted by padding waste — cutting padding lets ~2.5x more real gradient updates fit in the same wall-clock budget, and that translated directly into the best val_bpb yet. The largest single-experiment win since the initial hidden_size jump. New best: val_bpb 1.637362.

## 791a0c4 — discard
**Source:** agent
**Change:** `TOKEN_BUDGET` 8192 → 4096, testing whether even more (smaller, real-token) updates helps under the now-low-padding bucketed loader.
**Result:** val_bpb 1.724212 (worse than 1.637362), memory 0.5GB (unchanged), num_steps 6094 (up from 3953)
**Notes:** Same pattern as the earlier `batch_size` sweep (64→32 helped, 32→16 hurt): there's a real-update-count sweet spot, and going past it into noisier/smaller-batch territory hurts val_bpb even with more steps — despite this being a different mechanism (shrinking the real-token budget vs. just shrinking batch size under heavy padding). top1_acc ticked up slightly (0.6624 vs 0.6615, same single-batch-noise caveat as before) but the ground-truth metric disagrees. TOKEN_BUDGET=8192 stands; testing 16384 next to check the other direction before calling it settled.

## 5ac5f32 — keep
**Source:** agent
**Change:** `TOKEN_BUDGET` 8192 → 16384 (doubled), checking the other bracket direction after 4096 hurt.
**Result:** val_bpb 1.617260 (down from 1.637362 — new best), memory 0.7GB (up slightly from 0.5GB), num_steps 2238 (down from 3953, roughly half as expected)
**Notes:** Surprising direction: unlike the `batch_size` sweep (where fewer/smaller-batch updates always hurt) and unlike 4096 (which also hurt), *larger* real-token batches with fewer, less-noisy updates won here. So the update-count-vs-noise tradeoff isn't monotonic in the same direction across every axis — under the bucketed loader specifically, bigger effective batches (with proportionally larger buckets across the whole length distribution, not just longer tunes) seem to give a better gradient signal-to-noise ratio than the extra step count from 8192 was worth. New best: val_bpb 1.617260. Testing even higher (32768) to see if the trend continues before settling.

## ff1eb40 — keep
**Source:** agent
**Change:** `TOKEN_BUDGET` 16384 → 32768 (doubled again), checking whether the larger-budget trend continues.
**Result:** val_bpb 1.614071 (down from 1.617260 — marginal new best), memory 1.4GB (roughly doubled from 0.7GB), num_steps 1400 (down from 2238)
**Notes:** Diminishing returns — the improvement here (~0.003) is much smaller than 8192→16384's (~0.02), while memory cost doubled and step count dropped further. Keeping it since it's a real (if small) win, but not chasing this axis further (e.g. 65536): a bigger, more promising lever just arrived from the human (truncated BPTT), pursuing that instead of squeezing more out of TOKEN_BUDGET.

## cf3c7b9 — keep
**Source:** human (idea via `human_input.md`: a second, deliberately separate dataloader rework — truncated BPTT with fixed short training windows, replacing the token-budget bucketing loader entirely rather than stacking on top of it)
**Change:** Added `make_windowed_dataloader`, replacing `make_bucketed_dataloader`. Cuts every train tune into consecutive fixed-length `TRAIN_SEQ_LEN=64` windows (x=window[:-1], y=window[1:]), pooled across all tunes and reshuffled every epoch, batched at `WINDOW_BATCH_SIZE=512` (chosen so batch_size×seq_len≈32768, matching the prior best's real-tokens-per-step). Stateless — no hidden-state carry across windows; the model's `forward()` already resets `h0` fresh every call, so no model code changed. `evaluate_bpb` and its internal loader (`prepare.py`, unmodified) plus `quick_eval`'s `val_loader` (`make_dataloader`, unmodified) both still evaluate on full sequences, so val_bpb stays comparable to every prior run in the ledger.
**Result:** val_bpb 1.526216 (down from 1.614071 — by far the largest single-experiment improvement in this worktree's history, ~0.088), memory 1.4GB (unchanged), num_steps 1463 (vs. bucketing's 1400 at similar real-tokens-per-step, as designed), total_tokens_M 47.6 (vs. 41.4 — more real tokens seen despite similar step count, since padding waste dropped further)
**Notes:** Motivation was explicit: a vanilla tanh RNN backpropagating through up to 1024 steps (the bucketed loader's whole-tune sequences) hits the pathological vanishing-gradient case — exactly the long-range dependency the assignment cares about — while also paying full compute for sequences whose tail gradient signal is already dead. Cutting to 64-token windows (well above the ~10-50 step effective memory a vanilla RNN actually has) removes that dead weight from both the gradient computation and the compute budget simultaneously. Measured padding with this scheme: ~10% (not the "quasi null" the human predicted — only each tune's own last, partial window pads — but still solidly below the ~59% random-batching baseline). The result decisively confirms the hypothesis: this is the single largest win found so far, larger even than the original hidden_size jump. New best: val_bpb 1.526216. Human flagged two natural follow-ups: (1) a STATEFUL variant carrying hidden state across a tune's own windows (detaching between them — true truncated BPTT, restoring cross-window context that the current stateless version discards), and (2) a `TRAIN_SEQ_LEN` sweep (32/128 — char-rnn used 50, the Karpathy blog post calls 50–100 near-equivalent).

## 86ed92e — discard
**Source:** agent (human-suggested sweep direction, agent-picked value)
**Change:** `TRAIN_SEQ_LEN` 64 → 32, `WINDOW_BATCH_SIZE` 512 → 1024 (kept batch_size×seq_len≈32768 constant, isolating window length from total real tokens/step).
**Result:** val_bpb 1.562154 (worse than 1.526216), memory 1.5GB, num_steps 1458 (essentially unchanged from 1463 — confirms the tokens/step control worked)
**Notes:** With step count and real-token throughput held equal, window length alone made the difference: 32 is too short to capture as much useful within-window dependency as 64, even though a vanilla RNN's effective memory is nominally in the same 10-50 range either window length exceeds. Testing 128 next to bracket the other direction before settling on 64.

## 928e2a9 — discard
**Source:** agent (human-suggested sweep direction, agent-picked value)
**Change:** `TRAIN_SEQ_LEN` 64 → 128, `WINDOW_BATCH_SIZE` 512 → 256 (kept batch_size×seq_len≈32768 constant, same isolation approach as the 32 test).
**Result:** val_bpb 1.551335 (worse than 1.526216), memory 1.4GB, num_steps 1456 (essentially unchanged, confirms tokens/step control)
**Notes:** Nicely brackets 64 as a local optimum on this axis: 32 (worse) → 64 (best) → 128 (worse). Consistent with a vanilla RNN's effective memory ceiling being somewhere around 64 — going longer doesn't add useful signal (the extra context beyond ~64 steps back is already inaccessible to the gradient) but does spend the same token budget on fewer, larger-window batches with less per-step diversity. `TRAIN_SEQ_LEN=64` settled. Moving to the human's other flagged follow-up: a STATEFUL variant of the windowed loader (carry hidden state across a tune's own windows, detach between them for true truncated BPTT, learned h0 only at each tune's real start) to restore the cross-window context this stateless version discards.

## fec4724 — discard
**Source:** human (idea via `human_input.md`: the second flagged follow-up to the windowed loader — carry hidden state across a tune's own windows instead of treating every window as independent)
**Change:** Added `make_stateful_windowed_dataloader` (`WINDOW_BATCH_SIZE=512` parallel lanes, each stepping through one tune's windows in order, yielding a `reset_mask`) and gave `CharRNN.forward` an optional `h0` kwarg plus a `self.last_hidden` (detached) side channel, so the training loop can thread hidden state across steps via `torch.where(reset_mask, learned_h0, carried_hidden)` without changing `forward`'s public return value (kept `evaluate_bpb`/`quick_eval` compatible, both untouched). True truncated BPTT: gradient horizon stays at 64, but state now persists across a tune's windows.
**Result:** val_bpb 1.530929 (vs. stateless's 1.526216 — essentially a wash, ~0.005 worse, within run-to-run noise), memory 1.4GB (unchanged), num_steps 1456 (unchanged)
**Notes:** No measurable win from restoring cross-window context. Plausible explanation the human anticipated: median tune length is ~258 tokens, i.e. only ~4 windows per tune at seq_len=64 — a vanilla RNN's gradient is truncated to 64 steps either way, so the *extra* history the stateful version's carried state could in principle offer (beyond what's already inside one 64-token window) is on the order of a few hundred additional tokens back, most of which a vanilla tanh RNN's ~10-50-step effective memory can't meaningfully use even when the state is technically present. The gain from cutting sequence length (1024→64) fixed the real problem (dead gradients); carrying state across already-short windows doesn't add much on top for tunes this short. Reverting cleanly per the human's own instruction. `TRAIN_SEQ_LEN=64`, stateless, remains the settled dataloader design — best: val_bpb 1.526216.

## 94ab015 — discard
**Source:** agent
**Change:** `HIDDEN_SIZE` 1024 → 1536, re-testing capacity now that the windowed loader fixes sequence length at `TRAIN_SEQ_LEN=64` regardless of hidden_size (unlike the old whole-tune loaders, where hidden_size and sequence length compounded).
**Result:** val_bpb 1.594133 (worse than 1.526216), memory 2.1GB, num_steps 719 (roughly half of 1024's 1456)
**Notes:** Confirms the hypothesis partially: no longer the catastrophic failure seen under whole-tune training (val_bpb 2.176296 at 468 steps back then) — this time it's a much gentler, ordinary worse-but-not-broken result. Step count still roughly halved though, since RNN per-step compute is dominated by the hidden-to-hidden matmul (scales with hidden_size²) regardless of sequence length — decoupling sequence length from hidden_size only removed one source of slowdown, not the other. `HIDDEN_SIZE=1024` stays settled. Not sweeping further on this axis for now — moving to other under-explored levers (embed_size, GRAD_CLIP, dropout, WARMUP_STEPS, WINDOW_BATCH_SIZE) that haven't been re-tested since the dataloader changed.
