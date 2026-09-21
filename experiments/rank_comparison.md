# LoRA rank comparison

Portable evidence and plots were exported on 2026-09-21:
[results index](results/README.md), [comparison CSV](results/rank_comparison.csv),
[validation loss](results/validation_loss.svg), and
[training resources](results/training_resources.svg).
Report links below use the portable copies, with metric values and response
text preserved and local artifact paths normalized. Original checkpoints and
reports remain in checkpoint/. No new training or inference was run.

## Completed rank-8 baseline

Reviewed on 2026-09-19. Sources:

- [Configuration](results/runs/lora_r8_seed42_baseline_v1/config.json)
- [Training metrics](results/runs/lora_r8_seed42_baseline_v1/metrics.json)
- [Independent evaluation and responses](results/runs/lora_r8_seed42_baseline_v1/evaluation.json)

Independent reload evaluation exactly reproduces the saved validation metrics.
Both models have six nonempty responses with matching prompt IDs. The saved
report key `lora.sample` was corrected to `lora.samples` without changing any
metric or response or rerunning inference.

| Metric | Base | LoRA r=8 |
| --- | ---: | ---: |
| Validation loss | 1.6708821270 | 1.3724845943 |
| Perplexity | 5.3168558715 | 3.9451406022 |

Training used 1,146,880 trainable parameters and 250 optimizer updates in
410.569 seconds. Peak allocated/reserved GPU memory was 3.545957/4.433594 GiB.
Timing excludes model loading, validation, and checkpoint serialization.
Validation supervises the entire formatted sequence, including the prompt.
Lower validation loss does not directly establish better answer quality.

Both models used the same six prompts, greedy decoding, input limit 256, and
128 new tokens. Qualitative observations:

| Prompt | Base | LoRA r=8 |
| --- | --- | --- |
| sky_explanation | Explains scattering, but ends mid-sentence | Completes the explanation |
| library_summary | Omits the next-Monday start date | Identical omission |
| polite_rewrite | Reverses the request into a promise to submit | Preserves request, deadline, and incorrect numbers, but adds an unnecessary inquiry |
| meeting_extraction | Correct fields inside a Markdown fence; fails JSON-only requirement | Correct standalone JSON |
| pencil_arithmetic | Correct answer 7, but two sentences | Correct answer 7 in one sentence |
| packing_list | Meets the list requirements | Meets the list requirements |

These are illustrative examples, not a benchmark or aggregate quality score.
Prompts and review criteria: [eval_prompts.json](../configs/eval_prompts.json).

## Completed rank-4 training and evaluation

The student completed `checkpoint/lora_r4_seed42_baseline_v1` on 2026-09-19.
Its saved configuration differs from rank 8 only in rank, alpha, and trainable
parameter count. Initial validation metrics match exactly.

| Metric | LoRA r=4 | LoRA r=8 |
| --- | ---: | ---: |
| Validation loss | 1.3779733255 | 1.3724845943 |
| Perplexity | 3.9668539534 | 3.9451406022 |
| Average training micro-batch loss | 1.5289745948 | 1.5021426765 |
| Trainable parameters | 573,440 | 1,146,880 |
| Training seconds | 497.130 | 410.569 |
| Peak allocated GPU memory (GiB) | 3.541455 | 3.545957 |
| Peak reserved GPU memory (GiB) | 4.427734 | 4.433594 |

Read-only artifact checks passed: epoch 1, 250 optimizer updates, matching
optimizer/scheduler progress, 112 finite adapter tensors, and exact adapter
weight equality with the full checkpoint. Final learning rate is zero as
expected. Sources: [config](results/runs/lora_r4_seed42_baseline_v1/config.json)
and [metrics](results/runs/lora_r4_seed42_baseline_v1/metrics.json).

Rank 8 has slightly lower validation loss in these runs (difference 0.005489).
Rank 4 halves the trainable parameter count but saves only 4.609375 MiB of
peak allocated memory. Frozen model weights and activations still require
memory. The single rank-4 run took longer; these timings alone do not establish
that lower rank causes slower training.

Independent [rank-4 evaluation](results/runs/lora_r4_seed42_baseline_v1/evaluation.json)
passed: loss/perplexity exactly match the training metrics. Metadata and adapter
path identify rank 4, alpha 8. Validation settings, generation settings, and
prompts match rank 8; Base metrics and all six Base responses are identical
across the two reports. Each model has six nonempty responses in prompt order.

Rank-4 versus rank-8 qualitative review:

- Sky explanation: both finish the explanation, with slightly different wording.
- Library summary: both omit the next-Monday start date.
- Polite rewrite: both keep the request and Friday deadline but add an
  unnecessary inquiry. Rank 8 states explicitly that the numbers are incorrect;
  rank 4 asks for updated numbers without explicitly stating the error.
- JSON extraction, arithmetic, and packing list: the rank-4 and rank-8 responses
  are exactly identical. JSON parses to the required fields; the list has the
  required three lines. The arithmetic answer is 7 in one sentence.

These six examples do not show a clear overall quality advantage for either
rank. Rank 8 has slightly lower full-sequence validation loss in this single-seed
comparison; this does not prove a general or statistically significant advantage.

## Completed rank-16 training and evaluation

Verified on 2026-09-20. Sources:
[config](results/runs/lora_r16_seed42_baseline_v1/config.json),
[metrics](results/runs/lora_r16_seed42_baseline_v1/metrics.json), and
[evaluation](results/runs/lora_r16_seed42_baseline_v1/evaluation.json).

| Metric | r=4 | r=8 | r=16 |
| --- | ---: | ---: | ---: |
| Validation loss | 1.3779733255 | 1.3724845943 | 1.3678385521 |
| Perplexity | 3.9668539534 | 3.9451406022 | 3.9268538260 |
| Trainable parameters | 573,440 | 1,146,880 | 2,293,760 |
| Training seconds | 497.130 | 410.569 | 253.339 |
| Peak allocated GPU memory (GiB) | 3.541455 | 3.545957 | 3.554959 |
| Peak reserved GPU memory (GiB) | 4.427734 | 4.433594 | 4.443359 |

Rank 16 used alpha 32; other saved training settings match ranks 4 and 8.
Average training micro-batch loss was 1.4817326024. The completed run has epoch
1, 250 updates, final learning rate zero, and matching optimizer/scheduler
progress. All 112 adapter tensors are finite, contain 2,293,760 parameters, and
exactly match the corresponding full-checkpoint weights.

Independent rank-16 evaluation exactly matches its training validation metrics.
Prompts and evaluation settings match the earlier runs; Base metrics and all
six Base responses are identical. Each model has six nonempty responses in
the expected order. Rank 16's validation loss is 0.004646 below rank 8's, with
9.21875 MiB more peak allocated GPU memory. The shorter recorded training time
does not establish that higher rank causes faster training; these are individual
runs without repeated timing or controlled machine-load measurements.

Qualitative review versus rank 8:

- Library summary, JSON extraction, arithmetic, and packing-list responses are
  exactly identical. The summary still omits the next-Monday start date.
- The sky explanation is shorter and complete, but does not explicitly explain
  the shorter-wavelength dependence requested by the review criteria.
- The polite rewrite keeps Friday and the incorrect-number issue but still
  introduces an unnecessary inquiry about additional requirements or changes.

Validation loss decreases across the three tested ranks at seed 42, but these
six examples do not establish a clear overall answer-quality improvement.
The rank-32 results and final comparison are recorded below.

## Completed rank-32 run and final single-seed comparison

Verified on 2026-09-20. Sources:
[config](results/runs/lora_r32_seed42_baseline_v1/config.json),
[metrics](results/runs/lora_r32_seed42_baseline_v1/metrics.json), and
[evaluation](results/runs/lora_r32_seed42_baseline_v1/evaluation.json).

| Metric | Base | r=4 | r=8 | r=16 | r=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation loss | 1.670882 | 1.377973 | 1.372485 | 1.367839 | 1.365197 |
| Perplexity | 5.316856 | 3.966854 | 3.945141 | 3.926854 | 3.916495 |
| Trainable adapter parameters | 0 | 573,440 | 1,146,880 | 2,293,760 | 4,587,520 |
| Training seconds | N/A | 497.130 | 410.569 | 253.339 | 253.521 |
| Peak allocated GPU memory (GiB) | N/A | 3.541455 | 3.545957 | 3.554959 | 3.572965 |
| Peak reserved GPU memory (GiB) | N/A | 4.427734 | 4.433594 | 4.443359 | 4.458984 |

Rank 32 uses alpha 64; other saved training settings match the previous runs.
It completed epoch 1 with 250 updates, training micro-batch loss 1.4667545394,
and final learning rate zero. Checks passed for 112 finite adapter tensors,
exact full-checkpoint adapter equality, and optimizer/scheduler step 250.
Independent evaluation exactly matches training validation metrics. Evaluation
settings, prompts, Base metrics, and all Base responses match earlier runs.

Rank 32's sky explanation, JSON, arithmetic, and list match rank 16 exactly.
JSON, arithmetic, and list match across all four ranks. Rank 32 does preserve
the library announcement's next-Monday start date, unlike the other runs.
Its rewrite keeps Friday but still adds an unnecessary inquiry and asks for
updated numbers without explicitly stating the original error.

Findings and limits:

- Validation loss decreases across the four ranks in this setup. Successive
  absolute reductions are 0.005489, 0.004646, and 0.002642: diminishing gains.
- Rank 4 to 32 increases trainable parameters eightfold while reducing loss
  by about 0.93% and perplexity by 1.27%. Base-to-rank-4 loss reduction is
  0.292909, versus 0.012776 for rank 4 to 32.
- Peak allocated memory grows only 32.265625 MiB from rank 4 to 32. Adapter
  parameter count is not total training memory. Single-run timing differences
  do not establish causal speed advantages.
- Six prompts show output similarity plus a specific summary improvement;
  they do not establish general answer-quality improvement.
- Full-sequence loss includes prompt tokens. Greedy decoding can emit the
  same tokens even if their probabilities differ, so loss and visible text
  need not change together.
- Lower ranks may already capture much of the useful adaptation for these
  2000 examples and one epoch. This is a hypothesis, not a proven capacity
  limit. One seed does not establish statistical significance.

The first single-seed rank experiment is complete.

## Controlled comparison

All four runs used the explicit multiplier `alpha / rank = 2`.
This does not guarantee equal update magnitudes
or identical optimization across ranks.

| Rank | Alpha | Expected trainable parameters | Status |
| ---: | ---: | ---: | --- |
| 4 | 8 | 573,440 | Training and independent evaluation complete |
| 8 | 16 | 1,146,880 | Completed baseline |
| 16 | 32 | 2,293,760 | Training and independent evaluation complete |
| 32 | 64 | 4,587,520 | Training and independent evaluation complete |

A has shape `[r, in_features]` and B has shape `[out_features, r]`, giving
`r * (in_features + out_features)` parameters per bias-free LoRA projection.
For this Qwen model, q_proj is 1024 -> 2048 and v_proj is 1024 -> 1024,
across 28 blocks: `28 * r * (1024 + 2048 + 1024 + 1024) = 143360 * r`.
Implementation checks should derive counts from actual projection dimensions.

Keep the saved 2000/200 split and order, seed 42, max length 512, full-sequence
labels, q/v targets, bf16, batch size 1, accumulation 8, one epoch,
AdamW lr 1e-4, weight decay 0.01, clipping 1, and linear schedule with
250 updates and 25 warmup updates fixed. Use identical evaluation settings.
Each run starts from the original base model with a fresh adapter and optimizer,
and saves to its own directory; do not resume from r=8.

Record validation loss/perplexity, trainable parameters, training time, GPU
peaks, and fixed-prompt responses. This is initially a single-seed comparison;
conclusions apply to these settings, not to rank universally.

## Completed answer-only evaluation

The optional Dataset mask is implemented with full-text token offsets; the
default remains full-sequence supervision. Checks on all 200 validation rows
confirmed unchanged inputs and attention masks, 26,587 valid shifted targets,
and no zero-target rows. There are 199 complete answer suffixes with EOS and
one truncated answer. Dynamic padding preserves the valid-target count.
The existing Dataset/Collator/Evaluation tests passed (34 tests, 2026-09-20).

All four answer-only reports are present and passed consistency checks:

| Model | Answer-only loss | Answer-only perplexity | Report |
| --- | ---: | ---: | --- |
| Base | 1.4025100728 | 4.0653915994 | Identical in all four reports |
| r=4 | 1.2791102662 | 3.5934410970 | [r4](results/runs/lora_r4_seed42_baseline_v1/evaluation_answer_only.json) |
| r=8 | 1.2763340948 | 3.5834789233 | [r8](results/runs/lora_r8_seed42_baseline_v1/evaluation_answer_only.json) |
| r=16 | 1.2734473694 | 3.5731493202 | [r16](results/runs/lora_r16_seed42_baseline_v1/evaluation_answer_only.json) |
| r=32 | 1.2722962533 | 3.5690385769 | [r32](results/runs/lora_r32_seed42_baseline_v1/evaluation_answer_only.json) |

Metadata, adapter paths, prompts, and evaluation settings match the respective
full-sequence reports. All six responses per model are exactly unchanged.
Base metrics and responses match across all answer-only reports. Every saved
perplexity matches exp(loss). These are saved-report checks, not fresh inference
reruns by the reviewer.

Answer-only loss also decreases across ranks, but rank 4 to 32 improves it by
only 0.53% with eight times the trainable parameters. Adjacent absolute gains
are 0.002776, 0.002887, and 0.001151; unlike the full-sequence results, these do
not strictly diminish at every doubling. Rank 32 improves answer-only loss by
9.28% versus Base. This supports a reference-answer prediction improvement
without establishing an equivalent free-generation quality gain. Compare ranks
within the same metric; absolute full-sequence and answer-only losses have
different target sets. All adapters were trained with full-sequence supervision.

## Next step

Rank-independent count checks are implemented in train.py and smoke_train.py.
Meta-device checks using the local Qwen configuration passed for ranks 4/8/16/32,
including actual injected parameter counts and trainable names; these checks
did not execute forward or backward passes.

The current rank evaluations, CLI/configuration entrypoints, initial setup
documentation, planned regression tests, and portable result exports are
complete. `summarize_results.py` regenerates the CSV and plots from JSON only.
Next organize the local repository for review, then consider the separate
held-out generation review described in AGENTS.md. Fresh-environment setup,
pinned model/data revisions, and broader quality evidence remain incomplete.
Do not automatically launch new training.

The student launches training manually. No new training run was started while
preparing this plan.
