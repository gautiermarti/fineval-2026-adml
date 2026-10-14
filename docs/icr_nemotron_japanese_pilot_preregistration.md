# Task 2 Japanese Nemotron judge pilot preregistration

Date frozen: 5 August 2026, before any FinEval judgment from this checkpoint.
Team: ADML.

## Hypothesis and evidential role

`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` is a genuinely Japanese-specialized
reasoning model from a different family than the previously screened Qwen
judges.  The hypothesis is that language specialization and family diversity
will improve direct application of the immutable JF-ICR guideline rubric.

This is a bounded feasibility screen, not a replacement-model evaluation.  A
pass authorizes a larger, separately frozen comparison or use as an independent
teacher.  It cannot promote over the 0.691700 supervised source-disjoint
primary from 22 labels.

## Frozen model and inference

- Checkpoint: `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`.
- Hugging Face revision: `3979dd16634988c34cc3bd911583c51e6a731d10`.
- Hugging Face safetensors total: 8,888,227,328 parameters.
- Serve the BF16 checkpoint with vLLM using `trust_remote_code`, the model's
  `nemotron_nano_v2_reasoning_parser.py`, reasoning parser
  `nemotron_nano_v2`, and `mamba_ssm_cache_dtype=float32`.
- Thinking mode: enabled.
- Sampling: temperature 0.6, top-p 0.95, seed 2026, maximum 2048 generated
  tokens.  One generation per item; no retry with another sampling mode.
- Require the existing strict JSON schema.  A transport failure may be retried
  byte-identically, but an invalid completed generation counts as invalid and
  incorrect.

## Frozen examples and prompt

Use the exact 22-row `select_pilot(frame, per_common_class=3)` selection already
used for Qwen:

`[25, 61, 86, 87, 90, 91, 95, 97, 107, 108, 109, 110, 117, 118, 126,
129, 157, 176, 189, 196, 209, 243]`.

Label counts are `-2:2, -1:11, 0:3, +1:3, +2:3`.  Use only the immutable
`guideline_prompt` and `STRUCTURED_SCHEMA` in `scripts/pilot_icr_judges.py`
(file SHA-256
`a44f6d9de16ddf4a2a998c6791b32f414ecdc6ed4be7ab87fd3306186d263b5c`).
No demonstrations, prompt rewrite, label prior, incumbent prediction, or source
metadata is supplied.

The frozen Qwen3.6-35B-A3B guideline comparator is read only from
`artifacts/icr_pilot/qwen3.6_35b-a3b.jsonl` (SHA-256
`d3d9aecea83e69e28dfae6f663c8d7275146b1675941533941199e9299f3b61e`).
On these exact IDs it has 11/22 correct: `-2:2/2, -1:5/11, 0:1/3,
+1:3/3, +2:0/3`.

## Frozen advance gate

Advance Japanese Nemotron beyond this screen only if every condition holds:

1. all 22 calls return schema-valid output with an allowed label;
2. at least 13/22 labels are correct, exceeding both the Qwen comparator and
   the pilot's 11/22 majority baseline by at least two rows;
3. at least 7/13 negative-label (`-2` or `-1`) rows are correct, so total gain
   cannot come by abandoning the rare boundary;
4. at least three of the five labels have nonzero recall; and
5. no current or hidden test input or answer is used.

Failure stops this model as a direct Task 2 judge.  It may remain a documented
architecture candidate, but no prompt track, thinking mode, seed, temperature,
demonstration set, or subset is changed after inspection.  Passing permits a
new preregistration for a larger source-aware evaluation; it does not itself
change the submission primary.
