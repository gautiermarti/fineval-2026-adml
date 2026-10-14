# Task 1 all-released-labels fairness-control amendment

Date: 5 August 2026. Team: ADML.

This amendment was written while the preregistered train+test pilot was still
running, after epoch 4 and before its final checkpoint or promotion audit. The
original preregistration remains identified by SHA-256
`b0860cf10cc28631499bf5d419111e0b525fe40c8301fef1039585f7c97e8026`.

## Reason

The persisted historical legal-base checkpoint reports five fully unaligned
training rows, whereas the current corrected aligner retains one of those rows
as partially alignable. The new pilot reports four fully unaligned original
training rows. Although this is only one row, comparing the new model solely to
the old checkpoint would confound extra public labels with the alignment-code
revision.

## Additional required control

Run `spyrosbriakos/greek_legal_bert_v2` with exactly the same current code,
seed 7, eight epochs, batch size 8, learning rate 3e-5, and held-out validation
evaluation, but train on the original `train` split only. The public-label
pilot may advance only if:

1. it still passes every original preregistered gate against the historical
   checkpoint;
2. its macro Entity F1 is strictly above this same-code train-only control;
3. at least two entity types are nonnegative versus the same-code control;
4. no type falls by more than 0.01 versus the same-code control.

This amendment only adds a harder attribution check; it does not loosen any
threshold. Current/hidden FinEval data remains excluded. If the control removes
the apparent gain, the public-label hypothesis is rejected.
