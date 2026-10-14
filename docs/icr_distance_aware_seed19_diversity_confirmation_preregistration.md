# Task 2 distance-aware seed-19 diversity confirmation preregistration

Date frozen: 7 August 2026, before seed-19 training or scoring. Team: ADML.

## Status and motivation

This is a new, transparently post-hoc-motivated diversity confirmation. It does
not amend or retroactively pass the original distance-aware experiment. That
experiment correctly stopped because `primary_replace_general_seed7` improved
only one source fold, below its advance gate. Its separately frozen
`secondary_general2_plus_distance_seed7` diagnostic nevertheless reached
179/253 correct versus 175/253 for the incumbent, with five rescues, one harm,
three improved folds, two tied folds, bootstrap P(gain)=0.91294, and p05=0.

The sole question here is whether the same equal-weight diversity effect
replicates with a second distance-aware training seed and remains present in a
fully symmetric four-model ensemble. This experiment may support a secondary
competition submission, but it cannot be reported as confirmation of the
original preregistered primary-replacement hypothesis.

## Frozen training configuration

Train seed 19 with exactly the seed-7 distance-aware configuration:

- base `cl-tohoku/bert-base-japanese-v3`, revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`;
- official 253-row JF-ICR public set and the existing five reconstructed-source
  folds;
- unchanged five-logit head;
- four epochs, batch size 8, maximum length 512, longest-first pair truncation;
- AdamW at 2e-5, weight decay 0.01, FP32;
- loss `cross_entropy + 0.5 * mean ordered-CDF squared error`;
- no class weights, resampling, external data, scheduler, early stopping,
  threshold search, or fold-dependent behavior.

Every fold must complete and persist aligned logits. Non-finite losses,
incomplete folds, input-hash drift, or provenance drift fail closed.

## Frozen profiles

All probabilities are averaged with equal weight, decoded by the existing
expected-score rounding rule, and then passed through the unchanged input-only
safe rules.

1. `incumbent_general2`: general seeds 7 and 19.
2. `replication_general2_plus_distance19`: both general seeds plus distance
   seed 19. This directly replicates the favorable seed-7 diversity profile.
3. `distance_two_seed`: distance seeds 7 and 19. Diagnostic only.
4. `final_general2_plus_distance2`: both general seeds and both distance-aware
   seeds. This is the sole promotable profile.

No alternative model weights, logits averaging, temperature, thresholds,
confidence routes, label-specific rules, changed-row intersection, or subset
selection may be evaluated after seed-19 labels are observed.

## Frozen confirmation and promotion gate

The seed-19 replication profile must:

- score strictly above the 175/253 incumbent;
- have rescues strictly exceed harms;
- improve at least two source folds and worsen at most one; and
- have source-cluster bootstrap P(gain) at least 0.80 with p05 at least zero.

The symmetric four-model profile is promotion-eligible only if it also:

- scores strictly above the incumbent;
- improves at least three of five source folds and worsens at most one;
- has at least two more rescues than harms;
- has source-cluster bootstrap P(gain) at least 0.90 and p05 at least zero; and
- uses no current or hidden test input, label, or answer.

Every condition in both gates must pass. A pass authorizes full-data distance
seed-7 and seed-19 refits and a policy-independent secondary portfolio lane.
The 0.691700 incumbent remains the primary unless the symmetric four-model
profile passes this complete gate. Failure is a stop: no third seed, loss
coefficient, weight, decoder, or routing follow-up is allowed on these labels.
