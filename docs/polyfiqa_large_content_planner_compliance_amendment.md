# Compliance amendment to the Task 3 large content-planner screen

Frozen: 2026-08-05, before any inference from the substitute model.

Original preregistration SHA-256:
`d17255ae06fda1a2e819857a51f5c5e1dc664e19c3220c83521397b09dca11ba`.

The original comparison named `llama3.3:70b-instruct-q4_K_M`. Before that
model completed downloading and before it produced any target output, an exact
architecture audit found that the marketed 70B Llama has approximately 70.55B
untied parameters. The organizer email says a model "must not exceed 70 billion
total parameters" and explicitly counts total rather than active MoE
parameters. To avoid an eligibility risk, the Llama download was interrupted
and the model is removed from the competition experiment.

The prospective compliant substitute is the official
`Qwen/Qwen2-57B-A14B-Instruct-GGUF:Q4_K_M`: 57B total parameters and 14B active.
The model card and GGUF release both identify the total size and the official
GGUF card explicitly documents Ollama execution. Quantization does not change
the total parameter count.

No target prompt, demonstration, target selection, JSON schema, parser, metric,
baseline, or advance gate changes. The already observed 35B screen result is
not used to choose a prompt or output rule. Because the substitute was chosen
after that 35B result was known, its comparison is reported as a prospective
model-only replication, not as a fully independent hypothesis test.

If the 57B model fails the original gate, the branch stops. If it passes, it
may advance only to the original 154-row planner evaluation; it still cannot
become an end-to-end submission without a separately preregistered grounded
compiler evaluation.
