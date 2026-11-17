# Frozen empirical source

This read-only scientific snapshot was extracted from private release `v1.3.1`,
commit `51cf883914f0dc30941fecb7f57b114ec10fee4b`. It preserves the code used by
`results/reviewer_prediction_release/wetlinks_prediction_release_manifest.json`.
Its seven recorded source hashes must continue to match that manifest exactly.
No empirical result or historical provenance hash was rewritten for R5.

The active implementation is the top-level `src/` tree. R5 changes fitted
synthetic prediction, adds parameter audits and corrects cause-incidence
normalization under active probability clipping. Do not use this snapshot for
new development or describe the frozen measured predictions as a rerun of the
updated implementation. Reproduction still requires lawful acquisition of the
source measurements and the original pinned environment.
