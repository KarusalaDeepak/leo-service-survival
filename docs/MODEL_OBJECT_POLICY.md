# Model-object policy

This repository releases source code, frozen configuration, tests, text/CSV
artifacts, and hashes. It does not release executable fitted model objects.

Forbidden public-release extensions include `.pkl`, `.pickle`, `.joblib`,
`.onnx`, `.ubj`, and `.model`. A reviewer should refit the pipelines from the
cited input data with `scripts/build_prediction_release_artifacts.py`. The
release manifest records the exact input and source-file hashes used for the
published derivative.

This policy has two purposes: loading an untrusted serialized model can execute
code, and an opaque binary does not document the endpoint, split, imputation,
calibration, or score conventions. In-memory serialization used only to measure
object size is not a released model and does not create a reusable artifact.

