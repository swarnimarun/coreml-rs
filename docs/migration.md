# Migrating from v1

v2 is a breaking API change. The v1 API and Swift bridge remain in the previous history.

- Replace `CoreMLModelWithState::new(...).load()` with `Model::compile(...)` followed by `compiled.load(...)`, or `Model::load(...)` for an existing `.mlmodelc`.
- Replace mutable `add_input` calls with a per-prediction `Inputs` map. Use `Tensor` and `MultiArray` for arrays, or image and scalar feature constructors where appropriate.
- Replace `ComputePlatform` with `ComputeUnits` in `ModelOptions`. `All` permits Core ML to choose among devices; it does not guarantee a placement.
- Replace `predict_with_retry` with caller-owned retry and fallback policy using structured errors. See `examples/tensors.rs`.
- Keep a `CompiledModel` alive through `CompiledModel::load`, or persist the artifact and load it from a stable location. The old implicit buffer persistence and cache behavior are gone.
- Replace the raw-byte model loader with async `Model::load_from_memory` on macOS 13 or later. For an ML Program with external weights, pass the specification and blob mapping to `Model::load_from_memory_with_blobs` on macOS 15 or later. The Swift `ffi` module and experimental engine switch have no v2 equivalents. `ndarray` conversion is available through an optional feature.

Model, input and output native objects are thread-confined in the safe API. The metrics observer is thread-safe. See [Usage](usage.md) for lifetimes, native interop and availability.
