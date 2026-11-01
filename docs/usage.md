# Usage

## Loading and lifetime

`Model::compile` compiles an `.mlmodel` or `.mlpackage` to a temporary `CompiledModel`. `CompiledModel::load` keeps that artifact alive for as long as the model needs it. Use `persist` to copy it to a new directory for later use. `Model::load` opens an existing `.mlmodelc`; keep that directory available for the lifetime of the model. Loading and compilation are synchronous and may take time.

Set `ModelOptions::compute_units` to request the CPU, CPU and GPU, CPU and Neural Engine, or all compute units. Core ML chooses actual placement. A display name helps identify a model in Instruments on macOS 13 or later. Named functions require macOS 15 or later. `description()` reports feature names, types, default array shapes and shape constraints, and default image dimensions.

### Loading model bytes

On macOS 13 or later, use `Model::load_from_memory` to load the bytes of a self-contained `.mlmodel` specification. On macOS 15 or later, use `Model::load_from_memory_with_blobs` for an ML Program whose specification refers to external weights. Supply each referenced blob using its model-relative path. For example, FastViT uses `weights/weight.bin`:

```rust,no_run
use coreml_rs::{Model, ModelOptions};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let legacy_bytes = std::fs::read("model.mlmodel")?;
let legacy = Model::load_from_memory(legacy_bytes, ModelOptions::default()).await?;

let specification = std::fs::read("MyModel.mlpackage/Data/com.apple.CoreML/model.mlmodel")?;
let weights = std::fs::read("MyModel.mlpackage/Data/com.apple.CoreML/weights/weight.bin")?;
let program = Model::load_from_memory_with_blobs(
    specification,
    [("weights/weight.bin", weights)],
    ModelOptions::default(),
).await?;
# let _ = (legacy, program);
# Ok(())
# }
```

Both methods consume their byte buffers and retain them for the model's lifetime. A program may refer to several blobs, each of which must be supplied. The input is a model specification and its blobs, not a zipped `.mlpackage` or compiled `.mlmodelc`. The crate does not stage the supplied data to a file; Core ML controls its own internal caching. Dropping a load future does not cancel the native load, and its buffers stay alive until Core ML releases the callback. These methods return `Error::Unsupported` on older macOS versions. The crate does not promise to erase model bytes from memory after use.

`Capabilities::current().in_memory_model` and `in_memory_blobs` report runtime availability. Core ML can reject particular models even when the APIs are available; see [known limitations](roadmap.md). The example in `examples/in_memory.rs` uses `futures-executor`, while library callers may use their existing async executor.

## Predictions and storage

`Inputs` holds named `FeatureValue`s. Construct scalar values directly, use `MultiArray::from_tensor` for a typed Rust tensor, or use `FeatureValue::image_from_rgba8` for tightly packed RGBA pixels. Check the model's input description and apply its required preprocessing. Outputs retain their native Core ML objects; use `as_string`, `as_dictionary` or `multi_array` to access them. `to_tensor` copies the actual data in logical order, including strided outputs.

`Inputs::prepare` retains a native feature provider for repeated calls. `MultiArray::update` reuses an array's allocation; all clones see updates. `PredictionOptions::with_output` proposes a reusable output array. Core ML may ignore the proposal, so check `is_same_object` on the returned array. `predict_batch` uses Core ML's batch API. `examples/tensors.rs` shows these patterns. Enable the `ndarray` feature for copying to and from ndarray views, including non-contiguous views.

`MultiArray::with_contiguous` exposes a scoped, read-only native buffer without copying. `with_contiguous_mut` allows scoped mutation. Both are unsafe because the caller must synchronize all aliases, Core ML calls, CPU threads and GPU commands during the callback. `MultiArray::from_pixel_buffer` and `metal::MetalTextureCache` support sharing a compatible pixel buffer as a Metal texture. The caller is responsible for device compatibility and synchronization, and must retain the texture guard until GPU work finishes. This path does not turn every Core ML array into an `MTLBuffer`, and Core ML may choose its own backing. See the Metal integration test for a concrete example.

## State, encryption and fallback

On macOS 15 or later, `new_state` creates state for a model that declares state features; `predict_with_state` takes a mutable state tied to its creating model. For a multifunction asset, `load_function_sharing_state` loads another named function and permits the same state only when Core ML reports matching state names, shapes and data types. Each function has separate metrics. `predict_prepared_with_state` reuses a prepared input provider for stateful calls. Do not share a state between concurrent predictions. The test suite includes small stateful and multifunction accumulators; validate the state behavior of your own model as well.

```rust,no_run
use coreml_rs::{Inputs, Model};

# async fn run(decode: &Model, prefill_inputs: &Inputs, decode_inputs: &Inputs) -> coreml_rs::Result<()> {
// `decode` is already loaded with ModelOptions::function_name = Some("decode").
let prefill = decode.load_function_sharing_state("prefill").await?;
let mut state = decode.new_state()?;
let prepared = prefill_inputs.prepare()?;
let _ = prefill.predict_prepared_with_state(&prepared, &mut state)?;
let _ = decode.predict_with_state(decode_inputs, &mut state)?;
# Ok(())
# }
```

An encrypted `.mlmodelc` prepared using Apple's model encryption workflow is loaded through `Model::load` or `CompiledModel::load`. Key retrieval belongs to Core ML. Native error domain, code and description are preserved; `NativeError::is_decryption_key_fetch` and `is_decryption` identify the documented Core ML error codes. This project does not create encrypted models or accept raw keys. See [Apple's model encryption documentation](https://developer.apple.com/documentation/coreml/encrypting-a-model-in-your-app).

The caller controls fallback and retries. Inspect the error and choose whether to retry or use another loaded model or compute configuration. The tensor example shows one such policy. Do not assume every error is transient.

## Observability and concurrency

`model.metrics()` snapshots started, completed, failed and in-flight calls, total elapsed time, batch item count, and the time spent loading the model. It also reports the last start and completion times, idle time, and p50/p95/p99 latency for up to the 256 most recent finished requests. Failed calls are included in latency; a batch is one request, with its item count reported separately. Clones of a `Model` share metrics, while separately loaded instances have independent metrics. A cloned `MetricsHandle` is `Send + Sync`, so another thread can observe a running call. Metrics cover calls through this model handle, including input conversion; direct native calls are not counted. They are not hardware utilization measurements.

On macOS 14.4 or later, `model.compute_plan().await` returns an owned snapshot of Core ML's anticipated execution plan for the model's load configuration. Neural-network layers and ML Program operations can report a preferred device and supported devices; program operations can also report a relative estimated cost. These values may be unavailable and do not measure elapsed time, current load or actual per-request placement. Pipelines and nested program blocks are represented in the snapshot. Core ML may spend time loading a plan, so request it outside a latency-sensitive prediction path. Dropping the future does not cancel Core ML's native work.

```rust,no_run
use coreml_rs::{Model, PlanStructure};

# async fn observe(model: &Model) -> coreml_rs::Result<()> {
let plan = model.compute_plan().await?;
if let PlanStructure::Program { functions } = plan.structure {
    for function in functions {
        for operation in function.block.operations {
            println!("{}: {:?}, cost {:?}",
                operation.operator_name, operation.device_usage, operation.estimated_cost);
        }
    }
}
let metrics = model.metrics();
println!("in flight: {}, idle: {:?}, latency: {:?}",
    metrics.in_flight, metrics.idle_for, metrics.latency);
# Ok(())
# }
```

`Capabilities::current()` reports the OS version, physical system memory, logical and active CPU counts, thermal state, and available Core ML device types on macOS 14 or later. Neural Engine core count and Metal's recommended GPU working set are device properties, not per-model utilization or guaranteed reservations. Per-model CPU, GPU, Neural Engine and memory attribution is reported as unavailable.

`ProcessMemory::current()` reports resident, wired and physical-footprint bytes for the whole process. This includes unrelated allocations and excludes resources held by other processes; changes cannot be treated as an exact model footprint. `MemoryPressureMonitor` observes system-wide memory-pressure notifications. It reports no event until macOS delivers one, and flags may be coalesced. Notifications are best effort, so they are not a live free-memory gauge or a guaranteed warning before every allocation.

`ModelCache` offers opt-in idle eviction on the thread that owns the models. Call `evict_idle` periodically or when your application decides it is appropriate. Models with an in-flight request, another `Model` clone, or a live `State` are skipped. An unused model ages from insertion; a used model ages from its last finished request. Eviction drops the cache's last model handle, but Core ML and the OS may retain internal caches. The crate does not start an eviction thread or infer per-model memory usage.

```rust,no_run
use coreml_rs::{MemoryPressureMonitor, Model, ModelCache, ProcessMemory};
use std::time::Duration;

# fn observe(model: Model) -> coreml_rs::Result<()> {
let pressure = MemoryPressureMonitor::new();
let mut cache = ModelCache::new();
cache.insert("classifier", model);
println!("process memory: {:?}", ProcessMemory::current()?);
println!("last system pressure event: {:?}", pressure.latest_event());
// Invoke this on the owning thread whenever your policy runs.
let evicted = cache.evict_idle(Duration::from_secs(300));
println!("evicted: {evicted:?}");
# Ok(())
# }
```

The synchronous Core ML prediction API requires serialized use of a model. `Model`, feature values and native arrays are intentionally not `Send` or `Sync`; use separate model instances on separate threads. Raw native access is unsafe and can invalidate these guarantees.
